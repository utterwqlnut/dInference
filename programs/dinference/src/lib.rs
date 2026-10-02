//! dInference — decentralized LLM inference marketplace on Solana (bounty edition).
//!
//! Protocol summary:
//! - User posts a bounty on-chain with (prompt_txid, bounty_lamports).
//! - Providers watching the chain race to call `claim_bounty` — first valid claim wins.
//! - After a challenge window, winner is paid the whole bounty.
//! - If challenged, a deterministic verifier committee runs the fingerprint check
//!   and votes; majority wins, losing side (provider or verifiers) gets slashed.
//!
//! No queue, no availability pool, no matching, no priority fees, no per-token
//! math. Pure race-to-first-valid-submission.

use anchor_lang::prelude::*;
use anchor_lang::solana_program::{hash::hash, system_instruction};

pub mod constants;
pub mod errors;
pub mod state;

#[cfg(test)]
mod tests;

use constants::*;
use errors::DInferenceError;
use state::*;

// Placeholder; replace with the real program id after `anchor keys sync`.
declare_id!("11111111111111111111111111111111");

#[program]
pub mod dinference {
    use super::*;

    // ---------------------------------------------------------------------
    //  Admin init
    // ---------------------------------------------------------------------

    pub fn init_config(ctx: Context<InitConfig>, args: InitConfigArgs) -> Result<()> {
        // Sanity: min stakes must at least cover the worst-case single-event slash
        // so a freshly-registered operator can't drop below zero on first slash.
        require!(
            args.min_verifier_stake >= args.dishonesty_fee_lamports,
            DInferenceError::StakeTooLow
        );
        let cfg = &mut ctx.accounts.config;
        cfg.admin = ctx.accounts.admin.key();
        cfg.min_provider_stake = args.min_provider_stake;
        cfg.min_verifier_stake = args.min_verifier_stake;
        cfg.min_bounty_lamports = args.min_bounty_lamports;
        cfg.verify_fee_lamports = args.verify_fee_lamports;
        cfg.dishonesty_fee_lamports = args.dishonesty_fee_lamports;
        cfg.challenger_reward_bps = args.challenger_reward_bps;
        cfg.challenge_window_slots = args.challenge_window_slots;
        cfg.vote_window_slots = args.vote_window_slots;
        cfg.unstake_cooldown_slots = args.unstake_cooldown_slots;
        cfg.seed_rotation_slots = args.seed_rotation_slots;
        cfg.unclaim_timeout_slots = args.unclaim_timeout_slots;
        cfg.cos_threshold = args.cos_threshold;
        cfg.verifier_committee_size = args.verifier_committee_size;
        cfg.quorum_fraction = args.quorum_fraction;
        cfg.max_new_tokens = args.max_new_tokens;
        cfg.bump = ctx.bumps.config;

        let s = &mut ctx.accounts.rotating_seed;
        s.current_seed = args.initial_seed;
        s.rotated_at_slot = Clock::get()?.slot;
        s.bump = ctx.bumps.rotating_seed;

        ctx.accounts.stake_vault.bump = ctx.bumps.stake_vault;
        Ok(())
    }

    pub fn init_model(ctx: Context<InitModel>, model_id: String, hidden_dim: u32) -> Result<()> {
        require!(model_id.len() <= MAX_MODEL_ID_LEN, DInferenceError::ModelStringTooLong);
        require_keys_eq!(ctx.accounts.admin.key(), ctx.accounts.config.admin, DInferenceError::AdminOnly);
        let id_hash = hash(model_id.as_bytes()).to_bytes();

        let m = &mut ctx.accounts.model;
        m.model_id = model_id;
        m.hidden_dim = hidden_dim;
        m.allowed = true;
        m.bump = ctx.bumps.model;

        let pool = &mut ctx.accounts.pool;
        pool.model_id_hash = id_hash;
        pool.available = Vec::new();
        pool.bump = ctx.bumps.pool;
        Ok(())
    }

    // ---------------------------------------------------------------------
    //  Registration + staking
    // ---------------------------------------------------------------------

    pub fn register_provider(ctx: Context<RegisterProvider>, model_id: String, stake: u64) -> Result<()> {
        require!(stake >= ctx.accounts.config.min_provider_stake, DInferenceError::StakeTooLow);
        require!(ctx.accounts.model.allowed, DInferenceError::ModelNotAllowed);
        require!(ctx.accounts.model.model_id == model_id, DInferenceError::ModelNotAllowed);

        transfer_lamports(
            &ctx.accounts.operator.to_account_info(),
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.system_program,
            stake,
        )?;

        let p = &mut ctx.accounts.provider;
        p.operator = ctx.accounts.operator.key();
        p.model_id = model_id;
        p.stake_lamports = stake;
        p.reserved_stake = 0;
        p.unstake_requested_at = 0;
        p.is_active = true;
        p.bump = ctx.bumps.provider;
        Ok(())
    }

    pub fn register_verifier(ctx: Context<RegisterVerifier>, model_id: String, stake: u64) -> Result<()> {
        let cfg = &ctx.accounts.config;
        require!(stake >= cfg.min_verifier_stake, DInferenceError::StakeTooLow);
        // Belt-and-suspenders: enforce stake can cover at least one dishonesty slash.
        // (This is already implied by the init_config check `min_verifier_stake >= dishonesty_fee`,
        //  but we re-check in case governance ever lowers min_verifier_stake.)
        require!(stake >= cfg.dishonesty_fee_lamports, DInferenceError::StakeTooLow);
        require!(ctx.accounts.model.allowed, DInferenceError::ModelNotAllowed);
        require!(ctx.accounts.model.model_id == model_id, DInferenceError::ModelNotAllowed);

        transfer_lamports(
            &ctx.accounts.operator.to_account_info(),
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.system_program,
            stake,
        )?;

        let v = &mut ctx.accounts.verifier;
        v.operator = ctx.accounts.operator.key();
        v.model_id = model_id;
        v.stake_lamports = stake;
        v.unstake_requested_at = 0;
        v.is_active = true;
        v.bump = ctx.bumps.verifier;

        let pool = &mut ctx.accounts.pool;
        require!(pool.available.len() < MAX_POOL_SIZE, DInferenceError::QueueFull);
        pool.available.push(v.operator);
        Ok(())
    }

    pub fn request_unstake_provider(ctx: Context<ProviderUnstakeReq>) -> Result<()> {
        let p = &mut ctx.accounts.provider;
        require!(p.unstake_requested_at == 0, DInferenceError::UnstakeAlreadyRequested);
        p.is_active = false;
        p.unstake_requested_at = Clock::get()?.slot as i64;
        Ok(())
    }

    pub fn claim_unstake_provider(ctx: Context<ProviderUnstakeClaim>) -> Result<()> {
        let p = &mut ctx.accounts.provider;
        require!(p.unstake_requested_at != 0, DInferenceError::UnstakeNotRequested);
        // Withdrawable = stake − reserved_stake. The reserved portion is locked
        // backing in-flight bounties and only releases when those bounties
        // finalize. Call this instruction repeatedly as collateral frees up.
        let withdrawable = p.stake_lamports.saturating_sub(p.reserved_stake);
        require!(withdrawable > 0, DInferenceError::UnstakeCooldownActive);
        vault_pay(
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.operator.to_account_info(),
            withdrawable,
        )?;
        p.stake_lamports = p.stake_lamports.saturating_sub(withdrawable);
        Ok(())
    }

    pub fn request_unstake_verifier(ctx: Context<VerifierUnstakeReq>) -> Result<()> {
        let v = &mut ctx.accounts.verifier;
        require!(v.unstake_requested_at == 0, DInferenceError::UnstakeAlreadyRequested);
        v.is_active = false;
        v.unstake_requested_at = Clock::get()?.slot as i64;

        // Remove from the available pool (noop if not currently in it, e.g. if
        // they're currently seated on an open committee).
        let pool = &mut ctx.accounts.pool;
        if let Some(idx) = pool.available.iter().position(|k| k == &v.operator) {
            pool.available.swap_remove(idx);
        }
        Ok(())
    }

    pub fn claim_unstake_verifier(ctx: Context<VerifierUnstakeClaim>) -> Result<()> {
        let v = &mut ctx.accounts.verifier;
        require!(v.unstake_requested_at != 0, DInferenceError::UnstakeNotRequested);
        // Verifiers aren't on committees once request_unstake flips is_active=false
        // (and removes them from the pool), so their full stake is immediately
        // withdrawable. Any in-flight committees they were seated on before
        // unstake can still slash dishonesty fees; stake drops accordingly and
        // this instruction can be re-called to withdraw whatever's left.
        let amount = v.stake_lamports;
        require!(amount > 0, DInferenceError::UnstakeCooldownActive);
        vault_pay(
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.operator.to_account_info(),
            amount,
        )?;
        v.stake_lamports = 0;
        Ok(())
    }

    // ---------------------------------------------------------------------
    //  Bounty lifecycle
    // ---------------------------------------------------------------------

    pub fn post_bounty(
        ctx: Context<PostBounty>,
        nonce: u64,
        prompt_txid: [u8; 32],
        bounty_lamports: u64,
    ) -> Result<()> {
        require!(ctx.accounts.model.allowed, DInferenceError::ModelNotAllowed);
        require!(bounty_lamports >= ctx.accounts.config.min_bounty_lamports, DInferenceError::BountyTooLow);

        transfer_lamports(
            &ctx.accounts.user.to_account_info(),
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.system_program,
            bounty_lamports,
        )?;

        let now = Clock::get()?.slot;
        let b = &mut ctx.accounts.bounty;
        b.user = ctx.accounts.user.key();
        b.model_id_hash = hash(ctx.accounts.model.model_id.as_bytes()).to_bytes();
        b.prompt_txid = prompt_txid;
        b.bounty_lamports = bounty_lamports;
        b.seed_at_submission = ctx.accounts.rotating_seed.current_seed;
        b.status = BountyStatus::Open;
        b.winner = Pubkey::default();
        b.collateral_reserved = 0;
        b.created_at_slot = now;
        b.claimed_at_slot = 0;
        b.response_commitment = Pubkey::default();
        b.nonce = nonce;
        b.bump = ctx.bumps.bounty;

        emit!(BountyPosted {
            bounty: b.key(),
            model_id_hash: b.model_id_hash,
            prompt_txid,
            bounty_lamports,
        });
        Ok(())
    }

    /// Provider races to call this. First valid tx to land wins the bounty.
    /// Subsequent attempts hit `WrongBountyStatus` (status flipped to Claimed)
    /// and revert.
    pub fn claim_bounty(
        ctx: Context<ClaimBounty>,
        response_txid: [u8; 32],
        fingerprint: Vec<f32>,
    ) -> Result<()> {
        require!(fingerprint.len() <= MAX_FINGERPRINT_DIM, DInferenceError::FingerprintTooLarge);

        let b = &mut ctx.accounts.bounty;
        let p = &mut ctx.accounts.provider;
        let cfg = &ctx.accounts.config;
        require!(b.status == BountyStatus::Open, DInferenceError::WrongBountyStatus);
        require!(p.is_active, DInferenceError::OperatorInactive);
        require!(p.model_id_hash() == b.model_id_hash, DInferenceError::OperatorModelMismatch);

        // Required reservation per bounty = 2 × (challenger_premium + verify_pool)
        //   challenger_premium = (challenger_reward_bps - 10_000) / 10_000 × bounty
        //                      = extra above the user's own escrow (default 0.25×)
        //   verify_pool = verify_fee × committee_size (the challenger's required bond)
        // 2× because only half of stake is slashed on cheat.
        let premium_bps = (cfg.challenger_reward_bps as u128).saturating_sub(10_000);
        let premium = (b.bounty_lamports as u128 * premium_bps / 10_000) as u64;
        let verify_pool = cfg.verify_fee_lamports
            .checked_mul(cfg.verifier_committee_size as u64)
            .ok_or(DInferenceError::MathOverflow)?;
        let per_bounty = premium
            .checked_add(verify_pool).ok_or(DInferenceError::MathOverflow)?
            .checked_mul(2).ok_or(DInferenceError::MathOverflow)?;

        let available = p.stake_lamports.saturating_sub(p.reserved_stake);
        require!(available >= per_bounty, DInferenceError::StakeInsufficientForBounty);
        p.reserved_stake = p.reserved_stake
            .checked_add(per_bounty).ok_or(DInferenceError::MathOverflow)?;
        b.collateral_reserved = per_bounty;

        let now = Clock::get()?.slot;
        b.status = BountyStatus::Claimed;
        b.winner = p.operator;
        b.claimed_at_slot = now;

        let c = &mut ctx.accounts.commitment;
        c.bounty = b.key();
        c.provider = p.operator;
        c.response_txid = response_txid;
        c.fingerprint = fingerprint;
        c.submitted_at_slot = now;
        c.challenge_deadline_slot = now
            .checked_add(ctx.accounts.config.challenge_window_slots)
            .ok_or(DInferenceError::MathOverflow)?;
        c.bump = ctx.bumps.commitment;

        b.response_commitment = c.key();

        emit!(BountyClaimed {
            bounty: b.key(),
            winner: p.operator,
            response_txid,
        });
        Ok(())
    }

    /// Permissionless. Called after challenge window with no challenge.
    /// Pays the full bounty to the winning provider.
    pub fn finalize_payment(ctx: Context<FinalizePayment>) -> Result<()> {
        let b = &mut ctx.accounts.bounty;
        let c = &ctx.accounts.commitment;
        let p = &mut ctx.accounts.provider;
        require!(b.status == BountyStatus::Claimed, DInferenceError::WrongBountyStatus);
        require!(Clock::get()?.slot > c.challenge_deadline_slot, DInferenceError::ChallengeWindowOpen);

        vault_pay(
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.winner_wallet.to_account_info(),
            b.bounty_lamports,
        )?;
        p.reserved_stake = p.reserved_stake.saturating_sub(b.collateral_reserved);
        b.status = BountyStatus::Finalized;
        Ok(())
    }

    /// User reclaims their bounty if no one has claimed it after the unclaim
    /// timeout. Account closes, SOL returned.
    pub fn reclaim_unclaimed_bounty(ctx: Context<ReclaimUnclaimed>) -> Result<()> {
        let b = &mut ctx.accounts.bounty;
        require!(b.status == BountyStatus::Open, DInferenceError::WrongBountyStatus);
        let now = Clock::get()?.slot;
        require!(
            now > b.created_at_slot + ctx.accounts.config.unclaim_timeout_slots,
            DInferenceError::UnclaimTimeoutActive
        );
        vault_pay(
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.user.to_account_info(),
            b.bounty_lamports,
        )?;
        b.status = BountyStatus::Finalized;
        Ok(())
    }

    // ---------------------------------------------------------------------
    //  Challenge path
    // ---------------------------------------------------------------------

    pub fn challenge(ctx: Context<ChallengeIx>, bond: u64) -> Result<()> {
        let cfg = &ctx.accounts.config;
        // Bond must exactly cover the verifier fee pool (verify_fee × committee).
        let verify_pool = cfg.verify_fee_lamports
            .checked_mul(cfg.verifier_committee_size as u64)
            .ok_or(DInferenceError::MathOverflow)?;
        require!(bond >= verify_pool, DInferenceError::BondTooLow);
        let b = &mut ctx.accounts.bounty;
        // Only the user who posted the bounty can challenge their own response.
        require_keys_eq!(
            ctx.accounts.challenger.key(), b.user,
            DInferenceError::NotOnCommittee
        );
        let c = &ctx.accounts.commitment;
        require!(b.status == BountyStatus::Claimed, DInferenceError::WrongBountyStatus);
        let now = Clock::get()?.slot;
        require!(now <= c.challenge_deadline_slot, DInferenceError::ChallengeWindowClosed);

        transfer_lamports(
            &ctx.accounts.challenger.to_account_info(),
            &ctx.accounts.stake_vault.to_account_info(),
            &ctx.accounts.system_program,
            bond,
        )?;

        // Random committee selection. For each of K slots, hash
        // (seed || bounty || iteration) to a pool index, swap_remove it,
        // and record. O(K) hashes, no sort, no allocation.
        let k = cfg.verifier_committee_size as usize;
        let pool = &mut ctx.accounts.pool;
        require!(pool.available.len() >= k, DInferenceError::InsufficientVerifiers);

        let seed = ctx.accounts.rotating_seed.current_seed;
        let bounty_bytes = b.key().to_bytes();
        let mut buf = [0u8; 72];
        buf[..32].copy_from_slice(&seed);
        buf[32..64].copy_from_slice(&bounty_bytes);

        let mut committee: Vec<Pubkey> = Vec::with_capacity(k);
        for i in 0..k {
            buf[64..].copy_from_slice(&(i as u64).to_le_bytes());
            let h = hash(&buf).to_bytes();
            let r = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize
                % pool.available.len();
            committee.push(pool.available[r]);
            pool.available.swap_remove(r);
        }

        let ch = &mut ctx.accounts.challenge;
        ch.bounty = b.key();
        ch.challenger = ctx.accounts.challenger.key();
        ch.bond_lamports = bond;
        ch.verifier_committee = committee.clone();
        ch.verdicts = Vec::new();
        ch.status = ChallengeStatus::Open;
        ch.created_at_slot = now;
        ch.vote_deadline_slot = now + ctx.accounts.config.vote_window_slots;
        ch.bump = ctx.bumps.challenge;

        b.status = BountyStatus::Challenged;

        emit!(ChallengeOpened { bounty: b.key(), committee });
        Ok(())
    }

    pub fn verifier_vote(ctx: Context<VerifierVoteIx>, verdict: bool, fp_cos_scaled: i32) -> Result<()> {
        let ch = &mut ctx.accounts.challenge;
        require!(ch.status == ChallengeStatus::Open, DInferenceError::ChallengeNotOpen);
        let now = Clock::get()?.slot;
        require!(now <= ch.vote_deadline_slot, DInferenceError::VoteWindowClosed);
        require!(ch.verifier_committee.contains(&ctx.accounts.voter.key()), DInferenceError::NotOnCommittee);
        require!(
            !ch.verdicts.iter().any(|v| v.verifier == ctx.accounts.voter.key()),
            DInferenceError::AlreadyVoted
        );
        ch.verdicts.push(VerifierVote {
            verifier: ctx.accounts.voter.key(),
            verdict,
            fp_cos: fp_cos_scaled,
            submitted_at_slot: now,
        });
        emit!(VoteSubmitted { bounty: ch.bounty, verifier: ctx.accounts.voter.key(), verdict });
        Ok(())
    }

    pub fn finalize_challenge<'info>(
        ctx: Context<'_, '_, 'info, 'info, FinalizeChallenge<'info>>,
    ) -> Result<()> {
        let ch = &mut ctx.accounts.challenge;
        let b = &mut ctx.accounts.bounty;
        let provider = &mut ctx.accounts.provider;
        let cfg = &ctx.accounts.config;
        require!(ch.status == ChallengeStatus::Open, DInferenceError::ChallengeNotOpen);
        let now = Clock::get()?.slot;

        // Early-finalize check: resolve the moment the outcome can no longer flip,
        // instead of waiting for every verifier to vote or for the deadline.
        let committee_len = ch.verifier_committee.len() as u64;
        let guilty_so_far = ch.verdicts.iter().filter(|v| v.verdict).count() as u64;
        let honest_so_far = ch.verdicts.iter().filter(|v| !v.verdict).count() as u64;
        let remaining     = committee_len.saturating_sub(guilty_so_far + honest_so_far);
        let quorum        = (committee_len * cfg.quorum_fraction as u64) / 10_000;
        let guilty_locked = guilty_so_far >= quorum;
        let honest_locked = guilty_so_far + remaining < quorum;
        require!(
            guilty_locked || honest_locked || now > ch.vote_deadline_slot,
            DInferenceError::VotesNotReady
        );

        let cheater = guilty_so_far >= quorum;
        let majority_verdict = cheater;
        // Challenger == bounty.user (enforced at challenge-open time).

        // Walk the full committee. For each member:
        //   - voted with majority → credit verify_fee to their VerifierAccount
        //   - voted against majority OR did not vote → slash dishonesty_fee
        // remaining_accounts must contain the K VerifierAccount PDAs in the
        // same order as ch.verifier_committee.
        require!(
            ctx.remaining_accounts.len() == ch.verifier_committee.len(),
            DInferenceError::NotOnCommittee
        );
        for (member, acct) in ch.verifier_committee.iter().zip(ctx.remaining_accounts.iter()) {
            let mut verifier_acct: Account<VerifierAccount> = Account::try_from(acct)?;
            require_keys_eq!(verifier_acct.operator, *member, DInferenceError::NotOnCommittee);

            let vote = ch.verdicts.iter().find(|v| v.verifier == *member);
            match vote {
                Some(v) if v.verdict == majority_verdict => {
                    // Honest voter: credit verify fee to their stake balance.
                    vault_pay(
                        &ctx.accounts.stake_vault.to_account_info(),
                        acct,
                        cfg.verify_fee_lamports,
                    )?;
                    verifier_acct.stake_lamports = verifier_acct.stake_lamports
                        .saturating_add(cfg.verify_fee_lamports);
                }
                _ => {
                    // Minority voter OR non-voter: slash dishonesty fee.
                    let slash = cfg.dishonesty_fee_lamports.min(verifier_acct.stake_lamports);
                    verifier_acct.stake_lamports = verifier_acct.stake_lamports
                        .saturating_sub(slash);
                    // Dishonesty slash stays in vault (we remove lamports from the
                    // verifier account's bookkeeping; the actual SOL was already
                    // in the vault from registration).
                    if verifier_acct.stake_lamports < cfg.min_verifier_stake {
                        verifier_acct.is_active = false;
                    }
                }
            }
            // Re-add to the pool if they're still active after this challenge.
            if verifier_acct.is_active {
                let pool = &mut ctx.accounts.pool;
                if pool.available.len() < MAX_POOL_SIZE
                    && !pool.available.contains(&verifier_acct.operator)
                {
                    pool.available.push(verifier_acct.operator);
                }
            }
            verifier_acct.exit(ctx.program_id)?;
        }
        // Challenger's bond stays in vault as protocol fee (minus whatever
        // we paid out as verify fees above).
        if cheater {
            // Provider cheated. Slash half their stake. The user (= challenger)
            // receives 1.25x bounty — of which 1x is sourced from their own
            // bounty escrow sitting in the vault and 0.25x comes from the
            // provider slash. Remainder of slash stays in vault.
            let slash = provider.stake_lamports / 2;
            provider.stake_lamports = provider.stake_lamports.saturating_sub(slash);
            if provider.stake_lamports < cfg.min_provider_stake {
                provider.is_active = false;
            }

            let challenger_reward = (b.bounty_lamports as u128
                * cfg.challenger_reward_bps as u128
                / 10_000) as u64;
            vault_pay(
                &ctx.accounts.stake_vault.to_account_info(),
                &ctx.accounts.challenger.to_account_info(),
                challenger_reward,
            )?;
            // No separate user refund — the challenger IS the user.
        } else {
            // Provider honest. Pay full bounty to the winner, no slashing.
            // Challenger's bond is already consumed by verify fees + vault above.
            vault_pay(
                &ctx.accounts.stake_vault.to_account_info(),
                &ctx.accounts.winner_wallet.to_account_info(),
                b.bounty_lamports,
            )?;
        }

        // Release provider's reserved collateral on this bounty — whether they
        // were cheater or honest, the bounty is now settled and the reservation
        // frees up for future claims.
        provider.reserved_stake = provider.reserved_stake.saturating_sub(b.collateral_reserved);

        ch.status = ChallengeStatus::Resolved;
        b.status = BountyStatus::Finalized;
        emit!(ChallengeResolved { bounty: b.key(), cheater });
        Ok(())
    }

    // ---------------------------------------------------------------------
    //  Housekeeping
    // ---------------------------------------------------------------------

    pub fn rotate_seed(ctx: Context<RotateSeed>) -> Result<()> {
        let now = Clock::get()?.slot;
        let s = &mut ctx.accounts.rotating_seed;
        require!(
            now >= s.rotated_at_slot + ctx.accounts.config.seed_rotation_slots,
            DInferenceError::SeedRotationNotReady
        );
        let mut buf = Vec::with_capacity(40);
        buf.extend_from_slice(&s.current_seed);
        buf.extend_from_slice(&now.to_le_bytes());
        s.current_seed = hash(&buf).to_bytes();
        s.rotated_at_slot = now;
        Ok(())
    }
}

// =============================================================================
//  Helpers
// =============================================================================

fn transfer_lamports<'info>(
    from: &AccountInfo<'info>,
    to: &AccountInfo<'info>,
    system_program: &Program<'info, System>,
    amount: u64,
) -> Result<()> {
    let ix = system_instruction::transfer(from.key, to.key, amount);
    anchor_lang::solana_program::program::invoke(
        &ix,
        &[from.clone(), to.clone(), system_program.to_account_info()],
    )?;
    Ok(())
}

/// Transfer out of the vault PDA via direct lamport mutation.
fn vault_pay<'info>(vault: &AccountInfo<'info>, to: &AccountInfo<'info>, amount: u64) -> Result<()> {
    if amount == 0 { return Ok(()); }
    **vault.try_borrow_mut_lamports()? = vault
        .lamports().checked_sub(amount).ok_or(DInferenceError::MathOverflow)?;
    **to.try_borrow_mut_lamports()? = to
        .lamports().checked_add(amount).ok_or(DInferenceError::MathOverflow)?;
    Ok(())
}

// =============================================================================
//  Account contexts
// =============================================================================

#[derive(Accounts)]
pub struct InitConfig<'info> {
    #[account(mut)] pub admin: Signer<'info>,
    #[account(init, payer = admin, space = GlobalConfig::SPACE, seeds = [CONFIG_SEED], bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(init, payer = admin, space = RotatingSeed::SPACE, seeds = [SEED_SEED], bump)]
    pub rotating_seed: Account<'info, RotatingSeed>,
    #[account(init, payer = admin, space = StakeVault::SPACE, seeds = [VAULT_SEED], bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct InitConfigArgs {
    pub min_provider_stake: u64,
    pub min_verifier_stake: u64,
    pub min_bounty_lamports: u64,
    pub verify_fee_lamports: u64,
    pub dishonesty_fee_lamports: u64,
    pub challenger_reward_bps: u16,
    pub challenge_window_slots: u64,
    pub vote_window_slots: u64,
    pub unstake_cooldown_slots: u64,
    pub seed_rotation_slots: u64,
    pub unclaim_timeout_slots: u64,
    pub cos_threshold: u16,
    pub verifier_committee_size: u8,
    pub quorum_fraction: u16,
    pub max_new_tokens: u16,
    pub initial_seed: [u8; 32],
}

#[derive(Accounts)]
#[instruction(model_id: String, hidden_dim: u32)]
pub struct InitModel<'info> {
    #[account(mut)] pub admin: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(
        init, payer = admin, space = ModelInfo::SPACE,
        seeds = [MODEL_SEED, &hash(model_id.as_bytes()).to_bytes()], bump,
    )]
    pub model: Account<'info, ModelInfo>,
    #[account(
        init, payer = admin, space = VerifierPool::SPACE,
        seeds = [POOL_SEED, &hash(model_id.as_bytes()).to_bytes()], bump,
    )]
    pub pool: Account<'info, VerifierPool>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(model_id: String, stake: u64)]
pub struct RegisterProvider<'info> {
    #[account(mut)] pub operator: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(seeds = [MODEL_SEED, &hash(model_id.as_bytes()).to_bytes()], bump = model.bump)]
    pub model: Account<'info, ModelInfo>,
    #[account(
        init, payer = operator, space = ProviderAccount::SPACE,
        seeds = [PROVIDER_SEED, operator.key().as_ref()], bump,
    )]
    pub provider: Account<'info, ProviderAccount>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(model_id: String, stake: u64)]
pub struct RegisterVerifier<'info> {
    #[account(mut)] pub operator: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(seeds = [MODEL_SEED, &hash(model_id.as_bytes()).to_bytes()], bump = model.bump)]
    pub model: Account<'info, ModelInfo>,
    #[account(
        init, payer = operator, space = VerifierAccount::SPACE,
        seeds = [VERIFIER_SEED, operator.key().as_ref()], bump,
    )]
    pub verifier: Account<'info, VerifierAccount>,
    #[account(mut, seeds = [POOL_SEED, &hash(model_id.as_bytes()).to_bytes()], bump = pool.bump)]
    pub pool: Account<'info, VerifierPool>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ProviderUnstakeReq<'info> {
    pub operator: Signer<'info>,
    #[account(mut, seeds = [PROVIDER_SEED, operator.key().as_ref()], bump = provider.bump,
              has_one = operator)]
    pub provider: Account<'info, ProviderAccount>,
}

#[derive(Accounts)]
pub struct ProviderUnstakeClaim<'info> {
    #[account(mut)] pub operator: Signer<'info>,
    #[account(mut, seeds = [PROVIDER_SEED, operator.key().as_ref()], bump = provider.bump,
              has_one = operator)]
    pub provider: Account<'info, ProviderAccount>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub config: Account<'info, GlobalConfig>,
}

#[derive(Accounts)]
pub struct VerifierUnstakeReq<'info> {
    pub operator: Signer<'info>,
    #[account(mut, seeds = [VERIFIER_SEED, operator.key().as_ref()], bump = verifier.bump,
              has_one = operator)]
    pub verifier: Account<'info, VerifierAccount>,
    #[account(mut, seeds = [POOL_SEED, &verifier.model_id_hash()], bump = pool.bump)]
    pub pool: Account<'info, VerifierPool>,
}

#[derive(Accounts)]
pub struct VerifierUnstakeClaim<'info> {
    #[account(mut)] pub operator: Signer<'info>,
    #[account(mut, seeds = [VERIFIER_SEED, operator.key().as_ref()], bump = verifier.bump,
              has_one = operator)]
    pub verifier: Account<'info, VerifierAccount>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub config: Account<'info, GlobalConfig>,
}

#[derive(Accounts)]
#[instruction(nonce: u64, prompt_txid: [u8; 32], bounty_lamports: u64)]
pub struct PostBounty<'info> {
    #[account(mut)] pub user: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    pub model: Account<'info, ModelInfo>,
    pub rotating_seed: Account<'info, RotatingSeed>,
    #[account(
        init, payer = user, space = Bounty::SPACE,
        seeds = [BOUNTY_SEED, user.key().as_ref(), &nonce.to_le_bytes()], bump,
    )]
    pub bounty: Account<'info, Bounty>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClaimBounty<'info> {
    #[account(mut)] pub provider_operator: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(seeds = [PROVIDER_SEED, provider_operator.key().as_ref()], bump = provider.bump,
              has_one = operator @ DInferenceError::OperatorInactive)]
    pub provider: Account<'info, ProviderAccount>,
    /// CHECK: operator pubkey from provider must match signer; enforced by has_one.
    pub operator: AccountInfo<'info>,
    #[account(mut)] pub bounty: Account<'info, Bounty>,
    #[account(
        init, payer = provider_operator, space = ResponseCommitment::SPACE,
        seeds = [RESPONSE_SEED, bounty.key().as_ref()], bump,
    )]
    pub commitment: Account<'info, ResponseCommitment>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FinalizePayment<'info> {
    pub caller: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(mut)] pub bounty: Account<'info, Bounty>,
    #[account(
        seeds = [RESPONSE_SEED, bounty.key().as_ref()], bump = commitment.bump,
        constraint = commitment.bounty == bounty.key() @ DInferenceError::WrongBountyStatus,
    )]
    pub commitment: Account<'info, ResponseCommitment>,
    #[account(
        mut,
        seeds = [PROVIDER_SEED, provider.operator.as_ref()], bump = provider.bump,
        constraint = provider.operator == bounty.winner @ DInferenceError::NotAssignedProvider,
    )]
    pub provider: Account<'info, ProviderAccount>,
    /// CHECK: winning provider's wallet. Must equal bounty.winner.
    #[account(mut, address = bounty.winner @ DInferenceError::NotAssignedProvider)]
    pub winner_wallet: AccountInfo<'info>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
}

#[derive(Accounts)]
pub struct ReclaimUnclaimed<'info> {
    #[account(mut)] pub user: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(mut, has_one = user)]
    pub bounty: Account<'info, Bounty>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
}

#[derive(Accounts)]
pub struct ChallengeIx<'info> {
    #[account(mut)] pub challenger: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(seeds = [SEED_SEED], bump = rotating_seed.bump)]
    pub rotating_seed: Account<'info, RotatingSeed>,
    #[account(mut)] pub bounty: Account<'info, Bounty>,
    #[account(
        seeds = [RESPONSE_SEED, bounty.key().as_ref()], bump = commitment.bump,
        constraint = commitment.bounty == bounty.key() @ DInferenceError::WrongBountyStatus,
    )]
    pub commitment: Account<'info, ResponseCommitment>,
    #[account(mut, seeds = [POOL_SEED, &bounty.model_id_hash], bump = pool.bump)]
    pub pool: Account<'info, VerifierPool>,
    #[account(
        init, payer = challenger, space = Challenge::SPACE,
        seeds = [CHALLENGE_SEED, bounty.key().as_ref()], bump,
    )]
    pub challenge: Account<'info, Challenge>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct VerifierVoteIx<'info> {
    pub voter: Signer<'info>,
    #[account(mut)] pub challenge: Account<'info, Challenge>,
}

#[derive(Accounts)]
pub struct FinalizeChallenge<'info> {
    pub caller: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(
        mut,
        seeds = [CHALLENGE_SEED, bounty.key().as_ref()], bump = challenge.bump,
        constraint = challenge.bounty == bounty.key() @ DInferenceError::ChallengeNotOpen,
    )]
    pub challenge: Account<'info, Challenge>,
    #[account(mut)] pub bounty: Account<'info, Bounty>,
    #[account(
        seeds = [RESPONSE_SEED, bounty.key().as_ref()], bump = commitment.bump,
        constraint = commitment.bounty == bounty.key() @ DInferenceError::WrongBountyStatus,
    )]
    pub commitment: Account<'info, ResponseCommitment>,
    #[account(
        mut,
        seeds = [PROVIDER_SEED, provider.operator.as_ref()], bump = provider.bump,
        constraint = provider.operator == bounty.winner @ DInferenceError::NotAssignedProvider,
    )]
    pub provider: Account<'info, ProviderAccount>,
    #[account(mut, seeds = [POOL_SEED, &bounty.model_id_hash], bump = pool.bump)]
    pub pool: Account<'info, VerifierPool>,
    /// CHECK: winner wallet. Enforced == bounty.winner.
    #[account(mut, address = bounty.winner @ DInferenceError::NotAssignedProvider)]
    pub winner_wallet: AccountInfo<'info>,
    /// CHECK: challenger wallet (= bounty.user; enforced at challenge-open time).
    #[account(mut, address = bounty.user @ DInferenceError::NotOnCommittee)]
    pub challenger: AccountInfo<'info>,
    #[account(mut, seeds = [VAULT_SEED], bump = stake_vault.bump)]
    pub stake_vault: Account<'info, StakeVault>,
}

#[derive(Accounts)]
pub struct RotateSeed<'info> {
    pub caller: Signer<'info>,
    #[account(seeds = [CONFIG_SEED], bump = config.bump)]
    pub config: Account<'info, GlobalConfig>,
    #[account(mut, seeds = [SEED_SEED], bump = rotating_seed.bump)]
    pub rotating_seed: Account<'info, RotatingSeed>,
}

// =============================================================================
//  Events
// =============================================================================

#[event]
pub struct BountyPosted {
    pub bounty: Pubkey,
    pub model_id_hash: [u8; 32],
    pub prompt_txid: [u8; 32],
    pub bounty_lamports: u64,
}

#[event]
pub struct BountyClaimed {
    pub bounty: Pubkey,
    pub winner: Pubkey,
    pub response_txid: [u8; 32],
}

#[event]
pub struct ChallengeOpened {
    pub bounty: Pubkey,
    pub committee: Vec<Pubkey>,
}

#[event]
pub struct VoteSubmitted {
    pub bounty: Pubkey,
    pub verifier: Pubkey,
    pub verdict: bool,
}

#[event]
pub struct ChallengeResolved {
    pub bounty: Pubkey,
    pub cheater: bool,
}
