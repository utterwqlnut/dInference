use anchor_lang::prelude::*;

use crate::constants::*;

// =========================================================================
//  Protocol singletons
// =========================================================================

#[account]
pub struct GlobalConfig {
    pub admin: Pubkey,
    pub min_provider_stake: u64,
    pub min_verifier_stake: u64,
    pub min_bounty_lamports: u64,
    pub challenge_window_slots: u64,
    pub vote_window_slots: u64,
    pub unstake_cooldown_slots: u64,
    pub seed_rotation_slots: u64,
    pub unclaim_timeout_slots: u64,
    pub verify_fee_lamports: u64,      // paid to voters who agreed with majority
    pub dishonesty_fee_lamports: u64,  // slashed from voters who disagreed OR didn't vote
    pub challenger_reward_bps: u16,    // scaled: 12500 = 125% of bounty
    pub cos_threshold: u16,            // scaled: 9900 = 0.99
    pub verifier_committee_size: u8,
    pub quorum_fraction: u16,          // scaled: 6700 = 67%
    pub max_new_tokens: u16,           // client cap (not used for billing)
    pub bump: u8,
}
impl GlobalConfig { pub const SPACE: usize = 8 + 32 + 8*8 + 8*2 + 2 + 2 + 1 + 2 + 2 + 1 + 32; }

#[account]
pub struct RotatingSeed {
    pub current_seed: [u8; 32],
    pub rotated_at_slot: u64,
    pub bump: u8,
}
impl RotatingSeed { pub const SPACE: usize = 8 + 32 + 8 + 1 + 16; }

#[account]
pub struct StakeVault { pub bump: u8 }
impl StakeVault { pub const SPACE: usize = 8 + 1 + 16; }

// =========================================================================
//  Per-model state
// =========================================================================

#[account]
pub struct ModelInfo {
    pub model_id: String,
    pub hidden_dim: u32,
    pub allowed: bool,
    pub bump: u8,
}
impl ModelInfo {
    pub const SPACE: usize = 8 + 4 + MAX_MODEL_ID_LEN + 4 + 1 + 1 + 16;
}

/// Pool of currently-available verifiers for a model. A verifier is added on
/// register, removed when picked for a committee, and re-added when the
/// challenge finalizes (if still active). Enforces one-committee-at-a-time per
/// verifier purely by pool membership.
#[account]
pub struct VerifierPool {
    pub model_id_hash: [u8; 32],
    pub available: Vec<Pubkey>,
    pub bump: u8,
}
impl VerifierPool {
    pub const SPACE: usize =
        8 + 32 + 4 + (32 * MAX_POOL_SIZE) + 1 + 16;
}

// =========================================================================
//  Per-operator state
// =========================================================================

#[account]
pub struct ProviderAccount {
    pub operator: Pubkey,
    pub model_id: String,
    pub stake_lamports: u64,
    pub reserved_stake: u64,           // sum of in-flight bounty collateral
    pub unstake_requested_at: i64,     // 0 = not requested
    pub is_active: bool,
    pub bump: u8,
}
impl ProviderAccount {
    pub const SPACE: usize =
        8 + 32 + 4 + MAX_MODEL_ID_LEN + 8 + 8 + 8 + 1 + 1 + 16;

    pub fn model_id_hash(&self) -> [u8; 32] {
        anchor_lang::solana_program::hash::hash(self.model_id.as_bytes()).to_bytes()
    }
}

#[account]
pub struct VerifierAccount {
    pub operator: Pubkey,
    pub model_id: String,
    pub stake_lamports: u64,
    pub unstake_requested_at: i64,
    pub is_active: bool,
    pub bump: u8,
}
impl VerifierAccount {
    pub const SPACE: usize =
        8 + 32 + 4 + MAX_MODEL_ID_LEN + 8 + 8 + 1 + 1 + 16;

    pub fn model_id_hash(&self) -> [u8; 32] {
        anchor_lang::solana_program::hash::hash(self.model_id.as_bytes()).to_bytes()
    }
}

// =========================================================================
//  Bounty lifecycle
// =========================================================================

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum BountyStatus {
    Open,          // posted, waiting for a provider to claim
    Claimed,      // a provider submitted a response; challenge window counting
    Challenged,   // user filed a challenge; verifier committee voting
    Finalized,    // payment / slashing settled
}

#[account]
pub struct Bounty {
    pub user: Pubkey,
    pub model_id_hash: [u8; 32],
    pub prompt_txid: [u8; 32],              // Arweave txid of the prompt
    pub bounty_lamports: u64,
    pub seed_at_submission: [u8; 32],       // snapshot of rotating seed for fingerprint P
    pub status: BountyStatus,
    pub winner: Pubkey,                     // Pubkey::default() until claimed
    pub collateral_reserved: u64,           // amount of winner.reserved_stake tied up here
    pub created_at_slot: u64,
    pub claimed_at_slot: u64,               // 0 until claimed
    pub response_commitment: Pubkey,        // Pubkey::default() until claimed
    pub nonce: u64,
    pub bump: u8,
}
impl Bounty {
    pub const SPACE: usize =
        8 + 32 + 32 + 32 + 8 + 32 + 1 + 32 + 8 + 8 + 8 + 32 + 8 + 1 + 16;
}

#[account]
pub struct ResponseCommitment {
    pub bounty: Pubkey,
    pub provider: Pubkey,
    pub response_txid: [u8; 32],            // Arweave txid of the response
    pub fingerprint: Vec<f32>,              // 64 floats
    pub submitted_at_slot: u64,
    pub challenge_deadline_slot: u64,
    pub bump: u8,
}
impl ResponseCommitment {
    pub const SPACE: usize =
        8 + 32 + 32 + 32 + 4 + (4 * MAX_FINGERPRINT_DIM) + 8 + 8 + 1 + 16;
}

// =========================================================================
//  Challenge lifecycle
// =========================================================================

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeStatus { Open, Resolved }

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct VerifierVote {
    pub verifier: Pubkey,
    pub verdict: bool,                 // true = provider cheated
    pub fp_cos: i32,                   // scaled: 9900 = 0.99
    pub submitted_at_slot: u64,
}

#[account]
pub struct Challenge {
    pub bounty: Pubkey,
    pub challenger: Pubkey,
    pub bond_lamports: u64,
    pub verifier_committee: Vec<Pubkey>,
    pub verdicts: Vec<VerifierVote>,
    pub status: ChallengeStatus,
    pub created_at_slot: u64,
    pub vote_deadline_slot: u64,
    pub bump: u8,
}
impl Challenge {
    pub const VOTE_SIZE: usize = 32 + 1 + 4 + 8;
    pub const SPACE: usize =
        8 + 32 + 32 + 8
        + 4 + (32 * MAX_COMMITTEE_SIZE)
        + 4 + (Self::VOTE_SIZE * MAX_VOTES)
        + 1 + 8 + 8 + 1 + 16;
}
