//! Unit tests for pure arithmetic / helper logic. Run under plain `cargo test`
//! without a validator. Integration tests that exercise instructions end-to-end
//! live in `/tests/dinference.ts` and run via `anchor test`.

use super::*;

// ---------------------------------------------------------------------
//  Quorum math
// ---------------------------------------------------------------------

#[test]
fn quorum_math_5_committee_67_percent() {
    let committee_len: u64 = 5;
    let quorum_fraction: u64 = 6700;
    let quorum = (committee_len * quorum_fraction) / 10_000;
    assert_eq!(quorum, 3);
    // guilty >= quorum means >= 3 guilty votes required.
    // With K=5 and quorum=3, outcome can be locked at:
    //   3 guilty → cheater (quorum met)
    //   3 honest → honest (guilty + remaining < 3)
}

#[test]
fn quorum_math_3_committee_67_percent() {
    let committee_len: u64 = 3;
    let quorum_fraction: u64 = 6700;
    let quorum = (committee_len * quorum_fraction) / 10_000;
    assert_eq!(quorum, 2);
}

#[test]
fn quorum_math_7_committee_67_percent() {
    let committee_len: u64 = 7;
    let quorum_fraction: u64 = 6700;
    let quorum = (committee_len * quorum_fraction) / 10_000;
    assert_eq!(quorum, 4);
}

#[test]
fn quorum_math_edge_51_percent() {
    let committee_len: u64 = 5;
    let quorum_fraction: u64 = 5100;
    let quorum = (committee_len * quorum_fraction) / 10_000;
    assert_eq!(quorum, 2); // 51% of 5 = 2.55, floor to 2
}

// ---------------------------------------------------------------------
//  Early-finalize decidability
// ---------------------------------------------------------------------

fn is_decidable(committee_len: u64, guilty: u64, honest: u64, quorum: u64) -> (bool, bool) {
    let remaining = committee_len.saturating_sub(guilty + honest);
    let guilty_locked = guilty >= quorum;
    let honest_locked = guilty + remaining < quorum;
    (guilty_locked, honest_locked)
}

#[test]
fn early_finalize_guilty_lock() {
    // Committee 5, quorum 4. Guilty count hits 4 → done regardless of remaining.
    let (g, h) = is_decidable(5, 4, 0, 4);
    assert!(g && !h);
    let (g, h) = is_decidable(5, 4, 1, 4);
    assert!(g && !h);
}

#[test]
fn early_finalize_honest_lock() {
    // Committee 5, quorum 4. Once honest reaches 2, guilty+remaining=3<4 → honest wins.
    let (g, h) = is_decidable(5, 0, 2, 4);
    assert!(!g && h);
    let (g, h) = is_decidable(5, 1, 2, 4);
    assert!(!g && h);
}

#[test]
fn early_finalize_not_yet_decided() {
    // 2 guilty, 1 honest, 2 remaining — could go either way.
    let (g, h) = is_decidable(5, 2, 1, 4);
    assert!(!g && !h);
}

#[test]
fn early_finalize_all_voted_tie_edge() {
    // 3 guilty, 2 honest at K=5, quorum=4. Honest wins (guilty < 4).
    let (g, h) = is_decidable(5, 3, 2, 4);
    assert!(!g && h);
}

// ---------------------------------------------------------------------
//  Slash / payout arithmetic
// ---------------------------------------------------------------------

#[test]
fn slash_half_math() {
    let stake: u64 = 10_000_000_000;
    let slash = stake / 2;
    assert_eq!(slash, 5_000_000_000);
    assert_eq!(stake.saturating_sub(slash), 5_000_000_000);
}

#[test]
fn slash_half_saturating_odd() {
    let stake: u64 = 99;
    let slash = stake / 2;
    assert_eq!(slash, 49);
    assert_eq!(stake.saturating_sub(slash), 50);
}

#[test]
fn slash_half_leaves_zero_when_tiny() {
    let stake: u64 = 1;
    let slash = stake / 2;
    assert_eq!(slash, 0);
    assert_eq!(stake.saturating_sub(slash), 1);
}

#[test]
fn dishonesty_slash_capped_at_stake() {
    let stake: u64 = 100;
    let fee: u64 = 1000;
    let slash = fee.min(stake);
    assert_eq!(slash, 100);
    assert_eq!(stake.saturating_sub(slash), 0);
}

// ---------------------------------------------------------------------
//  Challenger reward (bps-scaled math)
// ---------------------------------------------------------------------

fn challenger_reward(bounty: u64, bps: u16) -> u64 {
    (bounty as u128 * bps as u128 / 10_000) as u64
}

#[test]
fn challenger_reward_default_125bps() {
    assert_eq!(challenger_reward(1_000_000, 12_500), 1_250_000);
}

#[test]
fn challenger_reward_1x_passthrough() {
    assert_eq!(challenger_reward(1_000_000, 10_000), 1_000_000);
}

#[test]
fn challenger_reward_large_bounty_no_overflow() {
    // 1 billion SOL in lamports (way above total supply).
    let b: u64 = 1_000_000_000 * 1_000_000_000;
    let r = challenger_reward(b, 12_500);
    assert_eq!(r, 1_250_000_000 * 1_000_000_000);
}

// ---------------------------------------------------------------------
//  Reservation formula: 2 × (premium + verify_pool)
// ---------------------------------------------------------------------

fn per_bounty_reservation(bounty: u64, reward_bps: u16, verify_fee: u64, committee_size: u8) -> u64 {
    let premium_bps = (reward_bps as u128).saturating_sub(10_000);
    let premium = (bounty as u128 * premium_bps / 10_000) as u64;
    let verify_pool = verify_fee * committee_size as u64;
    (premium + verify_pool) * 2
}

#[test]
fn reservation_default_125bps_5_committee() {
    // bounty = 1 SOL, verify_fee = 0.001 SOL, K = 5
    // premium = 0.25 SOL, verify_pool = 0.005 SOL
    // reservation = 2 × (0.25 + 0.005) SOL = 0.51 SOL
    let r = per_bounty_reservation(1_000_000_000, 12_500, 1_000_000, 5);
    assert_eq!(r, 510_000_000);
}

#[test]
fn reservation_scales_linearly_with_bounty() {
    let a = per_bounty_reservation(1_000, 12_500, 100, 3);
    let b = per_bounty_reservation(2_000, 12_500, 100, 3);
    // Both have same verify_pool contribution; premium doubles.
    let premium_delta = (b - a) / 2;
    assert_eq!(premium_delta, 2_000 / 4 - 1_000 / 4);
}

#[test]
fn reservation_zero_premium_when_1x_reward() {
    // If challenger reward is 100% (no premium), reservation = 2 × verify_pool only.
    let r = per_bounty_reservation(1_000_000, 10_000, 100, 5);
    assert_eq!(r, 2 * 500); // just 2 × verify_pool
}

#[test]
fn stake_covers_slash_exactly_at_reservation_floor() {
    // Provider staked exactly at reservation. slash = stake / 2. Need slash >= premium.
    let bounty: u64 = 1_000_000;
    let bps: u16 = 12_500;
    let verify_fee: u64 = 100;
    let k: u8 = 5;
    let reservation = per_bounty_reservation(bounty, bps, verify_fee, k);
    let slash = reservation / 2;
    let premium = (bounty as u128 * (bps - 10_000) as u128 / 10_000) as u64;
    // slash should cover premium with headroom equal to verify_pool
    assert!(slash >= premium);
    assert_eq!(slash - premium, verify_fee as u64 * k as u64);
}

// ---------------------------------------------------------------------
//  Verify pool bond math
// ---------------------------------------------------------------------

#[test]
fn bond_covers_verify_pool_exactly() {
    let verify_fee: u64 = 100_000;
    let k: u8 = 5;
    let min_bond = verify_fee * k as u64;
    assert_eq!(min_bond, 500_000);
    // Each voter gets exactly verify_fee → total paid = min_bond, no surplus.
    let total_paid_if_all_vote_majority = verify_fee * k as u64;
    assert_eq!(min_bond, total_paid_if_all_vote_majority);
}

#[test]
fn bond_surplus_when_some_in_minority() {
    // 5-committee, bond = 5 × verify_fee. If only 3 voted majority, 2 verify_fees
    // worth sits in vault as protocol surplus.
    let verify_fee: u64 = 100_000;
    let k: u8 = 5;
    let min_bond = verify_fee * k as u64;
    let majority_paid = verify_fee * 3;
    let surplus = min_bond - majority_paid;
    assert_eq!(surplus, 2 * 100_000);
}

// ---------------------------------------------------------------------
//  Reservation release
// ---------------------------------------------------------------------

#[test]
fn reservation_release_full_lifecycle() {
    let mut stake: u64 = 10_000_000;
    let mut reserved: u64 = 0;

    // Claim bounty with reservation 2M.
    let per_bounty: u64 = 2_000_000;
    reserved += per_bounty;
    assert_eq!(stake - reserved, 8_000_000);

    // Finalize payment (honest) → reservation released.
    reserved = reserved.saturating_sub(per_bounty);
    assert_eq!(reserved, 0);
    assert_eq!(stake - reserved, 10_000_000);

    // Claim another, cheater case.
    reserved += per_bounty;
    let slash = stake / 2;
    stake = stake.saturating_sub(slash);
    // Challenger reward payout from vault (not from stake directly) = premium.
    // Reservation released after finalize_challenge.
    reserved = reserved.saturating_sub(per_bounty);
    assert_eq!(stake, 5_000_000);
    assert_eq!(reserved, 0);
}

#[test]
fn multi_bounty_reservation_prevents_overclaim() {
    let stake: u64 = 1_000_000;
    let per_bounty: u64 = 300_000;
    let mut reserved: u64 = 0;

    // Three claims OK.
    for _ in 0..3 {
        let available = stake.saturating_sub(reserved);
        assert!(available >= per_bounty);
        reserved += per_bounty;
    }
    // Fourth should be rejected.
    let available = stake.saturating_sub(reserved);
    assert!(available < per_bounty);
}

// ---------------------------------------------------------------------
//  EMA (kept from earlier; still used nowhere currently but useful if we re-add)
// ---------------------------------------------------------------------

#[test]
fn ema_convergence() {
    // Integer-division EMA with α=1/8 converges to floor of fixed point.
    // Steady state satisfies ema = (ema*7 + sample) / 8 → ema ≈ sample.
    // Due to integer floor, convergence is a few units below `sample`.
    let mut ema: u32 = 0;
    for _ in 0..100 {
        ema = ((ema as u64 * 7 + 50) / 8) as u32;
    }
    assert!(ema >= 40 && ema <= 50, "ema should converge near 50, got {}", ema);
}

// ---------------------------------------------------------------------
//  Pool operations (swap_remove semantics)
// ---------------------------------------------------------------------

#[test]
fn pool_swap_remove_is_o1_but_reorders() {
    let mut pool = vec![1, 2, 3, 4, 5];
    // Remove index 1 (value 2) via swap_remove.
    let removed = pool.swap_remove(1);
    assert_eq!(removed, 2);
    // Pool now has last element in the removed slot.
    assert_eq!(pool, vec![1, 5, 3, 4]);
}

#[test]
fn pool_random_select_deterministic() {
    // Simulates committee selection: for i in 0..K, hash → index, swap_remove.
    // Same inputs must produce the same committee.
    fn select(pool: &mut Vec<u8>, seed: u8, k: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            // Deterministic "hash": just use (seed + i) % pool.len()
            let r = ((seed as usize + i) % pool.len());
            out.push(pool[r]);
            pool.swap_remove(r);
        }
        out
    }

    let mut p1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut p2 = p1.clone();
    let c1 = select(&mut p1, 42, 3);
    let c2 = select(&mut p2, 42, 3);
    assert_eq!(c1, c2);
}

// ---------------------------------------------------------------------
//  Vault invariants (abstract check)
// ---------------------------------------------------------------------

#[test]
fn vault_invariant_cheater_case() {
    // Model: provider stakes S, user escrows B, challenger bonds verify_pool.
    // On cheater: pay challenger 1.25B, pay K voters verify_pool total.
    // Net vault = S + B + verify_pool - 1.25B - verify_pool = S - 0.25B
    let s: i64 = 10_000;
    let b: i64 = 1_000;
    let verify_pool: i64 = 100;
    let net = s + b + verify_pool - (125 * b / 100) - verify_pool;
    assert_eq!(net, s - b / 4);
    assert!(net > 0, "vault must stay solvent as long as S > 0.25B");
}

#[test]
fn vault_invariant_honest_case() {
    // Honest: pay provider B, pay voters verify_pool total.
    // Net vault contribution = S + B + verify_pool - B - verify_pool = S.
    let s: i64 = 10_000;
    let b: i64 = 1_000;
    let verify_pool: i64 = 100;
    let net = s + b + verify_pool - b - verify_pool;
    assert_eq!(net, s, "honest case leaves provider stake fully in vault");
}
