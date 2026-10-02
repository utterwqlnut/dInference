"""Borsh schemas matching the Rust state + instruction args.

Must be kept in lockstep with programs/dinference/src/state.rs and the
instruction signatures in lib.rs. Any field reorder on the Rust side is a
breaking change here.
"""

from __future__ import annotations

from borsh_construct import (
    CStruct, U8, U16, U32, U64, I64, I32, Vec, Bool, F32, String
)
from construct import Bytes as RawBytes

# ---- Primitives ----

PUBKEY = RawBytes(32)
HASH32 = RawBytes(32)

# ---- Instruction args ----

INIT_CONFIG_ARGS = CStruct(
    "min_provider_stake"       / U64,
    "min_verifier_stake"       / U64,
    "min_bounty_lamports"      / U64,
    "verify_fee_lamports"      / U64,
    "dishonesty_fee_lamports"  / U64,
    "challenger_reward_bps"    / U16,
    "challenge_window_slots"   / U64,
    "vote_window_slots"        / U64,
    "unstake_cooldown_slots"   / U64,
    "seed_rotation_slots"      / U64,
    "unclaim_timeout_slots"    / U64,
    "cos_threshold"            / U16,
    "verifier_committee_size"  / U8,
    "quorum_fraction"          / U16,
    "max_new_tokens"           / U16,
    "initial_seed"             / HASH32,
)

# ---- Account layouts (after 8-byte discriminator) ----

GLOBAL_CONFIG_LAYOUT = CStruct(
    "admin"                    / PUBKEY,
    "min_provider_stake"       / U64,
    "min_verifier_stake"       / U64,
    "min_bounty_lamports"      / U64,
    "verify_fee_lamports"      / U64,
    "dishonesty_fee_lamports"  / U64,
    "challenger_reward_bps"    / U16,
    "challenge_window_slots"   / U64,
    "vote_window_slots"        / U64,
    "unstake_cooldown_slots"   / U64,
    "seed_rotation_slots"      / U64,
    "unclaim_timeout_slots"    / U64,
    "cos_threshold"            / U16,
    "verifier_committee_size"  / U8,
    "quorum_fraction"          / U16,
    "max_new_tokens"           / U16,
    "bump"                     / U8,
)

ROTATING_SEED_LAYOUT = CStruct(
    "current_seed"     / HASH32,
    "rotated_at_slot"  / U64,
    "bump"             / U8,
)

STAKE_VAULT_LAYOUT = CStruct("bump" / U8)

MODEL_INFO_LAYOUT = CStruct(
    "model_id"    / String,
    "hidden_dim"  / U32,
    "allowed"     / Bool,
    "bump"        / U8,
)

VERIFIER_POOL_LAYOUT = CStruct(
    "model_id_hash"  / HASH32,
    "available"      / Vec(PUBKEY),
    "bump"           / U8,
)

PROVIDER_ACCOUNT_LAYOUT = CStruct(
    "operator"              / PUBKEY,
    "model_id"              / String,
    "stake_lamports"        / U64,
    "reserved_stake"        / U64,
    "unstake_requested_at"  / I64,
    "is_active"             / Bool,
    "bump"                  / U8,
)

VERIFIER_ACCOUNT_LAYOUT = CStruct(
    "operator"              / PUBKEY,
    "model_id"              / String,
    "stake_lamports"        / U64,
    "unstake_requested_at"  / I64,
    "is_active"             / Bool,
    "bump"                  / U8,
)

BOUNTY_LAYOUT = CStruct(
    "user"                 / PUBKEY,
    "model_id_hash"        / HASH32,
    "prompt_txid"          / HASH32,
    "bounty_lamports"      / U64,
    "seed_at_submission"   / HASH32,
    "status"               / U8,     # BountyStatus enum: 0 Open, 1 Claimed, 2 Challenged, 3 Finalized
    "winner"               / PUBKEY,
    "collateral_reserved"  / U64,
    "created_at_slot"      / U64,
    "claimed_at_slot"      / U64,
    "response_commitment"  / PUBKEY,
    "nonce"                / U64,
    "bump"                 / U8,
)

RESPONSE_COMMITMENT_LAYOUT = CStruct(
    "bounty"                   / PUBKEY,
    "provider"                 / PUBKEY,
    "response_txid"            / HASH32,
    "fingerprint"              / Vec(F32),
    "submitted_at_slot"        / U64,
    "challenge_deadline_slot"  / U64,
    "bump"                     / U8,
)

VERIFIER_VOTE_LAYOUT = CStruct(
    "verifier"            / PUBKEY,
    "verdict"             / Bool,
    "fp_cos"              / I32,
    "submitted_at_slot"   / U64,
)

CHALLENGE_LAYOUT = CStruct(
    "bounty"              / PUBKEY,
    "challenger"          / PUBKEY,
    "bond_lamports"       / U64,
    "verifier_committee"  / Vec(PUBKEY),
    "verdicts"            / Vec(VERIFIER_VOTE_LAYOUT),
    "status"              / U8,  # ChallengeStatus: 0 Open, 1 Resolved
    "created_at_slot"     / U64,
    "vote_deadline_slot"  / U64,
    "bump"                / U8,
)

# ---- Event layouts (after 8-byte discriminator) ----

BOUNTY_POSTED_EVENT = CStruct(
    "bounty"           / PUBKEY,
    "model_id_hash"    / HASH32,
    "prompt_txid"      / HASH32,
    "bounty_lamports"  / U64,
)

BOUNTY_CLAIMED_EVENT = CStruct(
    "bounty"         / PUBKEY,
    "winner"         / PUBKEY,
    "response_txid"  / HASH32,
)

CHALLENGE_OPENED_EVENT = CStruct(
    "bounty"     / PUBKEY,
    "committee"  / Vec(PUBKEY),
)

VOTE_SUBMITTED_EVENT = CStruct(
    "bounty"    / PUBKEY,
    "verifier"  / PUBKEY,
    "verdict"   / Bool,
)

CHALLENGE_RESOLVED_EVENT = CStruct(
    "bounty"   / PUBKEY,
    "cheater"  / Bool,
)
