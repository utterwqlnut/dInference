//! Protocol-wide constants and PDA seeds.

use anchor_lang::prelude::*;

// --- PDA seeds ---
#[constant] pub const VAULT_SEED:     &[u8] = b"vault";
#[constant] pub const CONFIG_SEED:    &[u8] = b"config";
#[constant] pub const SEED_SEED:      &[u8] = b"seed";
#[constant] pub const MODEL_SEED:     &[u8] = b"model";
#[constant] pub const PROVIDER_SEED:  &[u8] = b"provider";
#[constant] pub const VERIFIER_SEED:  &[u8] = b"verifier";
#[constant] pub const BOUNTY_SEED:    &[u8] = b"bounty";
#[constant] pub const RESPONSE_SEED:  &[u8] = b"resp";
#[constant] pub const CHALLENGE_SEED: &[u8] = b"chal";
#[constant] pub const POOL_SEED:      &[u8] = b"pool";

// --- Protocol caps ---
pub const MAX_MODEL_ID_LEN: usize = 128;
pub const MAX_COMMITTEE_SIZE: usize = 15;
pub const MAX_FINGERPRINT_DIM: usize = 128;
pub const MAX_VOTES: usize = MAX_COMMITTEE_SIZE;
pub const MAX_POOL_SIZE: usize = 128;

// --- Scoring scale ---
pub const SCORE_SCALE: i32 = 10_000;
