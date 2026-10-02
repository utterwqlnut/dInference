"""Anchor discriminators — 8-byte prefixes for instructions, accounts, events.

Anchor derives discriminators as the first 8 bytes of sha256(prefix:name) where
`prefix` is "global" for instructions, "account" for account structs, "event"
for events. These are what the on-chain program uses to route calls and
identify account types.
"""

from __future__ import annotations
import hashlib


def _disc(prefix: str, name: str) -> bytes:
    return hashlib.sha256(f"{prefix}:{name}".encode()).digest()[:8]


# ---- Instructions ----

IX_INIT_CONFIG                = _disc("global", "init_config")
IX_INIT_MODEL                 = _disc("global", "init_model")
IX_REGISTER_PROVIDER          = _disc("global", "register_provider")
IX_REGISTER_VERIFIER          = _disc("global", "register_verifier")
IX_REQUEST_UNSTAKE_PROVIDER   = _disc("global", "request_unstake_provider")
IX_CLAIM_UNSTAKE_PROVIDER     = _disc("global", "claim_unstake_provider")
IX_REQUEST_UNSTAKE_VERIFIER   = _disc("global", "request_unstake_verifier")
IX_CLAIM_UNSTAKE_VERIFIER     = _disc("global", "claim_unstake_verifier")
IX_POST_BOUNTY                = _disc("global", "post_bounty")
IX_CLAIM_BOUNTY               = _disc("global", "claim_bounty")
IX_FINALIZE_PAYMENT           = _disc("global", "finalize_payment")
IX_RECLAIM_UNCLAIMED_BOUNTY   = _disc("global", "reclaim_unclaimed_bounty")
IX_CHALLENGE                  = _disc("global", "challenge")
IX_VERIFIER_VOTE              = _disc("global", "verifier_vote")
IX_FINALIZE_CHALLENGE         = _disc("global", "finalize_challenge")
IX_ROTATE_SEED                = _disc("global", "rotate_seed")


# ---- Account types ----

ACC_GLOBAL_CONFIG     = _disc("account", "GlobalConfig")
ACC_ROTATING_SEED     = _disc("account", "RotatingSeed")
ACC_STAKE_VAULT       = _disc("account", "StakeVault")
ACC_MODEL_INFO        = _disc("account", "ModelInfo")
ACC_VERIFIER_POOL     = _disc("account", "VerifierPool")
ACC_PROVIDER_ACCOUNT  = _disc("account", "ProviderAccount")
ACC_VERIFIER_ACCOUNT  = _disc("account", "VerifierAccount")
ACC_BOUNTY            = _disc("account", "Bounty")
ACC_RESPONSE_COMMITMENT = _disc("account", "ResponseCommitment")
ACC_CHALLENGE         = _disc("account", "Challenge")


# ---- Events ----

EVT_BOUNTY_POSTED      = _disc("event", "BountyPosted")
EVT_BOUNTY_CLAIMED     = _disc("event", "BountyClaimed")
EVT_CHALLENGE_OPENED   = _disc("event", "ChallengeOpened")
EVT_VOTE_SUBMITTED     = _disc("event", "VoteSubmitted")
EVT_CHALLENGE_RESOLVED = _disc("event", "ChallengeResolved")
