use anchor_lang::prelude::*;

#[error_code]
pub enum DInferenceError {
    #[msg("Model is not in the allowed registry")]
    ModelNotAllowed,
    #[msg("Stake amount below minimum")]
    StakeTooLow,
    #[msg("Bounty amount below minimum")]
    BountyTooLow,
    #[msg("Bond amount below minimum")]
    BondTooLow,
    #[msg("Operator is not active")]
    OperatorInactive,
    #[msg("Operator model mismatch")]
    OperatorModelMismatch,
    #[msg("Unstake already requested")]
    UnstakeAlreadyRequested,
    #[msg("Unstake not yet requested")]
    UnstakeNotRequested,
    #[msg("Unstake cooldown has not elapsed")]
    UnstakeCooldownActive,
    #[msg("Bounty is not in the expected status (already claimed, challenged, or finalized?)")]
    WrongBountyStatus,
    #[msg("Challenge window has elapsed")]
    ChallengeWindowClosed,
    #[msg("Challenge window still open")]
    ChallengeWindowOpen,
    #[msg("Vote window has elapsed")]
    VoteWindowClosed,
    #[msg("Not on the verifier committee")]
    NotOnCommittee,
    #[msg("Already voted on this challenge")]
    AlreadyVoted,
    #[msg("Challenge is not open")]
    ChallengeNotOpen,
    #[msg("Not enough votes and vote window still open")]
    VotesNotReady,
    #[msg("Seed rotation window has not elapsed")]
    SeedRotationNotReady,
    #[msg("Unclaim timeout has not elapsed")]
    UnclaimTimeoutActive,
    #[msg("Overflow in arithmetic")]
    MathOverflow,
    #[msg("Fingerprint dimension too large")]
    FingerprintTooLarge,
    #[msg("Model string too long")]
    ModelStringTooLong,
    #[msg("Admin only")]
    AdminOnly,
    #[msg("Provider stake is insufficient to cover a bounty of this size")]
    StakeInsufficientForBounty,
    #[msg("Pool is at capacity")]
    QueueFull,
    #[msg("Not enough verifiers in pool to form a committee")]
    InsufficientVerifiers,
    #[msg("Account passed does not match the bounty's winning provider")]
    NotAssignedProvider,
}
