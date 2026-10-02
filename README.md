# dInference

A decentralized AI inference marketplace on Solana. Users post bounties for LLM inference work; independent operators race to serve them; correctness is enforced by an optimistic rollup with an **activation-fingerprint** check at the heart of verification.

Validated at **100% TPR / 0% FPR** on 1,500 cross-family samples at `cos_threshold = 0.99`.

## How it works

1. **User posts a bounty.** User escrows SOL on-chain, uploads prompt to Arweave, submits `(prompt_txid, model_id, bounty_lamports)` via one tx. Chain emits `BountyPosted`.
2. **Providers race.** Every provider daemon for that model sees the event, fetches the prompt from Arweave, runs inference, uploads response to Arweave, and submits `claim_bounty(response_txid, fingerprint)`. First valid tx to land wins; losers' txs revert.
3. **Challenge window opens.** For a configurable window (default ~7 days), anyone can challenge the response by posting a bond. The chain deterministically picks a committee of K verifiers from the model's verifier pool via `hash(seed || bounty || i) mod pool_size` for each slot.
4. **Verifiers vote.** Each committee member fetches prompt+response from Arweave, recomputes the fingerprint under the real model, compares cosine similarity, and submits `verifier_vote` on-chain.
5. **Early finalize.** The moment vote counts can no longer flip the outcome (quorum of guilty reached, or honest majority mathematically locked), anyone calls `finalize_challenge`. Payouts and slashing happen atomically.
6. **Optimistic payment.** If no challenge arrives within the window, `finalize_payment` pays the winning provider the full bounty.

## Economics

| Actor | Action | Reward | Penalty |
|---|---|---|---|
| User | Post bounty | — | Escrowed; refunded only if cheater proven |
| Provider | Win bounty race (honest) | Full bounty | — |
| Provider | Win bounty race (cheated) | — | Half of stake slashed |
| Challenger | Open challenge (correct) | 1.25× bounty + user refund | — |
| Challenger | Open challenge (wrong) | — | Bond consumed by verify fees + vault |
| Verifier | Vote with majority | `verify_fee` credited to stake | — |
| Verifier | Vote against majority OR didn't vote | — | `dishonesty_fee` slashed from stake |

Verifier fees are **constant and independent of verdict** — removes directional voting incentive.

Provider stake at claim time: `stake ≥ 2 × bounty × 1.25` so a half-stake slash always covers the challenger reward.

Verifier stake at register time: `stake ≥ dishonesty_fee` so one slash cannot underflow.

## Verification: activation fingerprint

During generation, the provider computes:

```
P = seeded_random_matrix(seed, hidden_dim × proj_dim)
f = Σᵢ hᵢ · P        for i = 1..n_tokens generated
```

where `hᵢ` is the final-layer hidden state at generation step i.

On challenge, the verifier teacher-forces `prompt+response` through the claimed model, recomputes `f'` the same way, and checks `cos(f, f') > cos_threshold` (default 0.99).

Honest providers hit `cos ≈ 1.0` across all tested models (0% FPR on 1,500 samples). Wrong-model attackers land near zero cosine because the projection through a different model's hidden-state manifold produces an unrelated vector (100% TPR).

**Why not zkML:** zero-knowledge inference proofs run 10³–10⁶× slower than native inference. That would kill tok/s and make the marketplace unusable. The fingerprint runs at native speed with microseconds of per-token overhead.

**Why stop at the activation level:** sampling tampering (greedy decoding, temperature changes) saves attackers ~0% compute. The attacks that save real compute (wrong model, no model at all) show up as fingerprint divergence. A sampling-level check would add complexity + false-positive risk for no economic security gain.

## Verifier pool & committee selection

- Each model has a **`VerifierPool` PDA** containing a list of currently-available verifier operator pubkeys.
- `register_verifier` pushes to the pool. `request_unstake_verifier` removes from it.
- On `challenge`: `K` verifiers are **randomly selected** from the pool using `hash(rotating_seed || bounty_pda || i) % pool.len()` for `i = 0..K`, swap-removed from the pool. One committee at a time per verifier (removed = not eligible for simultaneous committees).
- On `finalize_challenge`: committee members whose `is_active` is still true get added back to the pool.

Selection is O(K) hashes, O(K) swap-removes. No sorting. Deterministic and reproducible — SDKs can replay the same hash chain to predict committee membership before the on-chain selection lands.

## On-chain state (Anchor accounts)

- `GlobalConfig` — admin + protocol parameters (stake mins, bounty mins, challenge window, verify/dishonesty fees, cos threshold, committee size, quorum).
- `RotatingSeed` — current seed, rotates on a schedule (permissionless `rotate_seed`).
- `StakeVault` — single PDA holding all operator stakes and user escrows.
- `ModelInfo` — per-model: id, hidden_dim, allowed flag.
- `VerifierPool` — per-model: Vec of available verifier pubkeys.
- `ProviderAccount` — per-operator: model, stake, active flag, unstake state.
- `VerifierAccount` — per-operator: model, stake, active flag, unstake state.
- `Bounty` — per-request: user, prompt_txid, bounty_lamports, status, winner, timestamps.
- `ResponseCommitment` — per-claim: response_txid, fingerprint, timestamps, challenge deadline.
- `Challenge` — per-challenge: committee, verdicts, status, vote deadline.

## Program instructions

**Admin / setup:**
- `init_config(args)` — one-time, sets protocol parameters.
- `init_model(model_id, hidden_dim)` — admin, per-model, creates `ModelInfo` + `VerifierPool`.

**Operator lifecycle:**
- `register_provider(model_id, stake)` / `register_verifier(model_id, stake)` — escrow stake, create account, (verifier) push to pool.
- `request_unstake_*` → `claim_unstake_*` — two-step unstake with cooldown ≥ challenge window.

**Bounty lifecycle:**
- `post_bounty(nonce, prompt_txid, bounty_lamports)` — user escrows bounty, creates `Bounty` PDA, emits `BountyPosted`.
- `claim_bounty(response_txid, fingerprint)` — provider race; atomic status flip `Open → Claimed`; emits `BountyClaimed`.
- `finalize_payment()` — permissionless; after challenge window closes, pays winner.
- `reclaim_unclaimed_bounty()` — user reclaims escrow if no provider claimed within timeout.

**Challenge:**
- `challenge(bond)` — picks committee from pool via random selection, creates `Challenge` PDA, emits `ChallengeOpened`.
- `verifier_vote(verdict, fp_cos_scaled)` — committee member submits vote; emits `VoteSubmitted`.
- `finalize_challenge()` — permissionless; early-finalizes when outcome is locked; pays verify fees, slashes dishonesty fees, settles bounty, adds surviving verifiers back to pool; emits `ChallengeResolved`.

**Housekeeping:**
- `rotate_seed()` — permissionless; advances rotating seed when due.

## Attack coverage

| Attack | Caught by |
|---|---|
| Run a different model | Fingerprint check — `cos ≈ 0` vs real model |
| Don't run the model at all | Fingerprint check — no valid hidden states to project |
| Truncation (short response) | Verifier checks last token is EOS if `n_tokens < max_new_tokens` |
| Sampling tampering (greedy, temp) | Not caught (economically irrational — saves ~0% compute) |
| Provider stalls after winning | Challenge window + `claim_bounty` reverts if re-attempted |
| Challenger spams fake challenges | Loses bond on each wrong challenge |
| Verifier collusion to vote guilty | Minority voters slashed `dishonesty_fee`; constant fee removes bias |
| Verifier doesn't vote | Slashed `dishonesty_fee` same as minority |
| Multiple providers claim same bounty | Atomic status check; second claim reverts |
| Provider under-stakes relative to bounty | `claim_bounty` requires `stake ≥ 2 × bounty × 1.25` |

## Off-chain components

**`FingerprintProvider`** (`off_chain_inference/`) — `transformers.Pipeline` subclass; runs model, computes fingerprint, returns `{text, fingerprint, seed}`.

**`FingerprintVerifier`** (`off_chain_verifier/`) — `transformers.Pipeline` subclass; teacher-forces response, recomputes fingerprint, returns `{fp_cos, passed, truncated, reason}`.

**Provider daemon** (planned) — WebSocket subscription to `BountyPosted` events, races to `claim_bounty`.

**Verifier daemon** (planned) — WebSocket subscription to `ChallengeOpened` events, submits `verifier_vote` + opportunistic `finalize_challenge`.

## Results

Evaluated on 500 prompts × 3 scenarios (1,500 samples total):

| Scenario | Actual model | Mean `fp_cos` | Pass rate |
|---|---|---|---|
| `honest` | Qwen2.5-1.5B-Instruct | **1.0000** | **1.0000** |
| `cheat_same_family_smaller` | Qwen2.5-0.5B-Instruct | -0.0059 | 0.0000 |
| `cheat_cross_family_similar` | SmolLM2-1.7B-Instruct | -0.0018 | 0.0000 |

**FPR 0.0000, TPR 1.0000, accuracy 1.0000** at `cos_threshold = 0.99`.

## Code status

- **Smart contract** (`programs/dinference/`) — compiles, 6/6 Rust unit tests pass, BPF binary builds. Deterministic committee selection implemented. Economic parameters configurable. Ready for devnet deployment.
- **Fingerprint pipelines** (`off_chain_inference/`, `off_chain_verifier/`) — validated end-to-end on cross-family evals. Ready to integrate with on-chain lifecycle.
- **Daemons + SDK** — not yet built.

## Running the evaluation

Uses [Modal](https://modal.com) for parallel GPU execution:

```bash
pip install modal
modal setup
modal run scripts/experiment_cross_family.py
```

Wall-clock ~45 min on A10G × 3 scenarios × 500 prompts (parallelized), after model weights are cached.
