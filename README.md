# dInference

A decentralized AI inference marketplace on Solana. Independent operators run open-source LLMs off-chain and get paid per token. Correctness is enforced by an optimistic rollup with an **activation fingerprint** check at the heart of verification.

## Overview

Users submit prompts on-chain and pay per token. Any node with a GPU can register as a **provider**, pick up jobs, run inference locally, and post the response back on-chain. Results are assumed honest and finalized after a challenge window (default: *X* days). Within that window, any user can challenge a response and trigger verification. Fraud — by a provider or by a verifier — is punished by slashing the offender's staked SOL.

## Why optimistic?

Running an LLM twice for every request would double the cost of the network. Instead, we only re-run inference when someone pays to challenge, and we make challenges reliable through a single lightweight proof the provider publishes alongside every response:

- **Activation fingerprint** — a compact vector derived from the model's final-layer hidden states as it generated the response. On challenge, the verifier recomputes it and checks a cosine-similarity threshold.

We rely on a hidden-state fingerprint rather than exact-token replay because GPU nondeterminism (different kernels, batch sizes, dtypes, flash-attn versions) makes bit-exact reproduction unreliable between honest providers and honest verifiers.

## Protocol

### 1. Inference

- The on-chain program maintains a **rotating seed** that advances on a fixed schedule. Each inference request is bound to the current seed.
- The provider runs the requested model on the prompt using the protocol-fixed sampling parameters.
- During generation, for each new token the provider captures the final-layer hidden state `h_i`, projects it through a seeded random matrix `P` (derived from the request's seed), and accumulates the result:

  ```
  f = Σ_i h_i · P
  ```

  where `P ∈ R^(hidden_dim × proj_dim)` is sampled from the per-request seed.

- The provider returns `(response, fingerprint f)` on-chain.

### 2. Challenge window

The response is treated as correct unless a user files a challenge within the window. Challenges require a bond; frivolous challenges forfeit it.

### 3. Verification

On challenge, the `(prompt, response, seed, fingerprint)` tuple is dispatched to a committee of **verifiers**. Each verifier:

- Teacher-forces the response through the claimed model in a single forward pass, capturing the final-layer hidden state at each response position.
- Rebuilds `P` from the per-request seed and recomputes `f'` by the same projection-and-sum procedure.
- Checks cosine similarity: `cos(f, f') > τ` (e.g. `τ = 0.99`). Pass → honest. Fail → provider slashed.

Why cosine similarity: scale-invariant, bounded in `[-1, 1]`, insensitive to response length and dtype magnitude. Honest providers hit `cos ≈ 1.0` (small GPU-noise drift). Different-model attackers land near zero or negative.

Verifiers reach **consensus**:
- If the majority says the response is invalid → the provider is slashed, the challenger is rewarded.
- If the majority says the response is valid → the challenger forfeits the bond.
- Verifiers in the losing minority are slashed to deter collusion and lazy voting.

## Scheduling and priority fees

Users can attach a **priority fee** to a request. The scheduler batches work per model (each model has its own queue) and matches jobs to providers using both price and speed:

- Each provider has a running-mean **tokens/second** stat for every model it serves. If this stat drops below a protocol-defined threshold too often, the provider is slashed.
- At each scheduling tick, the contract looks at the pool of available providers for a given model, pulls the top `N` requests from that model's queue (where `N` = available providers), and assigns the highest-priority-fee jobs to the fastest providers.

This gives users a clean dial: pay more, land on a faster provider.

## Slashing summary

| Actor | Offense | Penalty |
|---|---|---|
| Provider | Fingerprint cosine similarity below threshold | Stake slashed, challenger rewarded |
| Provider | Chronically below declared tok/s | Stake slashed |
| Verifier | Voted with the losing minority | Stake slashed |
| Challenger | Challenge rejected by consensus | Bond forfeited |

## Architecture decisions

### Why a fingerprint check instead of a zero-knowledge proof?

The obvious alternative is a zero-knowledge proof of correct inference (zkML): the provider produces a cryptographic proof that they ran the exact claimed model on the prompt. We deliberately don't go this route — **zkML proving slows inference by 10³–10⁶×**, which kills the whole value proposition of a decentralized inference marketplace. A provider that generates 1 tok/s because it's producing a SNARK alongside each token is not a useful inference provider.

Our approach inverts the trade-off:

- **Providers run at native speed.** The projection + sum per token is a few hundred microseconds of extra work and uses a hidden state the model already computes.
- **Verification is slow but rare.** Verifiers do a single real forward pass only when challenged. In the optimistic case (the vast majority of requests), no verification happens at all.
- **The check is statistical, not cryptographic.** We accept that a sophisticated attacker could in principle craft hidden states that project to a matching fingerprint. In exchange we get a protocol that's actually usable.

### Why stop verification at the activation level?

A natural question is whether we should *also* verify sampling — that the provider used the mandated temperature / top-p / sampler, didn't greedy-decode, didn't tamper with the LM head. We prototyped a statistical sampling check (a KS test on the probabilities of selected tokens) and found that it works in principle but adds significant complexity and false-positive risk.

More importantly, we concluded **sampling verification isn't needed for economic security.** Walk through what an attacker actually saves by tampering with sampling:

| Attack | Compute saved | Worth catching? |
|---|---|---|
| Greedy instead of sampling | ~1% of per-token cost (skip softmax + multinomial draw) | No |
| Lower temperature | Zero. Same forward pass, same softmax, same draw. | No |
| Top-k / top-p tampering | Negligible. | No |
| Smaller model | 2–100× cheaper per token | **Yes — fingerprint catches this** |
| Not running the model at all | 100% | **Yes — fingerprint catches this** |

The forward pass is >99% of per-token compute. Any attack that doesn't change the forward pass saves the attacker essentially nothing — so a rational provider has no economic incentive to tamper with sampling. The attacks with real payoff (wrong model, no model) all show up as a different hidden-state trajectory and get caught by the fingerprint.

Stopping at the activation level therefore:

- **Covers every economically rational attack** with a single, interpretable check.
- **Avoids false positives** on honest providers (statistical sampling tests are sensitive to GPU nondeterminism, near-deterministic positions, and discrete-distribution artifacts).
- **Keeps the protocol minimal** — one threshold (`cos_threshold`) rather than a battery of statistical tunables.

If the threat model later expands to include adversarial *content* manipulation (e.g. biasing responses regardless of compute cost), sampling verification can be added as a second layer. For the economic-security use case, the fingerprint is sufficient.

## Results

We evaluated the fingerprint check on **500 diverse prompts** across three scenarios, all claiming the same model (`Qwen2.5-1.5B-Instruct`) and verified against it:

| Scenario | Actual model | Mean `fp_cos` | Pass rate |
|---|---|---|---|
| `honest` | Qwen2.5-1.5B-Instruct | **1.0000** | **1.0000** |
| `cheat_same_family_smaller` | Qwen2.5-0.5B-Instruct | -0.0059 | 0.0000 |
| `cheat_cross_family_similar` | SmolLM2-1.7B-Instruct | -0.0018 | 0.0000 |

**Aggregate:**
- False positive rate (honest rejected): **0.0000**
- True positive rate (dishonest caught): **1.0000**
- Overall accuracy: **1.0000**

Perfect separation across all 1,500 samples at a fixed `cos_threshold = 0.99`. The two cheating scenarios test the two attack types that matter: same-family downsizing (which saves ~3× compute by swapping a smaller model from the same family) and cross-family substitution. Both land near zero cosine similarity as predicted — different hidden-state manifolds project through independent per-model `P` matrices to effectively independent 64-dim vectors.

## Running the evaluation

The eval scales linearly with (prompts × scenarios × model-size), so we run it on **[Modal](https://modal.com)** for parallel GPU execution rather than a laptop. The scripts in `scripts/` are self-contained Modal apps:

```bash
pip install modal
modal setup  # one-time auth

modal run scripts/experiment_cross_family.py
```

Each scenario's generation is dispatched to its own A10G worker via `generate_responses.spawn(...)`, so total wall-clock is roughly the slowest single-scenario run rather than their sum. A shared Modal `Volume` caches HuggingFace model weights across runs so the second invocation skips downloads entirely.

Full run on A10G (3 scenarios × 500 prompts × 128 new tokens, including verification): **~45 minutes** wall-clock after model weights are cached.

## Status

Early design / work-in-progress. Contributions, critiques, and attacks on the protocol are welcome.
