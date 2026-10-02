# dInference

A decentralized AI inference marketplace on Solana. Independent operators run open-source LLMs off-chain and get paid per token. Correctness is enforced by an optimistic rollup with a statistical sampling check at the heart of verification.

## Overview

Users submit prompts on-chain and pay per token. Any node with a GPU can register as a **provider**, pick up jobs, run inference locally, and post the response back on-chain. Results are assumed honest and finalized after a challenge window (default: *X* days). Within that window, any user can challenge a response and trigger verification. Fraud — by a provider or by a verifier — is punished by slashing the offender's staked SOL.

## Why optimistic?

Running an LLM twice for every request would double the cost of the network. Instead, we only re-run inference when someone pays to challenge, and we make challenges reliable through a single statistical test on the provider's output:

- **Selected-token probability test** — the probabilities the real model assigns to the tokens the provider actually selected must, in aggregate, match what honest sampling would produce. Wrong-model, greedy-when-sampling, temperature tampering, and token injection all distort this distribution in detectable ways.

We rely on a statistical check rather than exact-token replay because GPU nondeterminism (different kernels, batch sizes, dtypes, flash-attn versions) makes bit-exact reproduction unreliable between honest providers and honest verifiers. The protocol is deliberately **loose at the token level and tight at the distribution level**.

## Protocol

### 1. Inference

- The on-chain program maintains a **rotating seed** that advances on a fixed schedule, used for request scheduling and provider rotation.
- The provider runs the requested model on the prompt using the protocol-fixed sampling parameters (temperature, top-p, top-k) and its own randomness.
- The provider returns the response on-chain.

### 2. Challenge window

The response is treated as correct unless a user files a challenge within the window. Challenges require a bond; frivolous challenges forfeit it.

### 3. Verification

On challenge, the `(prompt, response)` pair is dispatched to a committee of **verifiers**. Each verifier:

- Runs the model **once** on the prompt (teacher-forcing the provider's response) to recover the per-position distributions `p_i` over the vocabulary.
- For each position, reads off `q_i = p_i(t_i)` — the probability the real model assigned to the token the provider actually selected.
- Under honest sampling with the mandated parameters, `{q_1, ..., q_N}` follows a known distribution: the **size-biased distribution** of the `p_i`'s (the probability of selecting a token with mass `p` is itself `p`). The verifier computes the expected distribution directly from the recomputed `p_i`'s and KS-tests the empirical `{q_i}` against it.

What this catches:

- **Wrong model.** The provider's tokens were plausible under a different model's distribution, so they're frequently low-probability under the real model → `q_i`'s pile up near zero.
- **Greedy-when-sampling.** `q_i = max_k p_i(k)` every time → distribution pushed hard to the top.
- **Temperature / top-k tampering.** Sharper effective distributions bias `q_i`'s high; flatter ones bias them low.
- **Token injection.** Injected tokens have tiny `q_i`'s and show up as low-tail outliers.

GPU noise perturbs individual `p_i` values only slightly, and the KS test aggregates over the full response, so the test is noise-robust by construction. A minimum response length is enforced for challenges so the KS test has adequate power.

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
| Provider | Failed selected-token probability (KS) test | Stake slashed, challenger rewarded |
| Provider | Chronically below declared tok/s | Stake slashed |
| Verifier | Voted with the losing minority | Stake slashed |
| Challenger | Challenge rejected by consensus | Bond forfeited |

## Architecture decisions

### Why a statistical check instead of a zero-knowledge proof?

The obvious alternative to our KS test is a zero-knowledge proof of correct inference (zkML): the provider produces a cryptographic proof that they ran the exact claimed model on the prompt, and anyone can verify it in milliseconds.

We deliberately chose not to go this route. **zkML proving slows inference by multiple orders of magnitude** — current state-of-the-art zkML systems for transformer inference run roughly 10³–10⁶× slower than native inference. For a decentralized *inference* marketplace, that's fatal: the whole value proposition is competitive tokens-per-second. A provider that generates 1 tok/s because it's producing a SNARK alongside each token is not a useful inference provider — users will go to centralized APIs and the network has no reason to exist.

Our approach inverts the trade-off:

- **Providers run at native speed.** No proving overhead. Tok/s is competitive with centralized providers.
- **Verification is slow but rare.** Verifiers do a real forward pass, which is expensive, but only on challenge. In the optimistic case (the vast majority of requests), no verification happens at all.
- **The check is statistical, not cryptographic.** We accept that a sufficiently sophisticated and economically irrational attacker could in principle slip through. In exchange we get a protocol that's actually usable.

The KS test is "loose" in the formal sense — it does not prove exact execution. But it is **tight against every economically rational attack**: smaller models, greedy decoding, temperature tampering, LM-head substitution, token injection. Any attack that would save the provider meaningful compute shows up in the `q_i` distribution. Attacks that don't save compute aren't worth running.

In short: zkML buys you cryptographic certainty at the cost of making the service unusable. We buy a usable service at the cost of statistical rather than cryptographic certainty. For an open inference marketplace, that's the right trade.

## Status

Early design / work-in-progress. Contributions, critiques, and attacks on the protocol are welcome.
