"""Benchmark fingerprint overhead and verifier wall time over N prompts.

Pulls prompts from databricks/databricks-dolly-15k (first N instructions),
runs each through: (1) baseline generation with no capture, (2) provider
pipeline with fingerprint capture, (3) verifier pipeline. Uses a fixed seed
for the projection matrix across all runs. Reports mean/median/p95 timings
and fingerprint cosine distribution.

Usage:
    python scripts/bench_fingerprint.py [--n 100] [--max-new-tokens 256] \
        [--model HuggingFaceTB/SmolLM2-135M-Instruct] [--seed 12345] \
        [--out /tmp/bench.json]
"""

import argparse
import json
import os
import statistics
import time

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from off_chain_inference.fingerprint_pipeline import FingerprintProvider
from off_chain_verifier.fingerprint_pipeline import FingerprintVerifier


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def stats(xs, name, scale=1000.0, unit="ms"):
    xs = sorted(x * scale for x in xs)
    n = len(xs)
    p95 = xs[min(n - 1, int(0.95 * n))]
    print(
        f"  {name:32s} mean={statistics.mean(xs):7.1f} {unit}  "
        f"median={statistics.median(xs):7.1f}  p95={p95:7.1f}  "
        f"min={xs[0]:7.1f}  max={xs[-1]:7.1f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--out", default="/tmp/bench.json")
    args = ap.parse_args()

    print(f"loading dataset {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    prompts = [ds[i]["instruction"] for i in range(args.n)]
    print(f"  {len(prompts)} prompts, median chars={statistics.median(len(p) for p in prompts):.0f}")

    print(f"loading {args.model} (provider + verifier) ...")
    prov = FingerprintProvider(args.model)
    ver = FingerprintVerifier(args.model)
    tok = prov.tokenizer
    dev = prov.model.device

    # Greedy decoding so baseline + provider generate the identical token sequence
    # (same prompt + deterministic sampling → same length, same work). Otherwise
    # per-run token count varies and the comparison is apples-to-oranges.
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )

    def baseline_time(prompt):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = enc["input_ids"].to(dev)
        attn = enc["attention_mask"].to(dev)
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            seq = prov.model.generate(input_ids=ids, attention_mask=attn, **gen_kwargs)
        synchronize()
        n_new = int(seq.shape[1] - ids.shape[1])
        return time.perf_counter() - t0, n_new

    def provider_time(prompt):
        synchronize()
        t0 = time.perf_counter()
        out = prov.generate(
            prompt, seed=args.seed,
            max_new_tokens=args.max_new_tokens, do_sample=False,
        )
        synchronize()
        return time.perf_counter() - t0, out

    def verifier_time(prompt, out):
        synchronize()
        t0 = time.perf_counter()
        r = ver({
            "prompt": prompt,
            "response": out["text"],
            "seed": args.seed,
            "fingerprint": out["fingerprint"],
        })
        synchronize()
        return time.perf_counter() - t0, r

    # Warmup.
    print("warmup ...")
    _ = baseline_time(prompts[0])
    _, warm_out = provider_time(prompts[0])
    _ = verifier_time(prompts[0], warm_out)

    bl_ts, fp_ts, vf_ts, coss, n_tok, n_tok_bl = [], [], [], [], [], []
    print(f"running {args.n} prompts (greedy, max_new_tokens={args.max_new_tokens}, seed={args.seed}) ...")
    for i, p in enumerate(prompts):
        t_bl, nbl = baseline_time(p)
        t_fp, out = provider_time(p)
        t_vf, vres = verifier_time(p, out)
        bl_ts.append(t_bl)
        fp_ts.append(t_fp)
        vf_ts.append(t_vf)
        coss.append(vres["fp_cos"])
        n_tok.append(vres["n_tokens"])
        n_tok_bl.append(nbl)
        if (i + 1) % 10 == 0 or i == args.n - 1:
            print(
                f"  [{i+1:3d}/{args.n}] bl={t_bl*1000:6.0f}ms/{nbl:3d}tok  "
                f"fp={t_fp*1000:6.0f}ms/{vres['n_tokens']:3d}tok  "
                f"vf={t_vf*1000:5.0f}ms  cos={vres['fp_cos']:.4f}"
            )

    print(f"\n=== Timings over {args.n} prompts ===")
    stats(bl_ts, "baseline (no capture)")
    stats(fp_ts, "provider w/ fingerprint")
    stats(vf_ts, "verifier (1 fwd pass)")

    mean_bl = statistics.mean(bl_ts)
    mean_fp = statistics.mean(fp_ts)
    mean_vf = statistics.mean(vf_ts)
    # Per-token comparison: greedy guarantees baseline and provider produced
    # the same sequence (same length), but we double-check and compute ms/token.
    mismatches = sum(1 for a, b in zip(n_tok, n_tok_bl) if a != b)
    if mismatches:
        print(f"  WARNING: {mismatches}/{args.n} runs had different baseline/provider token counts")
    bl_per_tok = [t / max(n, 1) for t, n in zip(bl_ts, n_tok_bl)]
    fp_per_tok = [t / max(n, 1) for t, n in zip(fp_ts, n_tok)]
    mean_bl_pt = statistics.mean(bl_per_tok)
    mean_fp_pt = statistics.mean(fp_per_tok)
    print(
        f"\n  mean baseline per-token:         {mean_bl_pt * 1000:7.2f} ms/tok"
    )
    print(
        f"  mean provider per-token:         {mean_fp_pt * 1000:7.2f} ms/tok  "
        f"(overhead {(mean_fp_pt/mean_bl_pt - 1)*100:+.1f}%)"
    )
    print(
        f"  mean provider overhead (total):  {(mean_fp - mean_bl) * 1000:+7.1f} ms  "
        f"({(mean_fp / mean_bl - 1) * 100:+.1f}%)"
    )
    print(
        f"  mean verifier wall time:         {mean_vf * 1000:7.1f} ms  "
        f"({mean_vf / mean_fp * 100:.1f}% of provider wall)"
    )

    print(f"\n=== Fingerprint quality over {args.n} prompts ===")
    print(f"  mean fp_cos:   {statistics.mean(coss):.6f}")
    print(f"  min fp_cos:    {min(coss):.6f}")
    print(f"  max fp_cos:    {max(coss):.6f}")
    print(f"  tokens:        mean={statistics.mean(n_tok):.1f}  median={statistics.median(n_tok):.1f}  max={max(n_tok)}")

    print(f"\n=== On-chain commitment ===")
    print(f"  fingerprint size: {len(warm_out['fingerprint'])} float32 = {len(warm_out['fingerprint'])*4} bytes")

    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "baseline_ms": [x * 1000 for x in bl_ts],
            "provider_ms": [x * 1000 for x in fp_ts],
            "verifier_ms": [x * 1000 for x in vf_ts],
            "cos": coss,
            "ntok": n_tok,
            "ntok_baseline": n_tok_bl,
        }, f)
    print(f"\nwrote raw data to {args.out}")


if __name__ == "__main__":
    main()
