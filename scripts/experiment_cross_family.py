"""Cross-model-family fingerprint verification experiment.

For each (claimed_model, actual_model) scenario:
  - Provider runs `actual_model` on every prompt, builds fingerprint using its
    own hidden states and the per-request seed.
  - Verifier is loaded with `claimed_model` and recomputes the fingerprint from
    its own hidden states on the same prompt+response+seed.
  - If actual == claimed → honest, cos should be ~1.0.
  - If actual != claimed → dishonest, cos should be far from 1.0.

Scenarios cover both same-family (different sizes) and cross-family attacks.

Run:
    modal run scripts/experiment_cross_family.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Claimed model = Qwen 1.5B. Attackers try to pass off cheaper (smaller) models.
# Economically rational attack: run a smaller model, claim the bigger one.
CLAIMED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Each entry: (actual_model, label, scenario_name). label=0 honest, label=1 dishonest.
SCENARIOS = [
    ("Qwen/Qwen2.5-1.5B-Instruct",            0, "honest"),
    ("Qwen/Qwen2.5-0.5B-Instruct",            1, "cheat_same_family_smaller"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct",   1, "cheat_cross_family_similar"),
]

MAX_NEW_TOKENS = 128
BASE_SEED = 1_000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.44",
        "torch>=2.3",
        "pyyaml>=6.0",
        "accelerate>=0.33",
        "pandas>=2.0",
    )
    .add_local_dir(ROOT / "off_chain_inference", remote_path="/root/off_chain_inference")
    .add_local_dir(ROOT / "off_chain_verifier", remote_path="/root/off_chain_verifier")
)

hf_cache = modal.Volume.from_name("dinference-hf-cache", create_if_missing=True)
app = modal.App("dinference-cross-family", image=image)


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache}, timeout=60 * 90)
def generate_responses(prompts: list[str], model_id: str, seeds: list[int]) -> list[dict]:
    from off_chain_inference import FingerprintProvider

    provider = FingerprintProvider(model_id)
    out = []
    for i, (p, s) in enumerate(zip(prompts, seeds)):
        if i % 50 == 0:
            print(f"[gen {model_id}] {i}/{len(prompts)}")
        out.append(provider.generate(p, seed=s, max_new_tokens=MAX_NEW_TOKENS))
    return out


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache}, timeout=60 * 90)
def verify_rows(rows: list[dict], claimed_model: str) -> list[dict]:
    from off_chain_verifier import FingerprintVerifier

    verifier = FingerprintVerifier(claimed_model)
    out = []
    for i, r in enumerate(rows):
        v = verifier(
            {
                "prompt": r["prompt"],
                "response": r["response"],
                "seed": r["seed"],
                "fingerprint": r["fingerprint"],
            }
        )
        if i % 50 == 0:
            print(f"[verify] {i}/{len(rows)}")
        out.append(v)
    return out


@app.local_entrypoint()
def main():
    import json
    import pandas as pd

    prompts_path = ROOT / "data" / "prompts.txt"
    prompts = [p.strip() for p in prompts_path.read_text().splitlines() if p.strip()]
    seeds = [BASE_SEED + i for i in range(len(prompts))]
    print(f"[local] {len(prompts)} prompts, {len(SCENARIOS)} scenarios")

    # Fan out generation across scenarios in parallel.
    gen_futures = {
        name: generate_responses.spawn(prompts, actual, seeds)
        for actual, _label, name in SCENARIOS
    }
    gen_results = {name: f.get() for name, f in gen_futures.items()}

    rows = []
    for actual, label, name in SCENARIOS:
        for p, s, g in zip(prompts, seeds, gen_results[name]):
            rows.append(
                {
                    "prompt": p, "seed": s, "response": g["text"],
                    "fingerprint": g["fingerprint"],
                    "actual_model": actual, "claimed_model": CLAIMED_MODEL,
                    "label": label, "scenario": name,
                }
            )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{**r, "fingerprint": json.dumps(r["fingerprint"])} for r in rows]
    ).to_csv(results_dir / "cross_family_dataset.csv", index=False)

    print(f"[local] verifying {len(rows)} rows against {CLAIMED_MODEL}")
    verdicts = verify_rows.remote(rows, CLAIMED_MODEL)

    df = pd.DataFrame(
        [{**r, "fingerprint": json.dumps(r["fingerprint"])} for r in rows]
    )
    for k in ("fp_cos", "n_tokens", "passed"):
        df[k] = [v[k] for v in verdicts]
    df.to_csv(results_dir / "cross_family_verified.csv", index=False)

    print("\n=== Per-scenario summary ===")
    print(f"{'scenario':<40} {'label':>5} {'n':>4} {'mean cos':>10} {'pass rate':>10}")
    for _, _, name in SCENARIOS:
        sub = df[df["scenario"] == name]
        print(
            f"{name:<40} {sub['label'].iloc[0]:>5} {len(sub):>4} "
            f"{sub['fp_cos'].mean():>10.4f} {sub['passed'].mean():>10.4f}"
        )

    correct = df["passed"] == (df["label"] == 0)
    honest = df[df["label"] == 0]
    dishonest = df[df["label"] == 1]
    fpr = 1.0 - honest["passed"].mean()         # honest wrongly rejected
    tpr = 1.0 - dishonest["passed"].mean()      # dishonest correctly rejected
    acc = correct.mean()
    print(f"\n[local] False positive rate (honest rejected):   {fpr:.4f}")
    print(f"[local] True positive rate  (dishonest caught):  {tpr:.4f}")
    print(f"[local] Overall accuracy:                        {acc:.4f}")
