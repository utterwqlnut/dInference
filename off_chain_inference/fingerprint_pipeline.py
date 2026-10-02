"""Provider as a custom transformers.Pipeline that returns the generated
response and an activation fingerprint.

Fingerprint = sum over generated-token positions of (final hidden state) @ P,
where P is a seeded random projection matrix (seed is per-inference-sample —
the on-chain rotating seed).

Pipeline input: {"prompt": str, "seed": int, "overrides": dict | None}
Pipeline output: {"text": str, "fingerprint": list[float], "seed": int}
"""

from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

DEFAULT_CONFIG = Path(__file__).parent / "config" / "models.yaml"

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class FingerprintProvider(Pipeline):
    def __init__(self, model_id: str, config_path: str | Path = DEFAULT_CONFIG):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        allowed = {m["id"]: m for m in cfg["models"]}
        if model_id not in allowed:
            raise ValueError(
                f"Model {model_id!r} not in valid-model set. Allowed: {sorted(allowed)}"
            )

        self.model_id = model_id
        self.sampling = cfg["sampling"]
        self.proj_dim = int(cfg["fingerprint"]["proj_dim"])
        self.model_type = allowed[model_id].get("type", "base")

        dtype = _DTYPES[allowed[model_id].get("dtype", "bfloat16")]
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        model.eval()
        self.hidden_dim = model.config.hidden_size

        super().__init__(model=model, tokenizer=tokenizer, framework="pt", device=device)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def _projection(self, seed: int, device=None) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        P = torch.randn(self.hidden_dim, self.proj_dim, generator=g, dtype=torch.float32)
        return P if device is None else P.to(device)

    def generate(self, prompt: str, seed: int, **overrides) -> dict:
        """Convenience wrapper so callers can do provider.generate(prompt, seed=...)."""
        return self({"prompt": prompt, "seed": seed, "overrides": overrides})

    def preprocess(self, inputs):
        prompt = inputs["prompt"]
        seed = int(inputs["seed"])
        overrides = inputs.get("overrides") or {}

        if self.model_type == "instruct" and getattr(self.tokenizer, "chat_template", None):
            out = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = out["input_ids"].to(self.model.device)
            attention_mask = out["attention_mask"].to(self.model.device)
        else:
            tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids = tok["input_ids"]
            attention_mask = tok.get("attention_mask")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": input_ids.shape[1],
            "seed": seed,
            "overrides": overrides,
        }

    def _forward(self, model_inputs):
        overrides = model_inputs["overrides"]
        gen_kwargs = {
            "do_sample": True,
            "temperature": self.sampling["temperature"],
            "top_p": self.sampling["top_p"],
            "top_k": self.sampling["top_k"],
            "max_new_tokens": self.sampling["max_new_tokens"],
            **overrides,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Capture only the last decoder layer's final-position hidden state per
        # step via a forward hook. ~2% overhead vs 50%+ for output_hidden_states=True,
        # which materializes every layer at every step.
        captured = []
        # Hook the final norm so the captured state matches what
        # `output_hidden_states=True` returns (final layer, post-norm). Hooking
        # the last decoder block instead would capture pre-norm, which drifts
        # from what the verifier reconstructs by ~a full normalization.
        final_norm = self.model.model.norm
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h[:, -1, :].detach())
        handle = final_norm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                sequences = self.model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    **gen_kwargs,
                )
        finally:
            handle.remove()

        prompt_len = model_inputs["prompt_len"]
        # First hook fire is the prompt pass (last position = token BEFORE first
        # generated token). We want the hidden states that *produced* each
        # generated token: that is the state at position p-1 for token at
        # position p. Drop the final capture (it would correspond to the
        # position after the last generated token, which doesn't produce anything).
        n_new = sequences.shape[1] - prompt_len
        last_hs = torch.cat(captured[:n_new], dim=0)  # [n_new, hidden]

        return {
            "sequences": sequences,
            "last_hs": last_hs,
            "prompt_len": prompt_len,
            "seed": model_inputs["seed"],
        }

    def postprocess(self, model_outputs):
        sequences = model_outputs["sequences"]
        last_hs = model_outputs["last_hs"].to(torch.float32)   # [n_new, hidden]
        prompt_len = model_outputs["prompt_len"]
        seed = model_outputs["seed"]

        P = self._projection(seed, device=last_hs.device)
        fp = (last_hs @ P).sum(dim=0).cpu()                    # [proj_dim]

        gen_ids = sequences[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        return {
            "text": text,
            "fingerprint": fp.tolist(),
            "seed": int(seed),
        }
