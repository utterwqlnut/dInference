"""Verifier pipeline: activation-fingerprint check only.

Pipeline input: {"prompt": str, "response": str, "seed": int, "fingerprint": list[float]}
Pipeline output: {
    "fp_cos": float, "n_tokens": int, "passed": bool,
    "truncated": bool, "reason": str | None,
}

`max_new_tokens` is a protocol-fixed value loaded from config. If a response is
shorter than that cap and does not end in EOS, verification fails
(completion-rule check — stops providers from truncating to save compute).

The provider's claimed fingerprint `f` was built as Σ_i h_i · P over the
final-layer hidden states of the response, with P seeded from the per-request
seed. The verifier teacher-forces the response through the claimed model,
recomputes `f'` the same way, and checks cos(f, f') > cos_threshold.
"""

from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

DEFAULT_CONFIG = (
    Path(__file__).parent.parent / "off_chain_inference" / "config" / "models.yaml"
)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class FingerprintVerifier(Pipeline):
    def __init__(self, model_id: str, config_path: str | Path = DEFAULT_CONFIG):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        allowed = {m["id"]: m for m in cfg["models"]}
        if model_id not in allowed:
            raise ValueError(
                f"Model {model_id!r} not in valid-model set. Allowed: {sorted(allowed)}"
            )

        self.model_id = model_id
        self.proj_dim = int(cfg["fingerprint"]["proj_dim"])
        self.cos_threshold = float(cfg["fingerprint"]["cos_threshold"])
        self.max_new_tokens = int(cfg["sampling"]["max_new_tokens"])

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

    def preprocess(self, inputs):
        prompt = inputs["prompt"]
        response = inputs["response"]
        seed = int(inputs["seed"])
        claimed_fp = torch.tensor(inputs["fingerprint"], dtype=torch.float32)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(prompt + response, return_tensors="pt").input_ids[0]
        prompt_len = prompt_ids.shape[0]

        return {
            "input_ids": full_ids.unsqueeze(0).to(self.model.device),
            "prompt_len": prompt_len,
            "seed": seed,
            "claimed_fp": claimed_fp,
        }

    def _forward(self, model_inputs):
        with torch.no_grad():
            out = self.model(
                input_ids=model_inputs["input_ids"],
                output_hidden_states=True,
                return_dict=True,
            )
        return {
            "hidden_states": out.hidden_states[-1],  # final layer, [1, T, hidden]
            "input_ids": model_inputs["input_ids"],
            "prompt_len": model_inputs["prompt_len"],
            "seed": model_inputs["seed"],
            "claimed_fp": model_inputs["claimed_fp"],
        }

    def postprocess(self, model_outputs):
        hidden = model_outputs["hidden_states"][0]     # [T, hidden]
        input_ids = model_outputs["input_ids"][0]      # [T]
        prompt_len = model_outputs["prompt_len"]
        seed = model_outputs["seed"]
        claimed_fp = model_outputs["claimed_fp"]

        gpu = self.model.device
        resp_hidden = hidden[prompt_len - 1 : -1].to(gpu).to(torch.float32)  # [N, hidden]
        n_tokens = int(input_ids.shape[0] - prompt_len)

        P = self._projection(seed, device=resp_hidden.device)
        f_prime = (resp_hidden @ P).sum(dim=0).cpu()
        fp_cos = float(
            torch.nn.functional.cosine_similarity(
                f_prime.unsqueeze(0), claimed_fp.unsqueeze(0), dim=1
            ).item()
        )

        # Completion-rule check: if the response is shorter than the protocol's
        # max_new_tokens cap and doesn't end in EOS, the provider truncated to
        # save compute. Slash.
        last_token = int(input_ids[-1].item())
        is_eos = (last_token == self.tokenizer.eos_token_id)
        truncated = (n_tokens < self.max_new_tokens) and not is_eos

        fp_ok = fp_cos > self.cos_threshold
        passed = fp_ok and not truncated

        reason = None
        if not fp_ok:
            reason = "fingerprint_mismatch"
        elif truncated:
            reason = "truncated_without_eos"

        return {
            "fp_cos": fp_cos,
            "n_tokens": n_tokens,
            "truncated": truncated,
            "passed": passed,
            "reason": reason,
        }
