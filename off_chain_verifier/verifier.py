from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import kstest
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

DEFAULT_CONFIG = (
    Path(__file__).parent.parent / "off_chain_inference" / "config" / "models.yaml"
)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class VerifierPipeline(Pipeline):
    """Custom pipeline whose postprocess runs the selected-token KS test.

    Input to __call__: {"prompt": str, "response": str}
    Output: {"ks_stat": float, "p_value": float, "n_tokens": int, "passed": bool,
             "q_i": list[float]}
    """

    def __init__(
        self,
        model_id: str,
        config_path: str | Path = DEFAULT_CONFIG,
        ks_threshold: float = 0.01,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        allowed = {m["id"]: m for m in cfg["models"]}
        if model_id not in allowed:
            raise ValueError(
                f"Model {model_id!r} is not in the protocol's valid-model set. "
                f"Allowed: {sorted(allowed)}"
            )

        self.model_id = model_id
        self.sampling = cfg["sampling"]
        self.ks_threshold = ks_threshold

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

        super().__init__(model=model, tokenizer=tokenizer, framework="pt", device=device)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        prompt = inputs["prompt"]
        response = inputs["response"]

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(prompt + response, return_tensors="pt").input_ids[0]

        prompt_len = prompt_ids.shape[0]
        return {
            "input_ids": full_ids.unsqueeze(0).to(self.model.device),
            "prompt_len": prompt_len,
        }

    def _forward(self, model_inputs):
        input_ids = model_inputs["input_ids"]
        prompt_len = model_inputs["prompt_len"]

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits  # [1, T, V]

        return {"logits": logits, "input_ids": input_ids, "prompt_len": prompt_len}

    def postprocess(self, model_outputs):
        logits = model_outputs["logits"][0]              # [T, V]
        input_ids = model_outputs["input_ids"][0]        # [T]
        prompt_len = model_outputs["prompt_len"]

        # Position i's logits predict token i+1. Response tokens live at
        # positions [prompt_len, T), so we need logits at [prompt_len-1, T-1).
        resp_logits = logits[prompt_len - 1 : -1]        # [N, V]
        resp_tokens = input_ids[prompt_len:]             # [N]

        probs = self._apply_sampling_params(resp_logits).to(torch.float32)
        q_i = probs[torch.arange(resp_tokens.shape[0]), resp_tokens].cpu().numpy()

        expected_cdf = self._build_expected_cdf(probs)

        ks_stat, p_value = kstest(q_i, expected_cdf)
        passed = bool(p_value > self.ks_threshold)

        return {
            "ks_stat": float(ks_stat),
            "p_value": float(p_value),
            "n_tokens": int(q_i.shape[0]),
            "passed": passed,
            "q_i": q_i.tolist(),
        }

    def _apply_sampling_params(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply protocol-fixed temperature / top-p / top-k to raw logits."""
        temperature = float(self.sampling["temperature"])
        top_p = float(self.sampling["top_p"])
        top_k = int(self.sampling["top_k"])

        logits = logits / max(temperature, 1e-6)

        if top_k and top_k > 0:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        return torch.softmax(logits, dim=-1)

    def _build_expected_cdf(self, probs: torch.Tensor):
        """Analytical CDF of the mixture size-biased distribution over all N
        response positions. For each (position i, vocab token k), honest
        sampling places mass p_i(k) on the value p_i(k). Pooled across N
        positions and normalized by N:

            F(x) = (sum of p in flat with p <= x) / N
        """
        flat = probs.reshape(-1).cpu().numpy().astype(np.float64)
        flat = flat[flat > 0]
        n_positions = probs.shape[0]
        order = np.argsort(flat)
        sorted_p = flat[order]
        cum = np.cumsum(sorted_p) / n_positions

        def cdf(x):
            x = np.atleast_1d(x)
            idx = np.searchsorted(sorted_p, x, side="right") - 1
            out = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
            return out if out.shape else float(out)

        return cdf
