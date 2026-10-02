from pathlib import Path

import torch
import yaml
from transformers import pipeline

DEFAULT_CONFIG = Path(__file__).parent / "config" / "models.yaml"

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ProviderInference:
    def __init__(self, model_id: str, config_path: str | Path = DEFAULT_CONFIG):
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

        dtype = _DTYPES[allowed[model_id].get("dtype", "bfloat16")]
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=dtype,
            device_map=device,
        )

    def generate(self, prompt: str, **overrides) -> str:
        gen_kwargs = {
            "do_sample": True,
            "temperature": self.sampling["temperature"],
            "top_p": self.sampling["top_p"],
            "top_k": self.sampling["top_k"],
            "max_new_tokens": self.sampling["max_new_tokens"],
            **overrides,
        }
        out = self.pipe(prompt, return_full_text=False, **gen_kwargs)
        return out[0]["generated_text"]
