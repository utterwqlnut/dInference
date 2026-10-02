"""Provider HTTP server.

On startup: loads the configured model and (placeholder) registers itself
on-chain with the model it serves. Handles /inference requests by generating
a response and the activation fingerprint.

Run:
    python -m off_chain_inference.server --model Qwen/Qwen2.5-1.5B-Instruct --port 8001
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chain import register_provider

from .fingerprint_pipeline import FingerprintProvider

log = logging.getLogger("provider")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

STATE: dict = {}


class InferenceRequest(BaseModel):
    prompt: str
    seed: int = Field(..., description="Per-request seed from the on-chain rotating seed")


class InferenceResponse(BaseModel):
    text: str
    fingerprint: list[float]
    seed: int
    model_id: str


def build_app(model_id: str, host: str, port: int, stake_lamports: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("loading provider for model=%s", model_id)
        STATE["provider"] = FingerprintProvider(model_id)
        STATE["model_id"] = model_id
        endpoint_url = f"http://{host}:{port}"
        STATE["registration"] = register_provider(model_id, endpoint_url, stake_lamports)
        log.info("ready: %s", STATE["registration"])
        yield

    app = FastAPI(title="dInference Provider", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_id": STATE.get("model_id"),
            "registration": STATE.get("registration").__dict__ if STATE.get("registration") else None,
        }

    @app.post("/inference", response_model=InferenceResponse)
    def inference(req: InferenceRequest):
        provider: FingerprintProvider = STATE["provider"]
        if provider is None:
            raise HTTPException(503, "provider not loaded")
        out = provider.generate(req.prompt, seed=req.seed)
        return InferenceResponse(
            text=out["text"],
            fingerprint=out["fingerprint"],
            seed=out["seed"],
            model_id=STATE["model_id"],
        )

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HuggingFace model id (must be in config)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--stake", type=int, default=1_000_000_000, help="Stake in lamports (placeholder)")
    args = p.parse_args()

    app = build_app(args.model, args.host, args.port, args.stake)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
