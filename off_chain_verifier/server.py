"""Verifier HTTP server.

On startup: loads the claimed model and (placeholder) registers itself on-chain
as a verifier for that model. Handles /verify requests by recomputing the
activation fingerprint and comparing against the provider's claimed one.

Run:
    python -m off_chain_verifier.server --model Qwen/Qwen2.5-1.5B-Instruct --port 8002
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chain import register_verifier

from .fingerprint_pipeline import FingerprintVerifier

log = logging.getLogger("verifier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

STATE: dict = {}


class VerifyRequest(BaseModel):
    prompt: str
    response: str
    seed: int
    fingerprint: list[float] = Field(..., description="Provider's claimed fingerprint vector")


class VerifyResponse(BaseModel):
    fp_cos: float
    n_tokens: int
    truncated: bool
    passed: bool
    reason: str | None
    model_id: str


def build_app(model_id: str, host: str, port: int, stake_lamports: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("loading verifier for model=%s", model_id)
        STATE["verifier"] = FingerprintVerifier(model_id)
        STATE["model_id"] = model_id
        endpoint_url = f"http://{host}:{port}"
        STATE["registration"] = register_verifier(model_id, endpoint_url, stake_lamports)
        log.info("ready: %s", STATE["registration"])
        yield

    app = FastAPI(title="dInference Verifier", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_id": STATE.get("model_id"),
            "registration": STATE.get("registration").__dict__ if STATE.get("registration") else None,
        }

    @app.post("/verify", response_model=VerifyResponse)
    def verify(req: VerifyRequest):
        verifier: FingerprintVerifier = STATE["verifier"]
        if verifier is None:
            raise HTTPException(503, "verifier not loaded")
        out = verifier(
            {
                "prompt": req.prompt,
                "response": req.response,
                "seed": req.seed,
                "fingerprint": req.fingerprint,
            }
        )
        return VerifyResponse(
            fp_cos=out["fp_cos"],
            n_tokens=out["n_tokens"],
            truncated=out["truncated"],
            passed=out["passed"],
            reason=out["reason"],
            model_id=STATE["model_id"],
        )

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HuggingFace model id of the model to verify")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--stake", type=int, default=1_000_000_000, help="Stake in lamports (placeholder)")
    args = p.parse_args()

    app = build_app(args.model, args.host, args.port, args.stake)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
