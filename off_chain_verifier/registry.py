"""Verifier on-chain registration.

Usage:
    python -m off_chain_verifier.registry \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --stake 1000000000
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from chain.client import DInferenceClient

log = logging.getLogger("verifier.registry")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def register(model_id: str, stake_lamports: int) -> str:
    client = DInferenceClient()
    try:
        log.info("registering verifier operator=%s model=%s stake=%d",
                 client.operator, model_id, stake_lamports)
        result = await client.register_verifier(model_id, stake_lamports)
        log.info("registered: %s", result.signature)
        return result.signature
    finally:
        await client.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stake", type=int, required=True)
    args = p.parse_args()
    sig = asyncio.run(register(args.model, args.stake))
    print(sig)


if __name__ == "__main__":
    main()
