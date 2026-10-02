"""Provider on-chain registration.

Submits `register_provider` and prints the resulting tx signature.

Usage:
    python -m off_chain_inference.registry \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --stake 10000000000
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from chain.client import DInferenceClient

log = logging.getLogger("provider.registry")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def register(model_id: str, stake_lamports: int) -> str:
    client = DInferenceClient()
    try:
        log.info("registering provider operator=%s model=%s stake=%d",
                 client.operator, model_id, stake_lamports)
        try:
            result = await client.register_provider(model_id, stake_lamports)
        except Exception as e:
            if "already in use" in str(e):
                log.info("already registered — ok")
                return "ALREADY_REGISTERED"
            raise
        log.info("registered: %s", result.signature)
        return result.signature
    finally:
        await client.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stake", type=int, required=True, help="lamports")
    args = p.parse_args()
    sig = asyncio.run(register(args.model, args.stake))
    print(sig)


if __name__ == "__main__":
    main()
