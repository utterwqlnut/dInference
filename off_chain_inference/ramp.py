"""Provider daemon.

Subscribes to the dInference program's event stream, grabs the first
BountyPosted for its model, runs inference locally, uploads the response to
Arweave via Bundlr/Irys, and submits `claim_bounty`. First-come-first-served
with no filtering by bounty amount (naive FCFS).

Usage:
    python -m off_chain_inference.ramp --model Qwen/Qwen2.5-1.5B-Instruct

Env:
    DINFERENCE_KEYPAIR_PATH  — operator Solana keypair (also funds Irys uploads)
    DINFERENCE_PROGRAM_ID    — deployed program id
    SOLANA_RPC_URL / SOLANA_WS_URL
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from solders.pubkey import Pubkey

from chain import arweave
from chain.client import DInferenceClient, model_id_hash
from chain.events import subscribe_events
from .fingerprint_pipeline import FingerprintProvider

log = logging.getLogger("provider.ramp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class ProviderDaemon:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model_hash = model_id_hash(model_id)
        log.info("loading model: %s", model_id)
        self.provider = FingerprintProvider(model_id)
        self.client = DInferenceClient()
        self.busy = False

    async def handle_bounty_posted(self, ev) -> None:
        if self.busy:
            log.info("busy — skipping bounty %s",
                     Pubkey.from_bytes(ev.data.bounty))
            return
        if bytes(ev.data.model_id_hash) != self._model_hash:
            return  # Different model.
        self.busy = True
        bounty_pk = Pubkey.from_bytes(ev.data.bounty)
        try:
            # Chain stores the per-request seed on the Bounty; we need to fetch
            # the account to get it (event doesn't carry seed_at_submission).
            bounty = await self.client.fetch_bounty(bounty_pk)
            if bounty is None:
                log.warning("[%s] bounty disappeared before claim", bounty_pk)
                return

            # 1. Fetch prompt from Arweave.
            log.info("[%s] fetching prompt from Arweave", bounty_pk)
            prompt_bytes = arweave.fetch(bytes(ev.data.prompt_txid))
            prompt = prompt_bytes.decode()

            # 2. Run inference.
            seed_int = int.from_bytes(bytes(bounty.seed_at_submission)[:8], "little")
            log.info("[%s] running inference (prompt=%d bytes)", bounty_pk, len(prompt_bytes))
            out = self.provider.generate(prompt, seed=seed_int)

            # 3. Upload response to Arweave.
            response_bytes = out["text"].encode()
            response_txid = arweave.upload(response_bytes)

            # 4. claim_bounty on-chain.
            log.info("[%s] submitting claim_bounty", bounty_pk)
            tx = await self.client.claim_bounty(
                bounty_pk, response_txid, out["fingerprint"]
            )
            log.info("[%s] claimed: %s", bounty_pk, tx.signature)
        except Exception as e:
            log.exception("[%s] failed: %s", bounty_pk, e)
        finally:
            self.busy = False

    async def run(self) -> None:
        log.info("provider daemon start: model=%s operator=%s", self.model_id, self.client.operator)
        try:
            async for ev in subscribe_events():
                if ev.name == "BountyPosted":
                    await self.handle_bounty_posted(ev)
        finally:
            await self.client.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    daemon = ProviderDaemon(args.model)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
