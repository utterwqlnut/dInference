"""User-facing SDK for dInference.

    import asyncio
    from sdk import DInference

    async def main():
        d = DInference(keypair_path="~/.config/solana/id.json")

        bounty_pda, response_text = await d.inference(
            prompt="Explain photosynthesis.",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            bounty_lamports=1_000_000,
        )
        print(response_text)

        # Only if the response looks wrong:
        # await d.challenge(bounty_pda, bond_lamports=500_000)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from typing import Optional

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from chain import arweave
from chain.client import DInferenceClient, load_keypair, response_pda

log = logging.getLogger("sdk")


@dataclass
class InferenceResult:
    bounty_pda: Pubkey
    response_text: Optional[str]
    response_txid: Optional[bytes]


class DInference:
    def __init__(self, keypair_path: Optional[str] = None,
                 keypair: Optional[Keypair] = None):
        kp = keypair or load_keypair(keypair_path)
        self.client = DInferenceClient(keypair=kp)

    @property
    def wallet(self) -> Pubkey:
        return self.client.operator

    async def close(self) -> None:
        await self.client.close()

    async def inference(self, prompt: str, model_id: str, bounty_lamports: int,
                        wait_for_response: bool = True, poll_s: int = 3,
                        timeout_s: int = 600) -> InferenceResult:
        # 1. Upload prompt.
        prompt_txid = arweave.upload(prompt.encode())

        # 2. Post bounty.
        nonce = secrets.randbits(63)
        bounty_pk, tx = await self.client.post_bounty(
            nonce=nonce, prompt_txid=prompt_txid,
            bounty_lamports=bounty_lamports, model_id=model_id,
        )
        log.info("posted bounty %s (tx=%s)", bounty_pk, tx.signature)

        if not wait_for_response:
            return InferenceResult(bounty_pda=bounty_pk, response_text=None, response_txid=None)

        # 3. Poll for response commitment.
        resp_pk = response_pda(bounty_pk)
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            commitment = await self.client.fetch_commitment(resp_pk)
            if commitment is not None:
                response_txid = bytes(commitment.response_txid)
                response_bytes = arweave.fetch(response_txid)
                return InferenceResult(
                    bounty_pda=bounty_pk,
                    response_text=response_bytes.decode(),
                    response_txid=response_txid,
                )
            await asyncio.sleep(poll_s)

        log.warning("timed out waiting for response on %s", bounty_pk)
        return InferenceResult(bounty_pda=bounty_pk, response_text=None, response_txid=None)

    async def challenge(self, bounty_pda: Pubkey, model_id: str,
                        bond_lamports: int) -> str:
        tx = await self.client.challenge(bounty_pda, model_id, bond_lamports)
        log.info("challenged %s (tx=%s)", bounty_pda, tx.signature)
        return tx.signature
