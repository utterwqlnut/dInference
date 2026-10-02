"""Verifier daemon.

Subscribes to the dInference program's event stream. When a ChallengeOpened
event lists this verifier in the committee, fetches prompt + response from
Arweave, runs the local fingerprint check, submits `verifier_vote`, and
opportunistically tries `finalize_challenge` once the outcome is decidable.

Usage:
    python -m off_chain_verifier.ramp --model Qwen/Qwen2.5-1.5B-Instruct

Env:
    DINFERENCE_KEYPAIR_PATH / DINFERENCE_PROGRAM_ID / SOLANA_RPC_URL / SOLANA_WS_URL
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from solders.pubkey import Pubkey

from chain import arweave
from chain.client import (
    DInferenceClient, challenge_pda, provider_pda, verifier_pda,
)
from chain.events import subscribe_events
from .fingerprint_pipeline import FingerprintVerifier

log = logging.getLogger("verifier.ramp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class VerifierDaemon:
    def __init__(self, model_id: str):
        self.model_id = model_id
        log.info("loading verifier model: %s", model_id)
        self.verifier = FingerprintVerifier(model_id)
        self.client = DInferenceClient()

    async def handle_challenge_opened(self, ev) -> None:
        bounty_pk = Pubkey.from_bytes(ev.data.bounty)
        committee = [Pubkey.from_bytes(p) for p in ev.data.committee]
        if self.client.operator not in committee:
            return

        try:
            # Fetch bounty + commitment to get prompt/response txids + seed + fingerprint.
            bounty = await self.client.fetch_bounty(bounty_pk)
            if bounty is None:
                log.warning("[%s] bounty gone", bounty_pk)
                return
            commitment_pk = Pubkey.from_bytes(bounty.response_commitment)
            commitment = await self.client.fetch_commitment(commitment_pk)
            if commitment is None:
                log.warning("[%s] commitment gone", bounty_pk)
                return

            # Fetch blobs. If fetch fails for ANY reason, auto-vote guilty —
            # provider didn't post what they claimed.
            try:
                prompt_bytes = arweave.fetch(bytes(bounty.prompt_txid))
                response_bytes = arweave.fetch(bytes(commitment.response_txid))
            except arweave.ArweaveError as e:
                log.warning("[%s] arweave fetch failed → voting guilty: %s", bounty_pk, e)
                tx = await self.client.verifier_vote(bounty_pk, verdict=True, fp_cos=-1.0)
                log.info("[%s] vote (unretrievable): %s", bounty_pk, tx.signature)
                return

            # Run fingerprint check.
            seed_int = int.from_bytes(bytes(bounty.seed_at_submission)[:8], "little")
            result = self.verifier({
                "prompt": prompt_bytes.decode(),
                "response": response_bytes.decode(),
                "seed": seed_int,
                "fingerprint": list(commitment.fingerprint),
            })
            verdict = not result["passed"]
            log.info("[%s] verdict=%s fp_cos=%.4f", bounty_pk, verdict, result["fp_cos"])

            tx = await self.client.verifier_vote(bounty_pk, verdict, result["fp_cos"])
            log.info("[%s] voted: %s", bounty_pk, tx.signature)

            # Opportunistic finalize.
            await self._maybe_finalize(bounty_pk, bounty)
        except Exception as e:
            log.exception("[%s] failed: %s", bounty_pk, e)

    async def _maybe_finalize(self, bounty_pk: Pubkey, bounty) -> None:
        """Check if the challenge is decidable; if so, call finalize_challenge."""
        try:
            ch_pk = challenge_pda(bounty_pk)
            ch = await self.client.fetch_challenge(ch_pk)
            cfg = await self.client.fetch_config()
            if ch is None or cfg is None:
                return
            if ch.status != 0:  # not Open
                return
            committee_len = len(ch.verifier_committee)
            guilty = sum(1 for v in ch.verdicts if v.verdict)
            honest = sum(1 for v in ch.verdicts if not v.verdict)
            remaining = committee_len - (guilty + honest)
            quorum = (committee_len * cfg.quorum_fraction) // 10_000
            decidable = guilty >= quorum or (guilty + remaining) < quorum
            if not decidable:
                return

            winner_pk = Pubkey.from_bytes(bounty.winner)
            user_pk = Pubkey.from_bytes(bounty.user)
            committee_verifier_pdas = [
                verifier_pda(Pubkey.from_bytes(op)) for op in ch.verifier_committee
            ]
            log.info("[%s] decidable, calling finalize_challenge", bounty_pk)
            tx = await self.client.finalize_challenge(
                bounty_pk, self.model_id, winner_pk, user_pk, committee_verifier_pdas,
            )
            log.info("[%s] finalized: %s", bounty_pk, tx.signature)
        except Exception as e:
            # Lost the race or chain state changed; another actor finalized.
            log.info("[%s] finalize skipped: %s", bounty_pk, e)

    async def run(self) -> None:
        log.info("verifier daemon start: model=%s operator=%s",
                 self.model_id, self.client.operator)
        try:
            async for ev in subscribe_events():
                if ev.name == "ChallengeOpened":
                    await self.handle_challenge_opened(ev)
        finally:
            await self.client.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    daemon = VerifierDaemon(args.model)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
