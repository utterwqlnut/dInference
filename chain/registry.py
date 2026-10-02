"""Placeholder Solana registration for providers and verifiers.

In production this would:
- Load the operator's Keypair from a local file.
- Build a transaction calling our on-chain program's `register_provider` or
  `register_verifier` instruction with (model_id, stake_amount, endpoint_url).
- Submit via AsyncClient and poll for confirmation.
- Store the returned registration PDA.

For now we stub the chain call out and only log what *would* be submitted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from solders.keypair import Keypair
from solders.pubkey import Pubkey

log = logging.getLogger(__name__)

# TODO: replace with the real deployed program id once the Anchor program ships.
DINFERENCE_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")

DEFAULT_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com")


@dataclass
class Registration:
    role: str              # "provider" or "verifier"
    operator_pubkey: str
    model_id: str
    endpoint_url: str
    stake_lamports: int
    program_id: str
    rpc_url: str
    tx_signature: str      # placeholder


def _load_keypair() -> Keypair:
    """Load the operator keypair. For now we generate a throwaway one; in
    production this would be `Keypair.from_bytes(json.loads(open(path).read()))`."""
    path = os.environ.get("DINFERENCE_KEYPAIR_PATH")
    if path and os.path.exists(path):
        import json
        with open(path) as f:
            raw = json.load(f)
        return Keypair.from_bytes(bytes(raw))
    log.warning("DINFERENCE_KEYPAIR_PATH unset; using an ephemeral keypair (dev only)")
    return Keypair()


def _register(role: str, model_id: str, endpoint_url: str, stake_lamports: int) -> Registration:
    kp = _load_keypair()
    reg = Registration(
        role=role,
        operator_pubkey=str(kp.pubkey()),
        model_id=model_id,
        endpoint_url=endpoint_url,
        stake_lamports=stake_lamports,
        program_id=str(DINFERENCE_PROGRAM_ID),
        rpc_url=DEFAULT_RPC_URL,
        tx_signature="PLACEHOLDER_TX_SIGNATURE",
    )

    # TODO: replace with a real on-chain tx. Shape of the call would be:
    #   client = AsyncClient(DEFAULT_RPC_URL)
    #   ix = Instruction(
    #       program_id=DINFERENCE_PROGRAM_ID,
    #       accounts=[AccountMeta(kp.pubkey(), True, True), ...],
    #       data=encode_register_ix(role, model_id, endpoint_url, stake_lamports),
    #   )
    #   tx = Transaction([kp], Message([ix], kp.pubkey()), recent_blockhash)
    #   sig = await client.send_transaction(tx)
    log.info(
        "[chain] (placeholder) would register %s: operator=%s model=%s endpoint=%s stake=%d",
        role, reg.operator_pubkey, model_id, endpoint_url, stake_lamports,
    )
    return reg


def register_provider(model_id: str, endpoint_url: str, stake_lamports: int = 1_000_000_000) -> Registration:
    return _register("provider", model_id, endpoint_url, stake_lamports)


def register_verifier(model_id: str, endpoint_url: str, stake_lamports: int = 1_000_000_000) -> Registration:
    return _register("verifier", model_id, endpoint_url, stake_lamports)
