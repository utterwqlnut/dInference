"""Solana client for dInference.

Builds, signs, and submits real transactions against a deployed Anchor program.
Also fetches + decodes on-chain accounts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment, Confirmed, Finalized
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.transaction import VersionedTransaction

from . import discriminators as D
from . import layouts as L

log = logging.getLogger("chain")

# ---- Env configuration ----

DEFAULT_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com")
DEFAULT_WS_URL  = os.environ.get("SOLANA_WS_URL",  "wss://api.devnet.solana.com")

_prog_id_str = os.environ.get("DINFERENCE_PROGRAM_ID", "11111111111111111111111111111111")
DINFERENCE_PROGRAM_ID = Pubkey.from_string(_prog_id_str)

# ---- PDA seeds (must match constants.rs) ----

VAULT_SEED     = b"vault"
CONFIG_SEED    = b"config"
SEED_SEED      = b"seed"
MODEL_SEED     = b"model"
POOL_SEED      = b"pool"
PROVIDER_SEED  = b"provider"
VERIFIER_SEED  = b"verifier"
BOUNTY_SEED    = b"bounty"
RESPONSE_SEED  = b"resp"
CHALLENGE_SEED = b"chal"


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def model_id_hash(model_id: str) -> bytes:
    return _sha256(model_id.encode())


def _find_pda(seeds: list[bytes]) -> Pubkey:
    pda, _ = Pubkey.find_program_address(seeds, DINFERENCE_PROGRAM_ID)
    return pda


# ---- PDA derivation ----

def config_pda() -> Pubkey:            return _find_pda([CONFIG_SEED])
def vault_pda() -> Pubkey:             return _find_pda([VAULT_SEED])
def seed_pda() -> Pubkey:              return _find_pda([SEED_SEED])
def model_pda(model_id: str) -> Pubkey:
    return _find_pda([MODEL_SEED, model_id_hash(model_id)])
def pool_pda(model_id: str) -> Pubkey:
    return _find_pda([POOL_SEED, model_id_hash(model_id)])
def provider_pda(operator: Pubkey) -> Pubkey:
    return _find_pda([PROVIDER_SEED, bytes(operator)])
def verifier_pda(operator: Pubkey) -> Pubkey:
    return _find_pda([VERIFIER_SEED, bytes(operator)])
def bounty_pda(user: Pubkey, nonce: int) -> Pubkey:
    return _find_pda([BOUNTY_SEED, bytes(user), nonce.to_bytes(8, "little")])
def response_pda(bounty: Pubkey) -> Pubkey:
    return _find_pda([RESPONSE_SEED, bytes(bounty)])
def challenge_pda(bounty: Pubkey) -> Pubkey:
    return _find_pda([CHALLENGE_SEED, bytes(bounty)])


# ---- Keypair ----

def load_keypair(path: Optional[str] = None) -> Keypair:
    path = path or os.environ.get("DINFERENCE_KEYPAIR_PATH")
    if path and Path(path).expanduser().exists():
        with open(Path(path).expanduser()) as f:
            raw = json.load(f)
        return Keypair.from_bytes(bytes(raw))
    log.warning("DINFERENCE_KEYPAIR_PATH unset; using an ephemeral keypair (dev only)")
    return Keypair()


# ---- Account meta helpers ----

def _rw(pk: Pubkey) -> AccountMeta: return AccountMeta(pk, is_signer=False, is_writable=True)
def _ro(pk: Pubkey) -> AccountMeta: return AccountMeta(pk, is_signer=False, is_writable=False)
def _sw(pk: Pubkey) -> AccountMeta: return AccountMeta(pk, is_signer=True,  is_writable=True)
def _sr(pk: Pubkey) -> AccountMeta: return AccountMeta(pk, is_signer=True,  is_writable=False)


# ---- Account decoders ----

@dataclass
class DecodedAccount:
    name: str
    data: Any   # construct Container


_ACCOUNT_TABLE = {
    D.ACC_GLOBAL_CONFIG:       ("GlobalConfig",       L.GLOBAL_CONFIG_LAYOUT),
    D.ACC_ROTATING_SEED:       ("RotatingSeed",       L.ROTATING_SEED_LAYOUT),
    D.ACC_STAKE_VAULT:         ("StakeVault",         L.STAKE_VAULT_LAYOUT),
    D.ACC_MODEL_INFO:          ("ModelInfo",          L.MODEL_INFO_LAYOUT),
    D.ACC_VERIFIER_POOL:       ("VerifierPool",       L.VERIFIER_POOL_LAYOUT),
    D.ACC_PROVIDER_ACCOUNT:    ("ProviderAccount",    L.PROVIDER_ACCOUNT_LAYOUT),
    D.ACC_VERIFIER_ACCOUNT:    ("VerifierAccount",    L.VERIFIER_ACCOUNT_LAYOUT),
    D.ACC_BOUNTY:              ("Bounty",             L.BOUNTY_LAYOUT),
    D.ACC_RESPONSE_COMMITMENT: ("ResponseCommitment", L.RESPONSE_COMMITMENT_LAYOUT),
    D.ACC_CHALLENGE:           ("Challenge",          L.CHALLENGE_LAYOUT),
}


def decode_account(raw: bytes) -> Optional[DecodedAccount]:
    if len(raw) < 8:
        return None
    disc = raw[:8]
    entry = _ACCOUNT_TABLE.get(disc)
    if not entry:
        return None
    name, layout = entry
    data = layout.parse(raw[8:])
    return DecodedAccount(name=name, data=data)


# ---- Client ----

@dataclass
class TxResult:
    signature: str
    confirmed: bool


class DInferenceClient:
    """Builds and submits transactions. Fetches and decodes accounts."""

    def __init__(self, keypair: Optional[Keypair] = None,
                 rpc_url: str = DEFAULT_RPC_URL,
                 commitment: Commitment = Confirmed):
        self.keypair = keypair or load_keypair()
        self.rpc_url = rpc_url
        self.commitment = commitment
        self._rpc = AsyncClient(rpc_url, commitment=commitment)

    @property
    def operator(self) -> Pubkey:
        return self.keypair.pubkey()

    async def close(self) -> None:
        await self._rpc.close()

    # ---- Low-level tx submission ----

    async def _send(self, ix: Instruction, extra_signers: Sequence[Keypair] = ()) -> TxResult:
        bh = (await self._rpc.get_latest_blockhash()).value.blockhash
        signers = [self.keypair, *extra_signers]
        msg = MessageV0.try_compile(self.operator, [ix], [], bh)
        tx = VersionedTransaction(msg, signers)
        sig = (await self._rpc.send_transaction(tx)).value
        await self._rpc.confirm_transaction(sig, commitment=self.commitment)
        return TxResult(signature=str(sig), confirmed=True)

    async def _fetch_account(self, pda: Pubkey) -> Optional[DecodedAccount]:
        resp = await self._rpc.get_account_info(pda, commitment=self.commitment)
        if resp.value is None:
            return None
        return decode_account(bytes(resp.value.data))

    # ---- Admin ----

    async def init_config(self, args: dict) -> TxResult:
        data = D.IX_INIT_CONFIG + L.INIT_CONFIG_ARGS.build(args)
        accounts = [
            _sw(self.operator),
            _rw(config_pda()),
            _rw(seed_pda()),
            _rw(vault_pda()),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    async def init_model(self, model_id: str, hidden_dim: int) -> TxResult:
        from borsh_construct import CStruct, String, U32
        args_layout = CStruct("model_id" / String, "hidden_dim" / U32)
        data = D.IX_INIT_MODEL + args_layout.build({"model_id": model_id, "hidden_dim": hidden_dim})
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _rw(model_pda(model_id)),
            _rw(pool_pda(model_id)),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    # ---- Operator lifecycle ----

    async def register_provider(self, model_id: str, stake_lamports: int) -> TxResult:
        from borsh_construct import CStruct, String, U64
        args_layout = CStruct("model_id" / String, "stake" / U64)
        data = D.IX_REGISTER_PROVIDER + args_layout.build(
            {"model_id": model_id, "stake": stake_lamports}
        )
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _ro(model_pda(model_id)),
            _rw(provider_pda(self.operator)),
            _rw(vault_pda()),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    async def register_verifier(self, model_id: str, stake_lamports: int) -> TxResult:
        from borsh_construct import CStruct, String, U64
        args_layout = CStruct("model_id" / String, "stake" / U64)
        data = D.IX_REGISTER_VERIFIER + args_layout.build(
            {"model_id": model_id, "stake": stake_lamports}
        )
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _ro(model_pda(model_id)),
            _rw(verifier_pda(self.operator)),
            _rw(pool_pda(model_id)),
            _rw(vault_pda()),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    # ---- Bounty lifecycle ----

    async def post_bounty(self, nonce: int, prompt_txid: bytes,
                          bounty_lamports: int, model_id: str) -> tuple[Pubkey, TxResult]:
        from borsh_construct import CStruct, U64
        from construct import Bytes as RawBytes
        args_layout = CStruct(
            "nonce" / U64,
            "prompt_txid" / RawBytes(32),
            "bounty_lamports" / U64,
        )
        data = D.IX_POST_BOUNTY + args_layout.build({
            "nonce": nonce,
            "prompt_txid": prompt_txid,
            "bounty_lamports": bounty_lamports,
        })
        b_pda = bounty_pda(self.operator, nonce)
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _ro(model_pda(model_id)),
            _ro(seed_pda()),
            _rw(b_pda),
            _rw(vault_pda()),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        tx = await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))
        return b_pda, tx

    async def claim_bounty(self, bounty: Pubkey, response_txid: bytes,
                           fingerprint: list[float]) -> TxResult:
        from borsh_construct import CStruct, Vec, F32
        from construct import Bytes as RawBytes
        args_layout = CStruct(
            "response_txid" / RawBytes(32),
            "fingerprint" / Vec(F32),
        )
        data = D.IX_CLAIM_BOUNTY + args_layout.build({
            "response_txid": response_txid,
            "fingerprint": fingerprint,
        })
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _rw(provider_pda(self.operator)),
            _rw(bounty),
            _rw(response_pda(bounty)),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    async def finalize_payment(self, bounty: Pubkey, winner: Pubkey) -> TxResult:
        data = D.IX_FINALIZE_PAYMENT
        accounts = [
            _sr(self.operator),
            _ro(config_pda()),
            _rw(bounty),
            _ro(response_pda(bounty)),
            _rw(provider_pda(winner)),
            _rw(winner),
            _rw(vault_pda()),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    # ---- Challenge path ----

    async def challenge(self, bounty: Pubkey, model_id: str,
                        bond_lamports: int) -> TxResult:
        from borsh_construct import CStruct, U64
        args_layout = CStruct("bond" / U64)
        data = D.IX_CHALLENGE + args_layout.build({"bond": bond_lamports})
        accounts = [
            _sw(self.operator),
            _ro(config_pda()),
            _ro(seed_pda()),
            _rw(bounty),
            _ro(response_pda(bounty)),
            _rw(pool_pda(model_id)),
            _rw(challenge_pda(bounty)),
            _rw(vault_pda()),
            _ro(SYSTEM_PROGRAM_ID),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    async def verifier_vote(self, bounty: Pubkey, verdict: bool, fp_cos: float) -> TxResult:
        from borsh_construct import CStruct, Bool, I32
        args_layout = CStruct("verdict" / Bool, "fp_cos_scaled" / I32)
        fp_scaled = int(fp_cos * 10_000)
        data = D.IX_VERIFIER_VOTE + args_layout.build({
            "verdict": verdict, "fp_cos_scaled": fp_scaled,
        })
        accounts = [
            _sr(self.operator),
            _rw(challenge_pda(bounty)),
        ]
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    async def finalize_challenge(self, bounty: Pubkey, model_id: str,
                                 winner: Pubkey, user: Pubkey,
                                 committee_verifier_pdas: list[Pubkey]) -> TxResult:
        data = D.IX_FINALIZE_CHALLENGE
        accounts = [
            _sr(self.operator),
            _ro(config_pda()),
            _rw(challenge_pda(bounty)),
            _rw(bounty),
            _ro(response_pda(bounty)),
            _rw(provider_pda(winner)),
            _rw(pool_pda(model_id)),
            _rw(winner),
            _rw(user),
            _rw(vault_pda()),
        ]
        # VerifierAccount PDAs as remaining_accounts
        for vpda in committee_verifier_pdas:
            accounts.append(_rw(vpda))
        return await self._send(Instruction(DINFERENCE_PROGRAM_ID, data, accounts))

    # ---- Account fetchers ----

    async def fetch_bounty(self, bounty_pk: Pubkey):
        acc = await self._fetch_account(bounty_pk)
        if acc is None or acc.name != "Bounty":
            return None
        return acc.data

    async def fetch_commitment(self, commitment_pk: Pubkey):
        acc = await self._fetch_account(commitment_pk)
        if acc is None or acc.name != "ResponseCommitment":
            return None
        return acc.data

    async def fetch_challenge(self, challenge_pk: Pubkey):
        acc = await self._fetch_account(challenge_pk)
        if acc is None or acc.name != "Challenge":
            return None
        return acc.data

    async def fetch_pool(self, model_id: str):
        acc = await self._fetch_account(pool_pda(model_id))
        if acc is None or acc.name != "VerifierPool":
            return None
        return acc.data

    async def fetch_config(self):
        acc = await self._fetch_account(config_pda())
        if acc is None or acc.name != "GlobalConfig":
            return None
        return acc.data
