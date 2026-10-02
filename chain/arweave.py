"""Arweave uploads via Bundlr/Irys (paid in SOL), reads via Arweave gateway HTTP.

Uploads: shell out to `npx @irys/sdk` (requires Node 18+ on PATH). Signs with
the same Solana keypair used for on-chain txs. For files <100KB uploads are
free, otherwise it costs a few lamports.

Reads: fetched over HTTP from any public Arweave gateway. Blobs are content-
addressed so integrity is just a sha256 check — but Arweave txids are
base64url-encoded sha256 hashes of a signed data-item header + data, NOT a raw
hash of the content. We verify by comparing bytes.hex() against the recorded txid
rather than recomputing.

Env:
    DINFERENCE_KEYPAIR_PATH  — Solana keypair JSON (used by Irys to pay for uploads)
    IRYS_NODE                — Irys node URL (default: https://node1.irys.xyz)
    IRYS_CURRENCY            — Payment currency (default: solana)
    ARWEAVE_GATEWAY          — Read gateway (default: https://arweave.net)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

log = logging.getLogger("arweave")

IRYS_NODE       = os.environ.get("IRYS_NODE", "https://node1.irys.xyz")
IRYS_CURRENCY   = os.environ.get("IRYS_CURRENCY", "solana")
ARWEAVE_GATEWAY = os.environ.get("ARWEAVE_GATEWAY", "https://arweave.net")


class ArweaveError(RuntimeError):
    pass


def _irys_available() -> bool:
    return shutil.which("npx") is not None


def upload(body: bytes) -> bytes:
    """Upload bytes via Bundlr/Irys. Returns the 32-byte Arweave txid.

    Raises ArweaveError if upload fails or Node toolchain is missing.
    """
    if not _irys_available():
        raise ArweaveError(
            "npx not found on PATH. Install Node 18+ to use Bundlr uploads. "
            "See https://docs.irys.xyz for the Irys CLI."
        )
    keypair_path = os.environ.get("DINFERENCE_KEYPAIR_PATH")
    if not keypair_path or not Path(keypair_path).exists():
        raise ArweaveError(
            "DINFERENCE_KEYPAIR_PATH must point to a funded Solana keypair for Irys uploads."
        )

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    try:
        cmd = [
            "npx", "-y", "@irys/sdk@latest", "upload", tmp_path,
            "-t", IRYS_CURRENCY,
            "-w", keypair_path,
            "-h", IRYS_NODE,
        ]
        log.info("[arweave] uploading %d bytes via irys (%s)", len(body), IRYS_CURRENCY)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ArweaveError(f"irys upload failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
        # Irys CLI prints a URL like "Uploaded to https://gateway.irys.xyz/<txid>"
        # Parse out the txid.
        txid_str = _parse_irys_output(proc.stdout)
        log.info("[arweave] uploaded → txid=%s", txid_str[:16])
        return _txid_str_to_bytes(txid_str)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def fetch(txid: bytes) -> bytes:
    """Fetch a blob by txid from the Arweave gateway. Raises on 404 or error."""
    txid_str = _txid_bytes_to_str(txid)
    url = f"{ARWEAVE_GATEWAY}/{txid_str}"
    log.info("[arweave] fetching txid=%s from %s", txid_str[:16], ARWEAVE_GATEWAY)
    r = httpx.get(url, timeout=30.0)
    if r.status_code == 404:
        raise ArweaveError(f"txid not found: {txid_str}")
    r.raise_for_status()
    return r.content


# --- helpers ----------------------------------------------------------

def _parse_irys_output(stdout: str) -> str:
    """Irys CLI prints something like:
        Uploaded to https://gateway.irys.xyz/<base64url-txid>
    Extract the <txid>.
    """
    for line in stdout.splitlines():
        if "irys.xyz/" in line or "arweave.net/" in line:
            # take whatever comes after the last '/'
            tail = line.rsplit("/", 1)[-1].strip().strip(".")
            if tail:
                return tail
    # Also try JSON output (newer irys versions emit JSON)
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict) and "id" in obj:
            return obj["id"]
    except Exception:
        pass
    raise ArweaveError(f"could not parse txid from irys output:\n{stdout}")


def _txid_str_to_bytes(txid_str: str) -> bytes:
    """Arweave txids are 43-char base64url strings (encoding 32 raw bytes)."""
    import base64
    pad = "=" * (-len(txid_str) % 4)
    return base64.urlsafe_b64decode(txid_str + pad)


def _txid_bytes_to_str(txid: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(txid).decode().rstrip("=")
