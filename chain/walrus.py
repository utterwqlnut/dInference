"""Content-addressed blob storage via Walrus (Sui's decentralized storage).

Public testnet endpoints — no auth, no SDK, pure HTTP, uploads and reads land
in seconds. Blob IDs are base64url-encoded content hashes we store on-chain
(same role as Arweave txids in the original design).

Env:
    WALRUS_PUBLISHER   default: https://publisher.walrus-testnet.walrus.space
    WALRUS_AGGREGATOR  default: https://aggregator.walrus-testnet.walrus.space
"""

from __future__ import annotations

import base64
import logging
import os

import httpx

log = logging.getLogger("blobs")

PUBLISHER = os.environ.get(
    "WALRUS_PUBLISHER",
    "https://publisher.walrus-testnet.walrus.space",
).rstrip("/")

AGGREGATOR = os.environ.get(
    "WALRUS_AGGREGATOR",
    "https://aggregator.walrus-testnet.walrus.space",
).rstrip("/")

# Store epochs — 1 is cheapest and plenty for the challenge window.
STORE_EPOCHS = int(os.environ.get("WALRUS_EPOCHS", "1"))


class WalrusError(RuntimeError):
    pass


def upload(body: bytes) -> bytes:
    """PUT /v1/blobs. Returns 32-byte raw blob-id."""
    url = f"{PUBLISHER}/v1/blobs?epochs={STORE_EPOCHS}"
    r = httpx.put(url, content=body, timeout=60.0)
    if r.status_code >= 300:
        raise WalrusError(f"walrus upload failed ({r.status_code}): {r.text}")

    data = r.json()
    # Response is one of two shapes:
    #   {"newlyCreated": {"blobObject": {..., "blobId": "..."}}}
    #   {"alreadyCertified": {"blobId": "..."}}
    if "newlyCreated" in data:
        blob_id = data["newlyCreated"]["blobObject"]["blobId"]
    elif "alreadyCertified" in data:
        blob_id = data["alreadyCertified"]["blobId"]
    else:
        raise WalrusError(f"unexpected walrus response: {data}")

    log.info("[walrus] uploaded %d bytes → %s", len(body), blob_id[:16])
    return _b64url_to_bytes(blob_id)


def fetch(txid: bytes) -> bytes:
    """GET /v1/blobs/<id>. Returns raw bytes."""
    blob_id = _bytes_to_b64url(txid)
    url = f"{AGGREGATOR}/v1/blobs/{blob_id}"
    log.info("[walrus] fetching %s", blob_id[:16])
    r = httpx.get(url, timeout=60.0, follow_redirects=True)
    if r.status_code == 404:
        raise WalrusError(f"blob not found: {blob_id}")
    r.raise_for_status()
    return r.content


def _b64url_to_bytes(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _bytes_to_b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")
