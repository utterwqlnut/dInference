"""Anchor event decoder + WebSocket subscription.

Anchor emits events as log messages of the form:
    "Program data: <base64(discriminator + borsh_data)>"

We subscribe to logsSubscribe on our program id, iterate the logs on each
notification, and decode anything matching one of our event discriminators.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from solana.rpc.websocket_api import connect
from solders.rpc.config import RpcTransactionLogsFilterMentions
from solders.pubkey import Pubkey

from . import discriminators as D
from . import layouts as L
from .client import DINFERENCE_PROGRAM_ID, DEFAULT_WS_URL

log = logging.getLogger("events")

_EVENT_TABLE = {
    D.EVT_BOUNTY_POSTED:      ("BountyPosted",      L.BOUNTY_POSTED_EVENT),
    D.EVT_BOUNTY_CLAIMED:     ("BountyClaimed",     L.BOUNTY_CLAIMED_EVENT),
    D.EVT_CHALLENGE_OPENED:   ("ChallengeOpened",   L.CHALLENGE_OPENED_EVENT),
    D.EVT_VOTE_SUBMITTED:     ("VoteSubmitted",     L.VOTE_SUBMITTED_EVENT),
    D.EVT_CHALLENGE_RESOLVED: ("ChallengeResolved", L.CHALLENGE_RESOLVED_EVENT),
}


@dataclass
class DecodedEvent:
    name: str
    data: object   # construct Container


def decode_event_log(line: str) -> Optional[DecodedEvent]:
    """Returns a DecodedEvent if `line` is one of our events, else None."""
    prefix = "Program data: "
    if not line.startswith(prefix):
        return None
    try:
        raw = base64.b64decode(line[len(prefix):])
    except Exception:
        return None
    if len(raw) < 8:
        return None
    disc = raw[:8]
    entry = _EVENT_TABLE.get(disc)
    if not entry:
        return None
    name, layout = entry
    try:
        data = layout.parse(raw[8:])
    except Exception as e:
        log.warning("event %s decode failed: %s", name, e)
        return None
    return DecodedEvent(name=name, data=data)


async def subscribe_events(ws_url: str = DEFAULT_WS_URL) -> AsyncIterator[DecodedEvent]:
    """Async generator yielding DecodedEvent objects as they arrive from the chain."""
    while True:
        try:
            async with connect(ws_url) as ws:
                await ws.logs_subscribe(
                    filter_=RpcTransactionLogsFilterMentions(DINFERENCE_PROGRAM_ID),
                )
                log.info("subscribed to program logs: %s", DINFERENCE_PROGRAM_ID)
                async for msg_batch in ws:
                    # Each msg from solana-py is a list of notifications.
                    msgs = msg_batch if isinstance(msg_batch, list) else [msg_batch]
                    for m in msgs:
                        logs = getattr(getattr(m, "result", None), "value", None)
                        if logs is None:
                            continue
                        for line in logs.logs:
                            ev = decode_event_log(line)
                            if ev is not None:
                                yield ev
        except Exception as e:
            log.warning("ws error, reconnecting in 5s: %s", e)
            await asyncio.sleep(5)
