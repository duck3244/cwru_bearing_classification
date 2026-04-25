# ============================================================================
# 파일: app/services/broadcaster.py
# 설명: SSE fan-out — 다중 구독자에게 epoch 이벤트 푸시
# ============================================================================

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class Broadcaster:
    """
    learning thread → asyncio loop publish 안전.
    각 SSE 연결마다 자신의 큐를 가짐. 학습 워커는 publish_threadsafe()로 푸시.
    """

    def __init__(self, max_queue: int = 256) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._max_queue = max_queue
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish_threadsafe(self, event: dict) -> None:
        """학습 스레드에서 호출. 각 구독자 큐에 안전하게 enqueue."""
        if not self._loop:
            logger.warning('Broadcaster has no loop attached; dropping event')
            return
        loop = self._loop
        for q in list(self._subscribers):
            loop.call_soon_threadsafe(self._enqueue, q, event)

    @staticmethod
    def _enqueue(q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning('SSE queue full; dropping event for one subscriber')

    async def stream(self, q: asyncio.Queue) -> AsyncIterator[dict]:
        try:
            while True:
                event = await q.get()
                if event.get('_terminate'):
                    return
                yield event
        finally:
            self.unsubscribe(q)
