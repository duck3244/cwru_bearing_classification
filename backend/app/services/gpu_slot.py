# ============================================================================
# 파일: app/services/gpu_slot.py
# 설명: 단일 GPU 슬롯 모델 매니저 — 동일 모델이면 상주, 다르면 swap
# ============================================================================

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from artifact import LoadedArtifact, load_artifact

logger = logging.getLogger(__name__)


@dataclass
class SlotState:
    artifact_id: str
    arch_hash: str
    artifact: LoadedArtifact


class GpuSlotManager:
    """
    한 시점에 GPU(또는 CPU)에 모델 0~1개만 상주.
    - 동일 (artifact_id, arch_hash)면 swap 없이 그대로 사용
    - 다르면 기존 release 후 새로 로드
    - 학습 중에는 release()로 비워두고, 학습 종료 후 새 모델로 acquire()
    """

    def __init__(self, device: Optional[str | torch.device] = None) -> None:
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._state: Optional[SlotState] = None
        self._lock = asyncio.Lock()

    @property
    def device_str(self) -> str:
        return str(self.device)

    @property
    def current(self) -> Optional[SlotState]:
        return self._state

    def is_loaded(self, artifact_id: str, arch_hash: str) -> bool:
        return (
            self._state is not None
            and self._state.artifact_id == artifact_id
            and self._state.arch_hash == arch_hash
        )

    async def acquire(self, artifact_id: str, arch_hash: str, artifact_dir: str) -> SlotState:
        """슬롯에 모델 로드 (이미 동일 모델이면 no-op). 반환값은 현재 슬롯 상태"""
        async with self._lock:
            if self.is_loaded(artifact_id, arch_hash):
                logger.debug('GPU slot already holds %s — reusing', artifact_id)
                return self._state  # type: ignore[return-value]
            self._unload_locked()
            if not os.path.isdir(artifact_dir):
                raise FileNotFoundError(f'Artifact directory not found: {artifact_dir}')
            logger.info('Loading artifact %s onto %s', artifact_id, self.device)
            loaded = load_artifact(artifact_dir, device=self.device)
            self._state = SlotState(artifact_id=artifact_id, arch_hash=arch_hash, artifact=loaded)
            return self._state

    async def release(self) -> None:
        """슬롯 비우기 (학습 시작 시 호출)"""
        async with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._state is None:
            return
        logger.info('Releasing slot (was %s)', self._state.artifact_id)
        del self._state.artifact.model
        self._state = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
