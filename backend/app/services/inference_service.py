# ============================================================================
# 파일: app/services/inference_service.py
# 설명: 추론 요청 처리 — GPU 슬롯에 현재 모델 보장 후 inference 호출
# ============================================================================

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

import numpy as np
from scipy.io import loadmat
from sqlmodel import Session

from app.db import InferenceLog, ModelArtifact, engine
from app.services.gpu_slot import GpuSlotManager
from artifact import LoadedArtifact
from data_loader import _DE_TIME_RE
from inference import InferenceResult, predict_signal

logger = logging.getLogger(__name__)


class NoCurrentModelError(RuntimeError):
    pass


class TrainingInProgressError(RuntimeError):
    pass


class InferenceService:
    def __init__(self, slot: GpuSlotManager, training_is_running) -> None:
        self._slot = slot
        self._training_is_running = training_is_running
        self._lock = asyncio.Lock()  # 단일 사용자라도 순차 실행 강제

    async def predict_mat_bytes(self, filename: str, content: bytes) -> dict:
        if self._training_is_running():
            raise TrainingInProgressError()

        async with self._lock:
            artifact = await self._ensure_current()
            signal = _extract_de_signal(content)
            result: InferenceResult = await asyncio.to_thread(
                predict_signal, signal, artifact, self._slot.device,
            )
            response = _result_to_dict(result, artifact, filename)
            await asyncio.to_thread(_log_inference, filename, result, artifact)
            return response

    async def _ensure_current(self) -> LoadedArtifact:
        with Session(engine) as s:
            from sqlmodel import select
            current = s.exec(
                select(ModelArtifact).where(ModelArtifact.is_current == True)  # noqa: E712
            ).first()
        if current is None:
            raise NoCurrentModelError('No current model registered')
        state = await self._slot.acquire(current.id, current.arch_hash, current.artifact_dir)
        return state.artifact


def _extract_de_signal(content: bytes) -> np.ndarray:
    mat = loadmat(BytesIO(content))
    de_keys = [k for k in mat.keys() if _DE_TIME_RE.match(k)]
    if not de_keys:
        raise ValueError(f'No DE_time key in mat file; keys={list(mat.keys())}')
    return np.asarray(mat[de_keys[0]], dtype=np.float64).flatten()


def _result_to_dict(result: InferenceResult, artifact: LoadedArtifact, filename: str) -> dict:
    return {
        'filename': filename,
        'model_artifact_id': artifact.meta.artifact_id,
        'model_val_acc': artifact.meta.val_acc,
        'n_windows': result.n_windows,
        'aggregated': {
            'label': result.aggregated_label,
            'label_name': result.aggregated_label_name,
            'probabilities': result.aggregated_probabilities,
        },
        'windows': [
            {'idx': w.idx, 'start': w.start, 'label': w.label,
             'label_name': w.label_name, 'probabilities': w.probabilities}
            for w in result.windows
        ],
        'processing_ms': result.processing_ms,
    }


def _log_inference(filename: str, result: InferenceResult, artifact: LoadedArtifact) -> None:
    log = InferenceLog(
        id=uuid4().hex,
        timestamp=datetime.now(timezone.utc),
        filename=filename,
        n_windows=result.n_windows,
        aggregated_label=result.aggregated_label,
        aggregated_label_name=result.aggregated_label_name,
        aggregated_probs=json.dumps(result.aggregated_probabilities, ensure_ascii=False),
        processing_ms=result.processing_ms,
        artifact_id=artifact.meta.artifact_id,
    )
    with Session(engine) as s:
        s.add(log); s.commit()
