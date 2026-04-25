# ============================================================================
# 파일: inference.py
# 설명: 단일 신호 → 윈도우 추론 → 집계
# ============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from artifact import LoadedArtifact
from feature_extraction import extract_features

logger = logging.getLogger(__name__)


@dataclass
class WindowPrediction:
    idx: int
    start: int
    label: int
    label_name: str
    probabilities: list[float]


@dataclass
class InferenceResult:
    n_windows: int
    aggregated_label: int
    aggregated_label_name: str
    aggregated_probabilities: dict[str, float]
    windows: list[WindowPrediction]
    processing_ms: int


def _sliding_windows(signal_1d: np.ndarray, window_size: int, overlap: float) -> Iterator[tuple[int, np.ndarray]]:
    stride = max(1, int(window_size * (1.0 - overlap)))
    n = len(signal_1d)
    for start in range(0, n - window_size + 1, stride):
        yield start, signal_1d[start:start + window_size]


def predict_signal(
    signal_1d: np.ndarray,
    artifact: LoadedArtifact,
    device: str | torch.device = 'cpu',
    batch_size: int = 256,
) -> InferenceResult:
    """1D 신호 전체에 대해 슬라이딩 윈도우 추론을 수행하고 집계 결과를 반환"""
    if signal_1d.ndim != 1:
        signal_1d = np.asarray(signal_1d, dtype=np.float64).flatten()

    t0 = time.time()
    meta = artifact.meta

    starts: list[int] = []
    feats: list[np.ndarray] = []
    for start, window in _sliding_windows(signal_1d, meta.window_size, meta.overlap):
        starts.append(start)
        feats.append(extract_features(window, meta.sampling_rate, meta.window_size))

    if not feats:
        raise ValueError(
            f'Signal too short for window_size={meta.window_size}: len={len(signal_1d)}'
        )

    feature_matrix = np.stack(feats)
    feature_matrix = artifact.scaler.transform(feature_matrix)

    model = artifact.model.to(device).eval()
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        x = torch.from_numpy(feature_matrix).float().to(device)
        for i in range(0, x.shape[0], batch_size):
            logits = model(x[i:i + batch_size])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    all_probs = np.concatenate(probs_list, axis=0)  # (n_windows, num_classes)
    labels = all_probs.argmax(axis=1)

    aggregated_probs = all_probs.mean(axis=0)
    aggregated_label = int(aggregated_probs.argmax())

    windows = [
        WindowPrediction(
            idx=i,
            start=int(starts[i]),
            label=int(labels[i]),
            label_name=meta.class_names[int(labels[i])],
            probabilities=[float(p) for p in all_probs[i]],
        )
        for i in range(len(starts))
    ]

    return InferenceResult(
        n_windows=len(starts),
        aggregated_label=aggregated_label,
        aggregated_label_name=meta.class_names[aggregated_label],
        aggregated_probabilities={
            name: float(aggregated_probs[i]) for i, name in enumerate(meta.class_names)
        },
        windows=windows,
        processing_ms=int((time.time() - t0) * 1000),
    )
