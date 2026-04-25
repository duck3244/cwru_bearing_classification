from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class WindowOut(BaseModel):
    idx: int
    start: int
    label: int
    label_name: str
    probabilities: list[float]


class AggregatedOut(BaseModel):
    label: int
    label_name: str
    probabilities: dict[str, float]


class InferenceResponse(BaseModel):
    filename: str
    model_artifact_id: str
    model_val_acc: float | None
    n_windows: int
    aggregated: AggregatedOut
    windows: list[WindowOut]
    processing_ms: int


class InferenceLogOut(BaseModel):
    id: str
    timestamp: datetime
    filename: str
    n_windows: int
    aggregated_label: int
    aggregated_label_name: str
    aggregated_probs: dict[str, float]
    processing_ms: int
    artifact_id: str
