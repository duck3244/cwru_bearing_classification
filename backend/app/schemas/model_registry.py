from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ArtifactOut(BaseModel):
    id: str
    run_id: str
    val_acc: float
    val_loss: float
    is_current: bool
    arch_hash: str
    created_at: datetime


class CurrentModelOut(BaseModel):
    artifact_id: Optional[str]
    val_acc: Optional[float]
    val_loss: Optional[float]
    arch_hash: Optional[str]
    class_names: Optional[list[str]]
    input_size: Optional[int]
    hidden_sizes: Optional[list[int]]
    created_at: Optional[datetime]
    device: str
