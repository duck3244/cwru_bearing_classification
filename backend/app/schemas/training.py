from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TrainParams(BaseModel):
    EPOCHS: int = Field(50, ge=1, le=500)
    BATCH_SIZE: int = Field(32, ge=1, le=1024)
    LEARNING_RATE: float = Field(1e-3, gt=0, le=1.0)
    WEIGHT_DECAY: float = Field(1e-5, ge=0, le=1.0)
    DROPOUT: float = Field(0.3, ge=0.0, le=0.9)
    HIDDEN_SIZES: list[int] = Field(default_factory=lambda: [128, 64, 32])
    EARLY_STOPPING_PATIENCE: int = Field(10, ge=1, le=100)


class TrainStartResponse(BaseModel):
    run_id: str


class CancelResponse(BaseModel):
    cancelled: bool


class EpochOut(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    elapsed_ms: int


class RunOut(BaseModel):
    id: str
    started_at: datetime
    finished_at: Optional[datetime]
    status: str
    best_val_acc: Optional[float]
    best_val_loss: Optional[float]
    best_epoch: Optional[int]
    n_epochs_run: int
    artifact_id: Optional[str]
    error_message: Optional[str]


class RunDetail(RunOut):
    epochs: list[EpochOut]
    config_snapshot: dict
