# ============================================================================
# 파일: artifact.py
# 설명: 모델 아티팩트 저장/로드 (model.pt + scaler.joblib + meta.json)
# ============================================================================

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from model import BearingClassifier

ARTIFACT_MODEL_FILE = 'model.pt'
ARTIFACT_SCALER_FILE = 'scaler.joblib'
ARTIFACT_META_FILE = 'meta.json'


@dataclass
class ModelMeta:
    """추론에 필요한 모든 메타데이터를 담는 컨테이너"""

    artifact_id: str
    arch_hash: str
    input_size: int
    hidden_sizes: list[int]
    num_classes: int
    dropout: float
    sampling_rate: int
    window_size: int
    overlap: float
    class_names: list[str]
    label_map: dict[str, int]
    val_acc: Optional[float] = None
    val_loss: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> 'ModelMeta':
        return cls(**json.loads(payload))


def compute_arch_hash(input_size: int, hidden_sizes: list[int], num_classes: int, dropout: float) -> str:
    """모델 구조 식별자 (GPU 슬롯 동일성 비교용)"""
    payload = json.dumps({
        'input_size': input_size,
        'hidden_sizes': list(hidden_sizes),
        'num_classes': num_classes,
        'dropout': dropout,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def save_artifact(
    artifact_dir: str,
    model_state_dict: dict,
    scaler: StandardScaler,
    meta: ModelMeta,
) -> str:
    """아티팩트 디렉토리에 model/scaler/meta 저장"""
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(model_state_dict, os.path.join(artifact_dir, ARTIFACT_MODEL_FILE))
    joblib.dump(scaler, os.path.join(artifact_dir, ARTIFACT_SCALER_FILE))
    with open(os.path.join(artifact_dir, ARTIFACT_META_FILE), 'w', encoding='utf-8') as f:
        f.write(meta.to_json())
    return artifact_dir


@dataclass
class LoadedArtifact:
    """load_artifact 결과 컨테이너"""
    model: BearingClassifier
    scaler: StandardScaler
    meta: ModelMeta


def load_artifact(artifact_dir: str, device: str | torch.device = 'cpu') -> LoadedArtifact:
    """아티팩트 디렉토리를 모델 객체로 복원"""
    meta_path = os.path.join(artifact_dir, ARTIFACT_META_FILE)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = ModelMeta.from_json(f.read())

    model = BearingClassifier(
        input_size=meta.input_size,
        hidden_sizes=meta.hidden_sizes,
        num_classes=meta.num_classes,
        dropout=meta.dropout,
    )
    state = torch.load(
        os.path.join(artifact_dir, ARTIFACT_MODEL_FILE),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()

    scaler: StandardScaler = joblib.load(os.path.join(artifact_dir, ARTIFACT_SCALER_FILE))

    return LoadedArtifact(model=model, scaler=scaler, meta=meta)
