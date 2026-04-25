import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from artifact import (
    ARTIFACT_META_FILE,
    ARTIFACT_MODEL_FILE,
    ARTIFACT_SCALER_FILE,
    ModelMeta,
    compute_arch_hash,
    load_artifact,
    save_artifact,
)
from model import BearingClassifier


def _make_meta(artifact_id: str = 'abc') -> ModelMeta:
    return ModelMeta(
        artifact_id=artifact_id,
        arch_hash=compute_arch_hash(19, [128, 64, 32], 4, 0.3),
        input_size=19,
        hidden_sizes=[128, 64, 32],
        num_classes=4,
        dropout=0.3,
        sampling_rate=12000,
        window_size=1024,
        overlap=0.5,
        class_names=['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault'],
        label_map={'Normal': 0, 'Ball': 1, 'IR': 2, 'OR': 3},
        val_acc=98.0,
        val_loss=0.05,
    )


def test_compute_arch_hash_deterministic() -> None:
    h1 = compute_arch_hash(19, [128, 64, 32], 4, 0.3)
    h2 = compute_arch_hash(19, [128, 64, 32], 4, 0.3)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 40  # sha1 hex


def test_compute_arch_hash_changes_on_diff() -> None:
    base = compute_arch_hash(19, [128, 64, 32], 4, 0.3)
    assert compute_arch_hash(19, [128, 64, 32], 4, 0.4) != base
    assert compute_arch_hash(19, [128, 32], 4, 0.3) != base
    assert compute_arch_hash(20, [128, 64, 32], 4, 0.3) != base


def test_save_and_load_artifact_roundtrip(tmp_path: Path) -> None:
    model = BearingClassifier(input_size=19, hidden_sizes=[128, 64, 32], num_classes=4, dropout=0.3)
    scaler = StandardScaler().fit(np.random.RandomState(0).randn(50, 19))
    meta = _make_meta('roundtrip')

    artifact_dir = tmp_path / 'roundtrip'
    save_artifact(str(artifact_dir), model.state_dict(), scaler, meta)

    # 파일 존재 확인
    assert (artifact_dir / ARTIFACT_MODEL_FILE).exists()
    assert (artifact_dir / ARTIFACT_SCALER_FILE).exists()
    assert (artifact_dir / ARTIFACT_META_FILE).exists()

    loaded = load_artifact(str(artifact_dir), device='cpu')

    # 메타 일치
    assert loaded.meta.artifact_id == 'roundtrip'
    assert loaded.meta.arch_hash == meta.arch_hash
    assert loaded.meta.input_size == 19
    assert loaded.meta.class_names == meta.class_names

    # 모델 출력 동일 (둘 다 eval 모드 — BatchNorm은 running stats 사용)
    x = torch.randn(2, 19)
    model.eval()
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded.model(x)
    torch.testing.assert_close(out_orig, out_loaded)

    # scaler 통계 동일
    np.testing.assert_array_equal(loaded.scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(loaded.scaler.scale_, scaler.scale_)


def test_meta_json_is_human_readable(tmp_path: Path) -> None:
    meta = _make_meta()
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(meta.to_json(), encoding='utf-8')
    payload = json.loads(meta_path.read_text(encoding='utf-8'))
    assert payload['artifact_id'] == 'abc'
    assert payload['hidden_sizes'] == [128, 64, 32]


def test_modelmeta_from_json_roundtrip() -> None:
    original = _make_meta()
    restored = ModelMeta.from_json(original.to_json())
    assert restored == original
