from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from artifact import ModelMeta, compute_arch_hash, load_artifact, save_artifact
from inference import predict_signal
from model import BearingClassifier


@pytest.fixture
def trained_artifact(tmp_path: Path):
    """소규모 더미 아티팩트. 학습은 안 했지만 inference 파이프라인 검증엔 충분."""
    model = BearingClassifier(input_size=19, hidden_sizes=[16, 8], num_classes=4, dropout=0.0)
    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.standard_normal((50, 19)))
    meta = ModelMeta(
        artifact_id='test',
        arch_hash=compute_arch_hash(19, [16, 8], 4, 0.0),
        input_size=19,
        hidden_sizes=[16, 8],
        num_classes=4,
        dropout=0.0,
        sampling_rate=12000,
        window_size=1024,
        overlap=0.5,
        class_names=['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault'],
        label_map={'Normal': 0, 'Ball': 1, 'IR': 2, 'OR': 3},
        val_acc=0.0,
        val_loss=0.0,
    )
    save_artifact(str(tmp_path), model.state_dict(), scaler, meta)
    return load_artifact(str(tmp_path), device='cpu')


def test_predict_signal_basic(trained_artifact) -> None:
    signal = np.random.RandomState(42).randn(1024 * 5).astype(np.float64)
    result = predict_signal(signal, trained_artifact, device='cpu')
    # overlap=0.5, stride=512 → (5*1024-1024)/512 + 1 = 9
    assert result.n_windows == 9
    assert len(result.windows) == 9
    assert 0 <= result.aggregated_label < 4
    assert result.aggregated_label_name in ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault']
    # 확률 합 ≈ 1
    total = sum(result.aggregated_probabilities.values())
    assert abs(total - 1.0) < 1e-5
    # 윈도우별 확률도 분포로 정상
    for w in result.windows:
        assert len(w.probabilities) == 4
        assert abs(sum(w.probabilities) - 1.0) < 1e-5


def test_predict_signal_too_short_raises(trained_artifact) -> None:
    signal = np.random.randn(500)  # window_size(1024)보다 짧음
    with pytest.raises(ValueError, match='Signal too short'):
        predict_signal(signal, trained_artifact, device='cpu')


def test_predict_signal_2d_input_flattened(trained_artifact) -> None:
    """입력이 (N, 1) shape이어도 flatten되어 동작"""
    signal = np.random.randn(1024 * 3, 1)
    result = predict_signal(signal, trained_artifact, device='cpu')
    assert result.n_windows >= 1
