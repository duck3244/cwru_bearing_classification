import numpy as np
import pytest

from config import Config
from data_loader import CWRUDataLoader


@pytest.fixture
def loader() -> CWRUDataLoader:
    return CWRUDataLoader(Config())


@pytest.mark.parametrize('filename, expected_key', [
    ('Time_Normal_1_098.mat', 'Normal'),
    ('Normal_baseline.mat', 'Normal'),
    ('B007_1_123.mat', 'Ball'),
    ('IR014_1_175.mat', 'IR'),
    ('OR007_6_1_136.mat', 'OR'),
])
def test_label_from_filename(loader: CWRUDataLoader, filename: str, expected_key: str) -> None:
    expected = loader.config.LABEL_MAP[expected_key]
    assert loader._label_from_filename(filename) == expected


def test_label_from_filename_unknown(loader: CWRUDataLoader) -> None:
    assert loader._label_from_filename('Z_unknown.mat') is None


def test_sliding_windows_lengths(loader: CWRUDataLoader) -> None:
    signal = np.arange(1024 * 5, dtype=np.float64)
    windows = list(loader._sliding_windows(signal))
    # overlap=0.5, stride=512 → (5*1024 - 1024)/512 + 1 = 9
    assert len(windows) == 9
    assert all(w.shape == (1024,) for w in windows)
    # 각 윈도우 시작점이 stride 간격
    assert windows[0][0] == 0
    assert windows[1][0] == 512


def test_split_data_shapes_and_disjoint(loader: CWRUDataLoader) -> None:
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.standard_normal((n, 19))
    y = rng.integers(0, 4, size=n)
    X_tr, X_val, X_te, y_tr, y_val, y_te = loader.split_data(X, y)
    assert X_tr.shape[0] + X_val.shape[0] + X_te.shape[0] == n
    # test 비율 ≈ 0.2
    assert abs(X_te.shape[0] / n - Config.TEST_SIZE) < 0.01
    # val 비율 ≈ 0.1
    assert abs(X_val.shape[0] / n - Config.VAL_SIZE) < 0.02


def test_split_data_reproducible(loader: CWRUDataLoader) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 19))
    y = rng.integers(0, 4, size=500)
    a = loader.split_data(X, y)
    b = loader.split_data(X, y)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_normalize_data_uses_train_stats(loader: CWRUDataLoader) -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((100, 5)) * 5 + 10
    X_val = rng.standard_normal((30, 5)) * 5 + 10
    X_te = rng.standard_normal((30, 5)) * 5 + 10
    X_tr_s, _, _ = loader.normalize_data(X_tr, X_val, X_te)
    # train은 평균 0, 표준편차 1로 정규화돼야 함
    np.testing.assert_allclose(X_tr_s.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_tr_s.std(axis=0), 1, atol=1e-10)
