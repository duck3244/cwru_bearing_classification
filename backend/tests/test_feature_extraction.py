import numpy as np
import pytest

from feature_extraction import (
    extract_features,
    extract_frequency_domain_features,
    extract_time_domain_features,
)


@pytest.fixture
def signal_1024() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(1024)


def test_extract_features_shape(signal_1024: np.ndarray) -> None:
    feats = extract_features(signal_1024)
    # 시간(13) + 주파수(6) = 19
    assert feats.shape == (19,)
    assert feats.dtype == np.float64


def test_time_domain_count(signal_1024: np.ndarray) -> None:
    assert len(extract_time_domain_features(signal_1024)) == 13


def test_frequency_domain_count(signal_1024: np.ndarray) -> None:
    assert len(extract_frequency_domain_features(signal_1024, sampling_rate=12000)) == 6


def test_features_are_finite(signal_1024: np.ndarray) -> None:
    feats = extract_features(signal_1024)
    assert np.all(np.isfinite(feats))


def test_features_deterministic(signal_1024: np.ndarray) -> None:
    a = extract_features(signal_1024)
    b = extract_features(signal_1024)
    np.testing.assert_array_equal(a, b)


def test_zero_signal_does_not_raise() -> None:
    """abs_mean=0인 경우에도 division-by-zero 없이 동작"""
    zero = np.zeros(1024)
    feats = extract_features(zero)
    assert np.all(np.isfinite(feats))
