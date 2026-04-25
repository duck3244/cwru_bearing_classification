# ============================================================================
# 파일: feature_extraction.py
# 설명: 특징 추출 함수들
# ============================================================================

import numpy as np
from scipy import stats


def extract_features(data: np.ndarray, sampling_rate: int = 12000, window_size: int = 1024) -> np.ndarray:
    """시계열 데이터에서 시간/주파수 도메인 특징을 추출"""
    features: list[float] = []
    features.extend(extract_time_domain_features(data))
    features.extend(extract_frequency_domain_features(data, sampling_rate))
    return np.array(features, dtype=np.float64)


def extract_time_domain_features(data: np.ndarray) -> list[float]:
    """시간 도메인 특징 (13개)"""
    features: list[float] = []

    features.append(float(np.mean(data)))           # 평균
    features.append(float(np.std(data)))            # 표준편차
    features.append(float(np.max(data)))            # 최대값
    features.append(float(np.min(data)))            # 최소값
    features.append(float(np.ptp(data)))            # Peak-to-peak

    rms = float(np.sqrt(np.mean(data ** 2)))
    features.append(rms)

    abs_mean = float(np.mean(np.abs(data)))
    features.append(abs_mean)

    peak = float(np.max(np.abs(data)))
    features.append(peak)

    # Shape factor / Impulse factor — abs_mean이 0이면 0으로 대체
    features.append(rms / abs_mean if abs_mean else 0.0)
    features.append(peak / abs_mean if abs_mean else 0.0)

    # 첨도/왜도 — Fisher 정의, 표본 보정. 분산 0이면 정의 불가 → 0으로 대체
    if np.std(data) > 0:
        features.append(float(stats.kurtosis(data, fisher=True, bias=False)))
        features.append(float(stats.skew(data, bias=False)))
    else:
        features.append(0.0)
        features.append(0.0)

    # 신호 에너지
    features.append(float(np.sum(data ** 2)))

    return features


def extract_frequency_domain_features(data: np.ndarray, sampling_rate: int) -> list[float]:
    """주파수 도메인 특징 (6개)"""
    features: list[float] = []

    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1.0 / sampling_rate)
    fft_power = np.abs(fft_vals) ** 2

    positive_freq_idx = fft_freq > 0
    fft_freq = fft_freq[positive_freq_idx]
    fft_power = fft_power[positive_freq_idx]

    features.append(float(np.mean(fft_power)))
    features.append(float(np.std(fft_power)))
    features.append(float(np.max(fft_power)))

    dominant_freq = float(fft_freq[np.argmax(fft_power)])
    features.append(dominant_freq)

    total_power = float(np.sum(fft_power))
    features.append(float(np.sum(fft_freq * fft_power) / total_power) if total_power else 0.0)

    power_norm = fft_power / total_power if total_power else fft_power
    power_norm = power_norm[power_norm > 0]
    features.append(float(-np.sum(power_norm * np.log2(power_norm))) if power_norm.size else 0.0)

    return features
