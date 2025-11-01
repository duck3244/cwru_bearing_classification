# ============================================================================
# 파일: feature_extraction.py
# 설명: 특징 추출 함수들
# ============================================================================

import numpy as np
import pandas as pd


def extract_features(data, sampling_rate=12000, window_size=1024):
    """시계열 데이터에서 특징 추출"""
    features = []

    # 시간 도메인 특징
    features.extend(extract_time_domain_features(data))

    # 주파수 도메인 특징
    features.extend(extract_frequency_domain_features(data, sampling_rate))

    return np.array(features)


def extract_time_domain_features(data):
    """시간 도메인 특징 추출"""
    features = []

    # 기본 통계량
    features.append(np.mean(data))  # 평균
    features.append(np.std(data))  # 표준편차
    features.append(np.max(data))  # 최대값
    features.append(np.min(data))  # 최소값
    features.append(np.ptp(data))  # Peak-to-peak

    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(data ** 2))
    features.append(rms)

    # 절대평균
    abs_mean = np.mean(np.abs(data))
    features.append(abs_mean)

    # 피크값
    features.append(np.max(np.abs(data)))

    # 파형 인자 (Shape Factor)
    if abs_mean != 0:
        features.append(rms / abs_mean)
    else:
        features.append(0)

    # 임펄스 인자 (Impulse Factor)
    if abs_mean != 0:
        features.append(np.max(np.abs(data)) / abs_mean)
    else:
        features.append(0)

    # 첨도 (Kurtosis)
    features.append(pd.Series(data).kurtosis())

    # 왜도 (Skewness)
    features.append(pd.Series(data).skew())

    # 신호 에너지
    features.append(np.sum(data ** 2))

    return features


def extract_frequency_domain_features(data, sampling_rate):
    """주파수 도메인 특징 추출"""
    features = []

    # FFT 계산
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1 / sampling_rate)
    fft_power = np.abs(fft_vals) ** 2

    # 양의 주파수만 사용
    positive_freq_idx = fft_freq > 0
    fft_freq = fft_freq[positive_freq_idx]
    fft_power = fft_power[positive_freq_idx]

    # 주파수 도메인 통계량
    features.append(np.mean(fft_power))  # 평균 파워
    features.append(np.std(fft_power))  # 표준편차
    features.append(np.max(fft_power))  # 최대 파워

    # 지배 주파수
    dominant_freq = fft_freq[np.argmax(fft_power)]
    features.append(dominant_freq)

    # 주파수 중심
    features.append(np.sum(fft_freq * fft_power) / np.sum(fft_power))

    # 스펙트럼 엔트로피
    power_norm = fft_power / np.sum(fft_power)
    power_norm = power_norm[power_norm > 0]
    features.append(-np.sum(power_norm * np.log2(power_norm)))

    return features