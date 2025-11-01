# ============================================================================
# 파일: data_loader.py
# 설명: 데이터 로딩 및 전처리
# ============================================================================

import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class CWRUDataLoader:
    """CWRU 베어링 데이터 로더"""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self, data_path=None):
        """데이터 로드"""
        if data_path is None:
            data_path = self.config.DATA_PATH

        all_features = []
        all_labels = []

        # 실제 구현: .mat 파일에서 데이터 로드
        # 여기서는 데모용 샘플 데이터 생성
        np.random.seed(self.config.RANDOM_STATE)

        for label_name, label_idx in self.config.LABEL_MAP.items():
            for i in range(100):
                data = self._generate_sample_data(label_name)
                features = self._extract_features(data)
                all_features.append(features)
                all_labels.append(label_idx)

        return np.array(all_features), np.array(all_labels)

    def _generate_sample_data(self, label_name):
        """데모용 샘플 데이터 생성"""
        window_size = self.config.WINDOW_SIZE

        if label_name == 'Normal':
            data = np.random.normal(0, 0.5, window_size)
        elif label_name == 'Ball':
            t = np.linspace(0, 1, window_size)
            data = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t) + np.random.normal(0, 0.3,
                                                                                                     window_size)
        elif label_name == 'IR':
            t = np.linspace(0, 1, window_size)
            data = 2 * np.sin(2 * np.pi * 75 * t) + np.random.normal(0, 0.5, window_size)
        else:  # OR
            t = np.linspace(0, 1, window_size)
            data = np.sin(2 * np.pi * 30 * t) + 3 * signal.square(2 * np.pi * 10 * t) + np.random.normal(0, 0.4,
                                                                                                         window_size)

        return data

    def _extract_features(self, data):
        """특징 추출은 feature_extraction.py의 함수 호출"""
        from feature_extraction import extract_features
        return extract_features(data, self.config.SAMPLING_RATE, self.config.WINDOW_SIZE)

    def split_data(self, X, y):
        """데이터 분할"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def normalize_data(self, X_train, X_test):
        """데이터 정규화"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled