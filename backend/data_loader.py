# ============================================================================
# 파일: data_loader.py
# 설명: 데이터 로딩 및 전처리
# ============================================================================

import logging
import os
import re
from typing import Iterator, Optional

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DE_TIME_RE = re.compile(r'^X\d+_DE_time$')


class CWRUDataLoader:
    """CWRU 베어링 데이터 로더"""

    def __init__(self, config) -> None:
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self, data_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """data 디렉토리의 .mat 파일을 모두 읽어 윈도우 단위 특징 행렬을 반환"""
        if data_path is None:
            data_path = self.config.DATA_PATH

        all_features: list[np.ndarray] = []
        all_labels: list[int] = []

        mat_files = sorted(f for f in os.listdir(data_path) if f.endswith('.mat'))
        if not mat_files:
            raise FileNotFoundError(f'No .mat files found under {data_path}')

        for filename in mat_files:
            label_idx = self._label_from_filename(filename)
            if label_idx is None:
                logger.warning('Skipping %s: unknown label prefix', filename)
                continue

            signal_1d = self._load_de_signal(os.path.join(data_path, filename))
            for window in self._sliding_windows(signal_1d):
                features = self._extract_features(window)
                all_features.append(features)
                all_labels.append(label_idx)

        return np.asarray(all_features), np.asarray(all_labels)

    def _label_from_filename(self, filename: str) -> Optional[int]:
        """파일명 접두어로부터 클래스 인덱스 추정"""
        name = filename.lower()
        if name.startswith('time_normal') or name.startswith('normal'):
            return self.config.LABEL_MAP['Normal']
        if name.startswith('ir'):
            return self.config.LABEL_MAP['IR']
        if name.startswith('or'):
            return self.config.LABEL_MAP['OR']
        if name.startswith('b'):
            return self.config.LABEL_MAP['Ball']
        return None

    def _load_de_signal(self, filepath: str) -> np.ndarray:
        """X{NNN}_DE_time 키를 찾아 1D float 배열로 반환"""
        mat = loadmat(filepath)
        de_keys = [k for k in mat.keys() if _DE_TIME_RE.match(k)]
        if not de_keys:
            raise KeyError(f'No DE_time key in {filepath}; keys={list(mat.keys())}')
        return np.asarray(mat[de_keys[0]], dtype=np.float64).flatten()

    def _sliding_windows(self, signal_1d: np.ndarray) -> Iterator[np.ndarray]:
        """슬라이딩 윈도우로 신호 분할 (overlap 적용)"""
        window = self.config.WINDOW_SIZE
        stride = max(1, int(window * (1.0 - self.config.OVERLAP)))
        n = len(signal_1d)
        for start in range(0, n - window + 1, stride):
            yield signal_1d[start:start + window]

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        from feature_extraction import extract_features
        return extract_features(data, self.config.SAMPLING_RATE, self.config.WINDOW_SIZE)

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """train/val/test 3-way 분할 (stratified)"""
        test_size = self.config.TEST_SIZE
        val_size = self.config.VAL_SIZE
        random_state = self.config.RANDOM_STATE

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_relative,
            random_state=random_state,
            stratify=y_trainval,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize_data(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """train 통계로 fit, val/test에 transform"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled
