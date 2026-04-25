# ============================================================================
# 파일: dataset.py
# 설명: PyTorch Dataset 클래스
# ============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class BearingDataset(Dataset):
    """베어링 데이터셋 클래스"""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
