# ============================================================================
# 파일: dataset.py
# 설명: PyTorch Dataset 클래스
# ============================================================================

import torch
from torch.utils.data import Dataset


class BearingDataset(Dataset):
    """베어링 데이터셋 클래스"""

    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]