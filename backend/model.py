# ============================================================================
# 파일: model.py
# 설명: PyTorch 모델 정의
# ============================================================================

from typing import Sequence

import torch
import torch.nn as nn


class BearingClassifier(nn.Module):
    """베어링 고장 분류 신경망 (MLP)"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
