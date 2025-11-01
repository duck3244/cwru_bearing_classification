# ============================================================================
# 파일: model.py
# 설명: PyTorch 모델 정의
# ============================================================================

import torch
import torch.nn as nn


class BearingClassifier(nn.Module):
    """베어링 고장 분류 신경망"""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super(BearingClassifier, self).__init__()

        layers = []
        prev_size = input_size

        # 히든 레이어 구성
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 출력 레이어
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)