# ============================================================================
# 파일: utils.py
# 설명: 공통 유틸리티 (재현성, 로깅 등)
# ============================================================================

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """전체 라이브러리에 대한 재현성 시드 설정"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: str = 'INFO') -> None:
    """루트 로거 설정 (멱등)"""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                           datefmt='%H:%M:%S'))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
