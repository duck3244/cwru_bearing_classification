# ============================================================================
# 파일: config.py
# 설명: 프로젝트 설정 및 하이퍼파라미터
# ============================================================================

import torch


class Config:
    """프로젝트 전역 설정"""

    # 데이터 관련
    DATA_PATH = 'data/'
    SAMPLING_RATE = 12000
    WINDOW_SIZE = 1024
    OVERLAP = 0.5

    # 모델 관련
    HIDDEN_SIZES = [128, 64, 32]
    DROPOUT = 0.3
    NUM_CLASSES = 4

    # 학습 관련
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    WEIGHT_DECAY = 1e-5

    # 학습률 스케줄러
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # 데이터 분할
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    # 디바이스
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 저장 경로
    MODEL_SAVE_PATH = 'models/'
    RESULTS_SAVE_PATH = 'results/'

    # 클래스 이름
    CLASS_NAMES = ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault']

    # 레이블 매핑
    LABEL_MAP = {
        'Normal': 0,
        'Ball': 1,
        'IR': 2,
        'OR': 3
    }