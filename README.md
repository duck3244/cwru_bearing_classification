# CWRU Bearing Fault Classification with PyTorch

베어링 고장 진단을 위한 딥러닝 기반 분류 시스템입니다. Case Western Reserve University (CWRU) 베어링 데이터셋을 사용하여 정상 상태와 다양한 고장 유형을 분류합니다.

## 🎯 프로젝트 개요

이 프로젝트는 베어링의 진동 신호를 분석하여 다음 4가지 상태를 분류합니다:

- **Normal**: 정상 상태
- **Ball Fault**: 볼 결함
- **Inner Race Fault**: 내륜 결함
- **Outer Race Fault**: 외륜 결함

## ✨ 주요 기능

### 1. 특징 추출
- **시간 도메인 특징**
  - 평균, 표준편차, RMS, 피크값
  - 첨도(Kurtosis), 왜도(Skewness)
  - 파형 인자, 임펄스 인자
  - 신호 에너지

- **주파수 도메인 특징**
  - FFT 기반 파워 스펙트럼
  - 지배 주파수, 주파수 중심
  - 스펙트럼 엔트로피

### 2. 딥러닝 모델
- 다층 퍼셉트론(MLP) 아키텍처
- Batch Normalization으로 안정적인 학습
- Dropout으로 과적합 방지
- Learning Rate Scheduler로 최적화

### 3. 시각화
- 학습 곡선 (Loss & Accuracy)
- 혼동 행렬 (Confusion Matrix)
- 분류 성능 리포트

## 📁 프로젝트 구조

```
cwru_bearing_classification/
│
├── config.py                  # 프로젝트 설정 및 하이퍼파라미터
├── data_loader.py             # 데이터 로딩 및 전처리
├── feature_extraction.py      # 특징 추출 함수
├── dataset.py                 # PyTorch Dataset 클래스
├── model.py                   # 신경망 모델 정의
├── trainer.py                 # 학습 및 평가 로직
├── visualizer.py              # 결과 시각화
├── main.py                    # 메인 실행 파일
│
├── data/                      # 데이터 디렉토리
│   └── (CWRU .mat 파일들)
│
├── models/                    # 저장된 모델
│   └── best_model.pth
│
├── results/                   # 결과 이미지
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── requirements.txt           # 패키지 의존성
└── README.md                  # 이 문서
```

## 🔧 설치 방법

### 1. 가상 환경 생성 (선택사항)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt 내용:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🚀 사용 방법

### 기본 실행

```bash
python main.py
```

### 단계별 실행

1. **데이터 로드 및 전처리**
```python
from config import Config
from data_loader import CWRUDataLoader

config = Config()
data_loader = CWRUDataLoader(config)
X, y = data_loader.load_data()
```

2. **모델 생성**
```python
from model import BearingClassifier

model = BearingClassifier(
    input_size=X.shape[1],
    hidden_sizes=[128, 64, 32],
    num_classes=4,
    dropout=0.3
)
```

3. **학습**
```python
from trainer import Trainer

trainer = Trainer(model, config)
trainer.train(train_loader, test_loader, epochs=50)
```

4. **평가 및 시각화**
```python
from visualizer import Visualizer

visualizer = Visualizer(config)
visualizer.plot_training_history(trainer.history)
visualizer.plot_confusion_matrix(y_true, y_pred)
```

## 📊 데이터셋

### CWRU 베어링 데이터셋

- **출처**: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- **샘플링 레이트**: 12,000 Hz
- **윈도우 크기**: 1,024 샘플
- **클래스**: 4개 (Normal, Ball, IR, OR)

### 데이터 구조

```
data/
├── Normal_0.mat
├── Ball_007.mat
├── IR_007.mat
└── OR_007.mat
```

### 실제 데이터 사용하기

`data_loader.py`의 `_generate_sample_data()` 함수를 실제 .mat 파일 로딩 코드로 교체하세요:

```python
def load_mat_file(self, filepath):
    """실제 CWRU .mat 파일 로드"""
    mat_data = loadmat(filepath)
    # 데이터 키는 파일마다 다를 수 있음
    # 예: 'DE_time', 'FE_time' 등
    data = mat_data['DE_time'].flatten()
    return data
```

## 🏗️ 모델 아키텍처

### 기본 구조

```
Input Layer (19 features)
    ↓
Dense Layer (128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense Layer (64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense Layer (32) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Output Layer (4 classes)
```

### 특징 벡터 구성 (19차원)

**시간 도메인 (13개)**
- 평균, 표준편차, 최대값, 최소값, Peak-to-peak
- RMS, 절대평균, 피크값
- 파형 인자, 임펄스 인자
- 첨도, 왜도, 신호 에너지

**주파수 도메인 (6개)**
- 평균 파워, 파워 표준편차, 최대 파워
- 지배 주파수, 주파수 중심, 스펙트럼 엔트로피

## 📈 결과

### 예상 성능

- **학습 정확도**: ~95-98%
- **검증 정확도**: ~92-96%
- **테스트 정확도**: ~90-95%

### 결과 파일

학습 완료 후 다음 파일들이 생성됩니다:

- `models/best_model.pth`: 최고 성능 모델
- `results/training_history.png`: 학습 곡선
- `results/confusion_matrix.png`: 혼동 행렬

## ⚙️ 설정 변경

`config.py` 파일에서 하이퍼파라미터를 조정할 수 있습니다:

```python
class Config:
    # 모델 구조
    HIDDEN_SIZES = [128, 64, 32]  # 히든 레이어 크기
    DROPOUT = 0.3                  # 드롭아웃 비율
    
    # 학습 파라미터
    BATCH_SIZE = 32                # 배치 크기
    LEARNING_RATE = 0.001          # 학습률
    EPOCHS = 50                    # 에포크 수
    WEIGHT_DECAY = 1e-5            # 가중치 감쇠
    
    # 데이터 파라미터
    WINDOW_SIZE = 1024             # 윈도우 크기
    SAMPLING_RATE = 12000          # 샘플링 레이트
```

## 🔬 성능 향상 팁

### 1. 데이터 증강
- 윈도우 슬라이딩으로 더 많은 샘플 생성
- 노이즈 추가로 robust한 모델 학습

### 2. 모델 튜닝
- 히든 레이어 수와 크기 조정
- Dropout 비율 변경
- Learning Rate 조정

### 3. 특징 엔지니어링
- 추가 시간/주파수 도메인 특징
- Wavelet 변환 특징
- Envelope 분석

### 4. 앙상블 방법
- 여러 모델의 예측 결합
- K-Fold 교차 검증

## 🐛 트러블슈팅

### CUDA 메모리 부족
```python
# config.py에서 배치 크기 감소
BATCH_SIZE = 16  # 32에서 16으로
```

### 과적합 발생
```python
# Dropout 비율 증가
DROPOUT = 0.5  # 0.3에서 0.5로

# Weight Decay 증가
WEIGHT_DECAY = 1e-4  # 1e-5에서 1e-4로
```

### 학습이 느린 경우
```python
# 학습률 증가
LEARNING_RATE = 0.01  # 0.001에서 0.01로

# GPU 사용 확인
print(torch.cuda.is_available())
```

## 📚 참고 자료

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
