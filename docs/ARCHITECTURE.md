# CWRU Bearing Classification — 아키텍처 문서

## 1. 개요

CWRU(Case Western Reserve University) 베어링 데이터셋을 사용한 베어링 결함 분류 시스템입니다.
진동 신호(.mat)를 입력받아 슬라이딩 윈도우 기반으로 19차원 특징을 추출하고, MLP 모델로 결함 유형을 분류합니다.

- **백엔드**: FastAPI + PyTorch + SQLite(SQLModel)
- **프론트엔드**: Vue 3 + TypeScript + Pinia + Chart.js
- **통신**: REST API + Server-Sent Events(SSE, 학습 진행률 실시간 스트리밍)

## 2. 시스템 구성도

```
┌─────────────────────────────────────────────────────────────┐
│              Frontend  (Vue 3 + Vite + Pinia)               │
│                                                              │
│   PredictView ─┐                                             │
│   TrainView   ─┼─→ axios(/api/*)  +  EventSource(/api/...) │
│   HistoryView ─┘                                             │
└──────────────────────────────┬──────────────────────────────┘
                               │  HTTP / SSE
┌──────────────────────────────▼──────────────────────────────┐
│              Backend  (FastAPI + Uvicorn)                    │
│                                                              │
│   API Layer       ── health / train / predict / models      │
│   Service Layer   ── TrainingService, InferenceService,     │
│                       GpuSlotManager, Broadcaster           │
│   ML Core         ── BearingClassifier, Trainer,            │
│                       CWRUDataLoader, predict_signal        │
│   Artifact Layer  ── ModelMeta / LoadedArtifact / save·load │
│   Persistence     ── SQLModel ORM                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│   Storage                                                    │
│   • data/                       (CWRU .mat 입력)            │
│   • models/{artifact_id}/       (model.pt + scaler + meta)  │
│   • db/cwru.sqlite              (런/메트릭/아티팩트/로그)    │
└──────────────────────────────────────────────────────────────┘
```

## 3. 디렉터리 구조

```
cwru_bearing_classification/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 앱 팩토리, lifespan
│   │   ├── db.py                      # SQLModel 테이블 정의
│   │   ├── api/                       # 라우터
│   │   │   ├── health.py
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   ├── models.py
│   │   │   └── inferences.py
│   │   ├── services/                  # 비즈니스 로직
│   │   │   ├── training.py            # TrainingService
│   │   │   ├── inference_service.py   # InferenceService
│   │   │   ├── gpu_slot.py            # GpuSlotManager
│   │   │   └── broadcaster.py         # Broadcaster (SSE)
│   │   └── schemas/                   # Pydantic 입출력 스키마
│   ├── model.py                       # BearingClassifier (nn.Module)
│   ├── trainer.py                     # Trainer (학습 루프)
│   ├── data_loader.py                 # CWRUDataLoader
│   ├── dataset.py                     # BearingDataset
│   ├── feature_extraction.py          # 19차원 특징 추출
│   ├── inference.py                   # predict_signal (윈도우 집계)
│   ├── artifact.py                    # ModelMeta, save/load
│   ├── config.py                      # 전역 하이퍼파라미터
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.vue
    │   ├── main.ts
    │   ├── api/client.ts              # axios 인스턴스 + 타입
    │   ├── router/index.ts
    │   ├── stores/training.ts         # Pinia (SSE 연결 포함)
    │   ├── views/
    │   │   ├── PredictView.vue
    │   │   ├── TrainView.vue
    │   │   └── HistoryView.vue
    │   └── components/MetricChart.vue
    ├── vite.config.ts
    └── package.json
```

## 4. 백엔드 레이어

### 4.1 API Layer (`backend/app/api/`)

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/api/health` | GET | 헬스 체크 |
| `/api/train/start` | POST | 학습 시작 (TrainParams) |
| `/api/train/cancel` | POST | 학습 취소 |
| `/api/train/events` | GET | SSE 스트림 (snapshot/epoch/complete/cancelled/error) |
| `/api/predict` | POST | .mat 업로드 → 추론 |
| `/api/model/current` | GET | 현재 활성 모델 정보 |
| `/api/models` | GET | 모델 아티팩트 목록 |
| `/api/models/{id}/promote` | POST | 활성 모델 변경 |
| `/api/inferences` | GET | 추론 로그 조회 |

### 4.2 Service Layer (`backend/app/services/`)

- **TrainingService** (`services/training.py:35`)
  학습 작업 비동기 오케스트레이션. 별도 스레드에서 PyTorch 학습을 실행하고 epoch 단위로 DB에 메트릭을 저장하며 Broadcaster로 SSE 이벤트를 게시함. 학습 종료 시 아티팩트를 저장하고 자동 승격(val_acc 기준) 정책을 적용.

- **InferenceService** (`services/inference_service.py:36`)
  추론 게이트웨이. 현재 활성 ModelArtifact를 DB에서 조회 → GpuSlotManager로 메모리에 보장 → `predict_signal` 호출 → InferenceLog 기록.

- **GpuSlotManager** (`services/gpu_slot.py:28`)
  단일 GPU/CPU 슬롯에 한 모델만 적재하여 메모리를 관리. `arch_hash`가 같으면 재사용, 다르면 release 후 새 모델 로드.

- **Broadcaster** (`services/broadcaster.py:15`)
  학습 스레드 → asyncio 이벤트 루프 → 다수 SSE 구독자로의 thread-safe fan-out 큐.

### 4.3 ML Core

- **BearingClassifier** (`backend/model.py:12`)
  Sequential MLP(BatchNorm + ReLU + Dropout). 입력 19, 은닉 [128, 64, 32], 출력 클래스 수.
- **Trainer** (`backend/trainer.py:22`)
  Adam + ReduceLROnPlateau, Early Stopping(patience=10), 베스트 가중치 추적.
- **CWRUDataLoader** (`backend/data_loader.py:21`)
  .mat 로드 → 슬라이딩 윈도우(1024 샘플, overlap 0.5) → 특징 추출 → stratified split → StandardScaler.
- **extract_features** (`backend/feature_extraction.py:10`)
  시간 도메인 13개 + 주파수 도메인 6개 = 총 19차원.
- **predict_signal** (`backend/inference.py:48`)
  윈도우별 추론 후 softmax 확률 평균으로 최종 라벨 집계.

### 4.4 Artifact Layer (`backend/artifact.py`)

- `ModelMeta` (dataclass): 아키텍처/전처리 파라미터 직렬화.
- `LoadedArtifact` (dataclass): 메모리에 적재된 `(model, scaler, meta)` 묶음.
- `save_artifact()` / `load_artifact()`: `model.pt + scaler.joblib + meta.json` 입출력.

### 4.5 Persistence (`backend/app/db.py`)

| 테이블 | 역할 |
|---|---|
| `TrainingRun` | 학습 작업 메타(상태/시작·종료/베스트 메트릭/에러) |
| `EpochMetric` | epoch 단위 손실·정확도·LR |
| `ModelArtifact` | 모델 레지스트리(아티팩트 경로, arch_hash, is_current) |
| `InferenceLog` | 추론 감사 로그 |

## 5. 프론트엔드 레이어

- **App.vue**: 헤더 + 라우터 뷰 + 5초 주기 헬스 체크.
- **PredictView**: .mat 업로드 → `/api/predict` → 집계 결과 + 윈도우별 결과 표시.
- **TrainView**: 하이퍼파라미터 폼 + 시작/취소 + 실시간 손실·정확도 차트.
- **HistoryView**: 학습 런·아티팩트·추론 로그 조회 + 모델 승격 버튼.
- **useTrainingStore (Pinia)**: SSE 연결 관리 (`/api/train/events` 구독), `status/runId/history/lastError/lastComplete` 상태 보관.
- **MetricChart**: Chart.js 기반 학습 곡선 컴포넌트.

## 6. 핵심 데이터 플로우

### 6.1 학습 (Training)

1. `TrainView` → `POST /api/train/start` (TrainParams)
2. `TrainingService.start()`가 `TrainingRun`을 RUNNING으로 생성, asyncio 태스크 시작
3. 별도 스레드에서 `_train_blocking()` 실행
   - `CWRUDataLoader.load_data()` → `split_data()` → `normalize_data()`
   - `BearingClassifier` 인스턴스화 → `Trainer.train()`
4. epoch 콜백: `EpochMetric` 저장 + `Broadcaster.publish_threadsafe(epoch)`
5. 종료 시 `save_artifact()` → `ModelArtifact` 저장(자동 승격 시 `is_current=True`)
6. `complete` 이벤트 브로드캐스트 → 프론트 차트/스토어 갱신

### 6.2 추론 (Prediction)

1. `PredictView` → `POST /api/predict` (multipart .mat)
2. `InferenceService.predict_mat_bytes()`
   - 현재 `ModelArtifact` 조회 → `GpuSlotManager.acquire()`로 모델 보장
   - `.mat`에서 DE_time 신호 추출
   - `predict_signal()`로 윈도우 단위 추론 → 확률 평균 집계
3. `InferenceLog` 기록 → JSON 응답 반환
4. 프론트가 집계 라벨 + 윈도우별 결과를 렌더링

## 7. 기술 스택

### Backend (`backend/requirements.txt`)
FastAPI ≥0.110, Uvicorn ≥0.30, SQLModel ≥0.0.16, sse-starlette ≥2.1,
PyTorch ≥2.0, NumPy ≥1.24, SciPy ≥1.11, scikit-learn ≥1.3, joblib ≥1.3,
Matplotlib ≥3.7, Seaborn ≥0.12.

### Frontend (`frontend/package.json`)
Vue 3.5, vue-router 4.4, Pinia 2.2, Axios 1.7, Vite 5.4, TypeScript 5.5,
Tailwind CSS 3.4, Chart.js 4.4, vue-chartjs 5.3.

## 8. 설계 포인트

- **단일 GPU 슬롯**: 메모리 효율을 위해 한 번에 하나의 모델만 적재. `arch_hash`로 재사용 판단.
- **스레드 ↔ asyncio 브릿지**: 학습은 별도 스레드(블로킹), 이벤트는 `Broadcaster`가 `loop.call_soon_threadsafe`로 안전 전달.
- **자동 승격 정책**: 새 학습 결과가 더 좋으면 `is_current` 플래그를 자동 갱신.
- **윈도우 집계 추론**: 단일 신호를 다수 윈도우로 분할해 확률을 평균하여 노이즈에 강함.
- **아티팩트 자기 기술성**: `meta.json`에 전처리/아키텍처 파라미터를 모두 저장 → 로드 시 복원 가능.
