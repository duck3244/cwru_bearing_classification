# CWRU Bearing Classification — UML 다이어그램

모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되어 있어 GitHub 등에서 바로 렌더링됩니다.

---

## 1. Class Diagram — 백엔드 핵심 클래스

```mermaid
classDiagram
    class BearingClassifier {
        +Sequential network
        +__init__(input_size, hidden_sizes, num_classes, dropout)
        +forward(x) Tensor
    }
    class Module {
        <<PyTorch>>
    }

    class BearingDataset {
        +Tensor features
        +Tensor labels
        +__len__() int
        +__getitem__(idx) Tuple
    }
    class Dataset {
        <<PyTorch>>
    }

    class CWRUDataLoader {
        +Config config
        +StandardScaler scaler
        +load_data() Tuple
        +_label_from_filename(name)
        +_load_de_signal(mat)
        +_sliding_windows(signal)
        +split_data(X, y) Tuple
        +normalize_data(X_train, X_val, X_test)
    }

    class Trainer {
        +BearingClassifier model
        +Optimizer optimizer
        +LRScheduler scheduler
        +dict history
        +float best_val_acc
        +int best_epoch
        +train_epoch() Tuple
        +evaluate(loader) Tuple
        +train(epochs, on_epoch_end) dict
        +load_best_into_model()
    }

    class ModelMeta {
        <<dataclass>>
        +str artifact_id
        +str arch_hash
        +int input_size
        +list hidden_sizes
        +int num_classes
        +float dropout
        +int sampling_rate
        +int window_size
        +float overlap
        +list class_names
        +dict label_map
        +float val_acc
        +float val_loss
        +str created_at
        +to_json() str
        +from_json(s) ModelMeta
    }

    class LoadedArtifact {
        <<dataclass>>
        +BearingClassifier model
        +StandardScaler scaler
        +ModelMeta meta
    }

    class InferenceResult {
        <<dataclass>>
        +int n_windows
        +int aggregated_label
        +str aggregated_label_name
        +list aggregated_probabilities
        +list windows
        +float processing_ms
    }

    Module <|-- BearingClassifier
    Dataset <|-- BearingDataset
    LoadedArtifact "1" o-- "1" BearingClassifier
    LoadedArtifact "1" o-- "1" ModelMeta
    Trainer "1" o-- "1" BearingClassifier
    CWRUDataLoader ..> BearingDataset : creates
```

---

## 2. Class Diagram — 서비스 / 인프라 레이어

```mermaid
classDiagram
    class TrainingService {
        -GpuSlotManager _slot
        -Broadcaster _broadcaster
        -bool _running
        -Event _cancel_event
        -dict _snapshot
        +start(params) str
        +cancel() bool
        -_run() async
        -_train_blocking() dict
        -_register_artifact(meta, artifact_dir)
    }

    class InferenceService {
        -GpuSlotManager _slot
        +predict_mat_bytes(filename, content) dict
        -_ensure_current() LoadedArtifact
        -_log_inference(result, artifact_id)
    }

    class GpuSlotManager {
        -SlotState _state
        -Lock _lock
        +acquire(artifact_id, arch_hash, dir) LoadedArtifact
        +release()
        +is_loaded(artifact_id) bool
        +current() Optional~SlotState~
    }

    class Broadcaster {
        -EventLoop _loop
        -set _subscribers
        +attach_loop(loop)
        +subscribe() Queue
        +unsubscribe(q)
        +publish_threadsafe(event)
    }

    class SlotState {
        <<dataclass>>
        +str artifact_id
        +str arch_hash
        +LoadedArtifact loaded
    }

    TrainingService "1" --> "1" GpuSlotManager : uses
    TrainingService "1" --> "1" Broadcaster : publishes
    InferenceService "1" --> "1" GpuSlotManager : uses
    GpuSlotManager "1" o-- "0..1" SlotState
    SlotState "1" o-- "1" LoadedArtifact
```

---

## 3. ER Diagram — 데이터베이스

```mermaid
erDiagram
    TrainingRun ||--o{ EpochMetric : has
    TrainingRun ||--o| ModelArtifact : produces
    ModelArtifact ||--o{ InferenceLog : used_by

    TrainingRun {
        string id PK
        datetime started_at
        datetime finished_at
        string status
        json config_snapshot
        float best_val_acc
        float best_val_loss
        int best_epoch
        int n_epochs_run
        string artifact_id FK
        string error_message
    }

    EpochMetric {
        int id PK
        string run_id FK
        int epoch
        float train_loss
        float train_acc
        float val_loss
        float val_acc
        float lr
        int elapsed_ms
    }

    ModelArtifact {
        string id PK
        string run_id FK
        string artifact_dir
        string arch_hash
        float val_acc
        float val_loss
        bool is_current
        datetime created_at
    }

    InferenceLog {
        string id PK
        datetime timestamp
        string filename
        int n_windows
        int aggregated_label
        string aggregated_label_name
        json aggregated_probs
        float processing_ms
        string artifact_id FK
    }
```

---

## 4. Sequence Diagram — 학습 (Training Flow)

```mermaid
sequenceDiagram
    actor U as User (TrainView)
    participant FE as Frontend (Pinia Store)
    participant API as /api/train/*
    participant TS as TrainingService
    participant Th as Training Thread
    participant DB as SQLite
    participant BC as Broadcaster
    participant SSE as /api/train/events

    U->>FE: Start (params)
    FE->>API: POST /api/train/start
    API->>TS: start(params)
    TS->>DB: INSERT TrainingRun(RUNNING)
    TS->>BC: publish(snapshot)
    TS-->>API: run_id
    API-->>FE: {run_id}
    FE->>SSE: EventSource subscribe
    SSE->>BC: subscribe()

    par Training in background
        TS->>Th: _train_blocking()
        loop epochs
            Th->>Th: Trainer.train_epoch / evaluate
            Th->>DB: INSERT EpochMetric
            Th->>BC: publish_threadsafe(epoch)
            BC-->>SSE: event(epoch)
            SSE-->>FE: SSE epoch
            FE->>FE: history push, MetricChart 갱신
        end
        Th->>Th: save_artifact()
        Th->>DB: INSERT ModelArtifact (is_current?)
        Th->>DB: UPDATE TrainingRun(COMPLETED)
        Th->>BC: publish_threadsafe(complete)
        BC-->>SSE: event(complete)
        SSE-->>FE: SSE complete
    end
```

---

## 5. Sequence Diagram — 추론 (Prediction Flow)

```mermaid
sequenceDiagram
    actor U as User (PredictView)
    participant FE as Frontend
    participant API as /api/predict
    participant IS as InferenceService
    participant DB as SQLite
    participant GS as GpuSlotManager
    participant PS as predict_signal
    participant Disk as Filesystem

    U->>FE: .mat 파일 업로드
    FE->>API: POST /api/predict (multipart)
    API->>IS: predict_mat_bytes(name, bytes)

    IS->>DB: SELECT ModelArtifact WHERE is_current
    DB-->>IS: artifact row

    alt 같은 arch_hash 적재됨
        IS->>GS: acquire(id, hash, dir)
        GS-->>IS: LoadedArtifact (재사용)
    else 다른 모델 적재됨
        IS->>GS: acquire(id, hash, dir)
        GS->>GS: release()
        GS->>Disk: load_artifact(dir)
        Disk-->>GS: model.pt + scaler + meta
        GS-->>IS: LoadedArtifact
    end

    IS->>PS: predict_signal(signal, artifact)
    Note over PS: sliding window → features →<br/>scaler → batch infer → softmax 평균
    PS-->>IS: InferenceResult

    IS->>DB: INSERT InferenceLog
    IS-->>API: dict
    API-->>FE: InferenceResponse JSON
    FE->>U: 집계 라벨 + 윈도우별 결과 표시
```

---

## 6. Component Diagram — 시스템 전체

```mermaid
flowchart LR
    subgraph FE [Frontend - Vue 3]
        App[App.vue]
        PV[PredictView]
        TV[TrainView]
        HV[HistoryView]
        Store[useTrainingStore<br/>Pinia]
        Chart[MetricChart<br/>Chart.js]
        AX[axios client]
        ES[EventSource]

        App --- PV
        App --- TV
        App --- HV
        TV --- Store
        TV --- Chart
        Store --- ES
        PV --- AX
        HV --- AX
        TV --- AX
    end

    subgraph BE [Backend - FastAPI]
        Health[/api/health/]
        TrainAPI[/api/train/*/]
        PredAPI[/api/predict/]
        ModelsAPI[/api/models/*/]
        InfAPI[/api/inferences/]

        TS[TrainingService]
        IS[InferenceService]
        GS[GpuSlotManager]
        BR[Broadcaster]

        ML[ML Core<br/>BearingClassifier<br/>Trainer<br/>predict_signal]
        AR[Artifact<br/>save/load]
    end

    subgraph ST [Storage]
        DB[(SQLite<br/>cwru.sqlite)]
        FS[(models/<br/>artifact dirs)]
        Data[(data/*.mat)]
    end

    AX -->|HTTP| TrainAPI
    AX -->|HTTP| PredAPI
    AX -->|HTTP| ModelsAPI
    AX -->|HTTP| InfAPI
    AX -->|HTTP| Health
    ES -->|SSE| TrainAPI

    TrainAPI --> TS
    PredAPI --> IS
    ModelsAPI --> GS
    InfAPI --> DB

    TS --> ML
    TS --> AR
    TS --> BR
    TS --> GS
    IS --> GS
    IS --> ML
    GS --> AR

    TS <--> DB
    IS <--> DB
    AR <--> FS
    ML --> Data
```

---

## 7. State Diagram — 학습 작업 상태

```mermaid
stateDiagram-v2
    [*] --> RUNNING : start()
    RUNNING --> COMPLETED : 정상 종료
    RUNNING --> CANCELLED : cancel() 호출
    RUNNING --> FAILED : 예외 발생
    COMPLETED --> [*]
    CANCELLED --> [*]
    FAILED --> [*]

    note right of RUNNING
        EpochMetric 누적 저장
        Broadcaster로 epoch 이벤트 발행
    end note

    note right of COMPLETED
        ModelArtifact 등록
        val_acc 비교 후 is_current 자동 승격
    end note
```

---

## 8. State Diagram — GPU 슬롯

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Loaded : acquire(new)
    Loaded --> Loaded : acquire(same arch_hash)
    Loaded --> Empty : release()
    Loaded --> Loaded : acquire(diff arch_hash)<br/>= release + load
```

---

## 9. 주요 파일 매핑

| 다이어그램 요소 | 소스 파일 | 라인 |
|---|---|---|
| BearingClassifier | `backend/model.py` | 12 |
| BearingDataset | `backend/dataset.py` | 11 |
| CWRUDataLoader | `backend/data_loader.py` | 21 |
| Trainer | `backend/trainer.py` | 22 |
| ModelMeta / LoadedArtifact | `backend/artifact.py` | 27 / 80 |
| InferenceResult | `backend/inference.py` | 32 |
| predict_signal | `backend/inference.py` | 48 |
| TrainingService | `backend/app/services/training.py` | 35 |
| InferenceService | `backend/app/services/inference_service.py` | 36 |
| GpuSlotManager | `backend/app/services/gpu_slot.py` | 28 |
| Broadcaster | `backend/app/services/broadcaster.py` | 15 |
| TrainingRun / EpochMetric / ModelArtifact / InferenceLog | `backend/app/db.py` | 38 / 52 / 64 / 75 |
| useTrainingStore | `frontend/src/stores/training.ts` | 8 |
| API client | `frontend/src/api/client.ts` | 3 |
