# ============================================================================
# 파일: app/db.py
# 설명: SQLite + SQLModel 엔진/세션
# ============================================================================

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import event
from sqlmodel import Field, Session, SQLModel, create_engine

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.environ.get('CWRU_DB_PATH', os.path.join(DB_DIR, 'cwru.sqlite'))
DB_URL = f'sqlite:///{DB_PATH}'

# SQLite는 sync only. WAL 모드는 connect 후 PRAGMA로 설정.
engine = create_engine(DB_URL, echo=False, connect_args={'check_same_thread': False})


@event.listens_for(engine, 'connect')
def _set_sqlite_pragmas(dbapi_connection, _):
    cur = dbapi_connection.cursor()
    cur.execute('PRAGMA journal_mode=WAL')
    cur.execute('PRAGMA synchronous=NORMAL')
    cur.execute('PRAGMA foreign_keys=ON')
    cur.close()


# ---------------------------------------------------------------------------
# 모델 정의
# ---------------------------------------------------------------------------

class TrainingRun(SQLModel, table=True):
    id: str = Field(primary_key=True)
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str  # RUNNING | COMPLETED | CANCELLED | FAILED
    config_snapshot: str  # JSON
    best_val_acc: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    n_epochs_run: int = 0
    artifact_id: Optional[str] = None
    error_message: Optional[str] = None


class EpochMetric(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    elapsed_ms: int


class ModelArtifact(SQLModel, table=True):
    id: str = Field(primary_key=True)
    run_id: str
    artifact_dir: str
    arch_hash: str
    val_acc: float
    val_loss: float
    is_current: bool = False
    created_at: datetime


class InferenceLog(SQLModel, table=True):
    id: str = Field(primary_key=True)
    timestamp: datetime
    filename: str
    n_windows: int
    aggregated_label: int
    aggregated_label_name: str
    aggregated_probs: str  # JSON
    processing_ms: int
    artifact_id: str


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """FastAPI Depends용 generator는 별도 함수에서 사용; 단발 호출은 with-block 권장."""
    return Session(engine)


def cleanup_stale_runs() -> int:
    """부팅 시 RUNNING 상태로 남은 row를 FAILED로 마킹"""
    from sqlmodel import select
    n = 0
    with Session(engine) as s:
        for run in s.exec(select(TrainingRun).where(TrainingRun.status == 'RUNNING')):
            run.status = 'FAILED'
            run.finished_at = datetime.utcnow()
            run.error_message = 'Server restarted while running'
            s.add(run)
            n += 1
        s.commit()
    return n


def get_current_artifact() -> Optional[ModelArtifact]:
    from sqlmodel import select
    with Session(engine) as s:
        return s.exec(select(ModelArtifact).where(ModelArtifact.is_current == True)).first()  # noqa: E712


def serialize_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
