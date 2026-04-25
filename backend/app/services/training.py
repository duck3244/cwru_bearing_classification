# ============================================================================
# 파일: app/services/training.py
# 설명: 학습 작업 라이프사이클 (asyncio + 별도 스레드)
# ============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import torch
from sqlmodel import Session, select
from torch.utils.data import DataLoader

from app.db import EpochMetric, ModelArtifact, TrainingRun, engine
from app.services.broadcaster import Broadcaster
from app.services.gpu_slot import GpuSlotManager
from artifact import ModelMeta, compute_arch_hash, save_artifact
from config import Config
from data_loader import CWRUDataLoader
from dataset import BearingDataset
from model import BearingClassifier
from trainer import Trainer
from utils import set_seed

logger = logging.getLogger(__name__)


class TrainingService:
    """단일 학습 작업 라이프사이클 매니저"""

    def __init__(self, slot: GpuSlotManager, broadcaster: Broadcaster) -> None:
        self._slot = slot
        self._broadcaster = broadcaster
        self._lock = asyncio.Lock()
        self._running = False
        self._cancel_event = threading.Event()
        self._current_run_id: Optional[str] = None
        self._snapshot: dict = {'status': 'IDLE', 'history': []}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def snapshot(self) -> dict:
        return self._snapshot

    async def start(self, params: dict) -> str:
        async with self._lock:
            if self._running:
                raise RuntimeError('Training already running')
            run_id = uuid4().hex
            now = datetime.now(timezone.utc)
            run = TrainingRun(
                id=run_id,
                started_at=now,
                status='RUNNING',
                config_snapshot=json.dumps(params, ensure_ascii=False, default=str),
                n_epochs_run=0,
            )
            with Session(engine) as s:
                s.add(run); s.commit()

            self._running = True
            self._cancel_event.clear()
            self._current_run_id = run_id
            self._snapshot = {
                'run_id': run_id,
                'status': 'RUNNING',
                'started_at': now.isoformat(),
                'history': [],
                'params': params,
            }
            await self._slot.release()  # 학습 시작 전 GPU 슬롯 비움
            asyncio.create_task(self._run(run_id, params))
            return run_id

    async def cancel(self) -> bool:
        if not self._running:
            return False
        self._cancel_event.set()
        return True

    async def _run(self, run_id: str, params: dict) -> None:
        try:
            await asyncio.to_thread(self._train_blocking, run_id, params)
        except Exception as e:  # noqa: BLE001
            logger.exception('Training failed')
            self._mark_failed(run_id, str(e))
            self._broadcaster.publish_threadsafe(
                {'type': 'error', 'run_id': run_id, 'message': str(e)}
            )
        finally:
            self._running = False
            self._current_run_id = None

    # ---------------- blocking thread ----------------

    def _train_blocking(self, run_id: str, params: dict) -> None:
        cfg = Config()
        # params override
        for key in ('EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'WEIGHT_DECAY',
                    'DROPOUT', 'EARLY_STOPPING_PATIENCE'):
            if key in params:
                setattr(cfg, key, params[key])
        if 'HIDDEN_SIZES' in params:
            cfg.HIDDEN_SIZES = list(params['HIDDEN_SIZES'])
        cfg.DEVICE = self._slot.device  # 학습은 슬롯과 같은 디바이스 사용

        set_seed(cfg.RANDOM_STATE)

        loader_obj = CWRUDataLoader(cfg)
        X, y = loader_obj.load_data()
        X_tr, X_val, X_te, y_tr, y_val, y_te = loader_obj.split_data(X, y)
        X_tr, X_val, X_te = loader_obj.normalize_data(X_tr, X_val, X_te)

        gen = torch.Generator().manual_seed(cfg.RANDOM_STATE)
        common = {'batch_size': cfg.BATCH_SIZE,
                  'num_workers': cfg.NUM_WORKERS,
                  'pin_memory': cfg.PIN_MEMORY}
        train_loader = DataLoader(BearingDataset(X_tr, y_tr), shuffle=True,
                                  generator=gen, **common)
        val_loader = DataLoader(BearingDataset(X_val, y_val), shuffle=False, **common)

        input_size = X_tr.shape[1]
        model = BearingClassifier(input_size=input_size,
                                  hidden_sizes=cfg.HIDDEN_SIZES,
                                  num_classes=cfg.NUM_CLASSES,
                                  dropout=cfg.DROPOUT)
        trainer = Trainer(model, cfg)

        def on_epoch(metrics: dict) -> None:
            with Session(engine) as s:
                s.add(EpochMetric(run_id=run_id, **metrics))
                run = s.get(TrainingRun, run_id)
                if run is not None:
                    run.n_epochs_run = metrics['epoch']
                    run.best_val_acc = trainer.best_val_acc
                    run.best_val_loss = trainer.best_val_loss
                    run.best_epoch = trainer.best_epoch
                    s.add(run)
                s.commit()
            self._snapshot['history'].append(metrics)
            self._broadcaster.publish_threadsafe(
                {'type': 'epoch', 'run_id': run_id, **metrics}
            )

        trainer.train(train_loader, val_loader, cfg.EPOCHS,
                      on_epoch_end=on_epoch,
                      should_stop=self._cancel_event.is_set)

        if self._cancel_event.is_set():
            self._mark_cancelled(run_id)
            self._broadcaster.publish_threadsafe(
                {'type': 'cancelled', 'run_id': run_id, 'reason': 'user'}
            )
            return

        if trainer.best_state_dict is None:
            self._mark_failed(run_id, 'No improvement during training')
            return

        # 아티팩트 저장
        artifact_id = uuid4().hex
        artifact_dir = os.path.join(cfg.MODEL_SAVE_PATH, artifact_id)
        meta = ModelMeta(
            artifact_id=artifact_id,
            arch_hash=compute_arch_hash(input_size, list(cfg.HIDDEN_SIZES),
                                        cfg.NUM_CLASSES, cfg.DROPOUT),
            input_size=input_size,
            hidden_sizes=list(cfg.HIDDEN_SIZES),
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
            sampling_rate=cfg.SAMPLING_RATE,
            window_size=cfg.WINDOW_SIZE,
            overlap=cfg.OVERLAP,
            class_names=list(cfg.CLASS_NAMES),
            label_map=dict(cfg.LABEL_MAP),
            val_acc=trainer.best_val_acc,
            val_loss=trainer.best_val_loss,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        save_artifact(artifact_dir, trainer.best_state_dict, loader_obj.scaler, meta)

        promoted = self._register_artifact(run_id, artifact_id, artifact_dir, meta)

        self._broadcaster.publish_threadsafe({
            'type': 'complete', 'run_id': run_id,
            'best_val_acc': trainer.best_val_acc,
            'best_val_loss': trainer.best_val_loss,
            'best_epoch': trainer.best_epoch,
            'artifact_id': artifact_id,
            'promoted': promoted,
        })

    # ---------------- DB helpers ----------------

    def _register_artifact(self, run_id: str, artifact_id: str, artifact_dir: str,
                           meta: ModelMeta) -> bool:
        """val_acc 비교 → 자동 promotion 여부 결정"""
        now = datetime.now(timezone.utc)
        with Session(engine) as s:
            current = s.exec(
                select(ModelArtifact).where(ModelArtifact.is_current == True)  # noqa: E712
            ).first()
            promote = current is None or (meta.val_acc or 0) > (current.val_acc or 0)
            if promote and current is not None:
                current.is_current = False
                s.add(current)
            artifact = ModelArtifact(
                id=artifact_id, run_id=run_id, artifact_dir=artifact_dir,
                arch_hash=meta.arch_hash,
                val_acc=meta.val_acc or 0.0,
                val_loss=meta.val_loss or 0.0,
                is_current=promote, created_at=now,
            )
            s.add(artifact)
            run = s.get(TrainingRun, run_id)
            if run is not None:
                run.status = 'COMPLETED'
                run.finished_at = now
                run.artifact_id = artifact_id
                s.add(run)
            s.commit()
        return promote

    def _mark_failed(self, run_id: str, error: str) -> None:
        with Session(engine) as s:
            run = s.get(TrainingRun, run_id)
            if run is not None:
                run.status = 'FAILED'
                run.finished_at = datetime.now(timezone.utc)
                run.error_message = error
                s.add(run); s.commit()
        self._snapshot['status'] = 'FAILED'

    def _mark_cancelled(self, run_id: str) -> None:
        with Session(engine) as s:
            run = s.get(TrainingRun, run_id)
            if run is not None:
                run.status = 'CANCELLED'
                run.finished_at = datetime.now(timezone.utc)
                s.add(run); s.commit()
        self._snapshot['status'] = 'CANCELLED'
