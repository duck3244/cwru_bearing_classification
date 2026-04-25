# ============================================================================
# 파일: app/main.py
# 설명: FastAPI 진입점
# ============================================================================

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, inferences, models, predict, train
from app.db import cleanup_stale_runs, get_current_artifact, init_db
from app.services.broadcaster import Broadcaster
from app.services.gpu_slot import GpuSlotManager
from app.services.inference_service import InferenceService
from app.services.training import TrainingService
from utils import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging('INFO')
    logger.info('Starting backend...')

    init_db()
    n_stale = cleanup_stale_runs()
    if n_stale:
        logger.warning('Marked %d stale RUNNING run(s) as FAILED', n_stale)

    slot = GpuSlotManager()
    broadcaster = Broadcaster()
    broadcaster.attach_loop(asyncio.get_running_loop())
    training = TrainingService(slot, broadcaster)
    inference_service = InferenceService(slot, lambda: training.is_running)

    # 서버 시작 시 현재 모델 워밍업
    current = get_current_artifact()
    if current is not None:
        try:
            await slot.acquire(current.id, current.arch_hash, current.artifact_dir)
            logger.info('Warmed up current model: %s (val_acc=%.2f)',
                        current.id, current.val_acc)
        except Exception as e:  # noqa: BLE001
            logger.warning('Failed to warm up current model %s: %s', current.id, e)

    app.state.slot = slot
    app.state.broadcaster = broadcaster
    app.state.training = training
    app.state.inference_service = inference_service

    yield

    logger.info('Shutting down backend...')
    await slot.release()


app = FastAPI(title='CWRU Bearing Classification API', version='0.1.0', lifespan=lifespan)

# dev 환경 (Vite proxy 사용 시 보통 불필요하지만 보호 차원에서)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(inferences.router)
