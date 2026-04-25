from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.db import EpochMetric, TrainingRun, engine
from app.schemas.training import (
    CancelResponse,
    EpochOut,
    RunDetail,
    RunOut,
    TrainParams,
    TrainStartResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/train', tags=['train'])


@router.post('/start', response_model=TrainStartResponse)
async def start(params: TrainParams, request: Request) -> TrainStartResponse:
    service = request.app.state.training
    try:
        run_id = await service.start(params.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return TrainStartResponse(run_id=run_id)


@router.post('/cancel', response_model=CancelResponse)
async def cancel(request: Request) -> CancelResponse:
    cancelled = await request.app.state.training.cancel()
    return CancelResponse(cancelled=cancelled)


@router.get('/runs', response_model=list[RunOut])
def list_runs() -> list[RunOut]:
    with Session(engine) as s:
        rows = s.exec(select(TrainingRun).order_by(TrainingRun.started_at.desc())).all()
    return [RunOut.model_validate(r, from_attributes=True) for r in rows]


@router.get('/runs/{run_id}', response_model=RunDetail)
def run_detail(run_id: str) -> RunDetail:
    with Session(engine) as s:
        run = s.get(TrainingRun, run_id)
        if run is None:
            raise HTTPException(404, 'Run not found')
        epochs = s.exec(
            select(EpochMetric).where(EpochMetric.run_id == run_id).order_by(EpochMetric.epoch)
        ).all()
    return RunDetail(
        **RunOut.model_validate(run, from_attributes=True).model_dump(),
        epochs=[EpochOut.model_validate(e, from_attributes=True) for e in epochs],
        config_snapshot=json.loads(run.config_snapshot or '{}'),
    )


@router.get('/runs/{run_id}/epochs', response_model=list[EpochOut])
def run_epochs(run_id: str, since: int = 0) -> list[EpochOut]:
    """Polling용. since 이후 epoch만 반환"""
    with Session(engine) as s:
        rows = s.exec(
            select(EpochMetric)
            .where(EpochMetric.run_id == run_id, EpochMetric.epoch > since)
            .order_by(EpochMetric.epoch)
        ).all()
    return [EpochOut.model_validate(r, from_attributes=True) for r in rows]


@router.get('/events')
async def events(request: Request):
    """SSE 스트림 — 연결 즉시 snapshot, 이후 실시간 epoch/complete/cancelled/error 이벤트"""
    service = request.app.state.training
    broadcaster = request.app.state.broadcaster

    async def gen():
        snapshot = service.snapshot
        yield {'event': 'snapshot', 'data': json.dumps(snapshot, ensure_ascii=False, default=str)}

        q = await broadcaster.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {'event': 'ping', 'data': ''}
                    continue
                ev_type = event.pop('type', 'message')
                yield {'event': ev_type,
                       'data': json.dumps(event, ensure_ascii=False, default=str)}
        finally:
            broadcaster.unsubscribe(q)

    return EventSourceResponse(gen())
