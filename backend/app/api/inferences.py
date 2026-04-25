from __future__ import annotations

import json

from fastapi import APIRouter
from sqlmodel import Session, select

from app.db import InferenceLog, engine
from app.schemas.inference import InferenceLogOut

router = APIRouter(prefix='/api', tags=['inferences'])


@router.get('/inferences', response_model=list[InferenceLogOut])
def list_inferences(limit: int = 100) -> list[InferenceLogOut]:
    with Session(engine) as s:
        rows = s.exec(
            select(InferenceLog).order_by(InferenceLog.timestamp.desc()).limit(limit)
        ).all()
    return [
        InferenceLogOut(
            id=r.id, timestamp=r.timestamp, filename=r.filename,
            n_windows=r.n_windows, aggregated_label=r.aggregated_label,
            aggregated_label_name=r.aggregated_label_name,
            aggregated_probs=json.loads(r.aggregated_probs),
            processing_ms=r.processing_ms, artifact_id=r.artifact_id,
        )
        for r in rows
    ]
