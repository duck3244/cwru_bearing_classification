from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from sqlmodel import Session, select

from app.db import ModelArtifact, engine
from app.schemas.model_registry import ArtifactOut, CurrentModelOut

router = APIRouter(prefix='/api', tags=['models'])


@router.get('/model/current', response_model=CurrentModelOut)
async def current_model(request: Request) -> CurrentModelOut:
    slot = request.app.state.slot
    with Session(engine) as s:
        current = s.exec(
            select(ModelArtifact).where(ModelArtifact.is_current == True)  # noqa: E712
        ).first()
    if current is None:
        return CurrentModelOut(
            artifact_id=None, val_acc=None, val_loss=None, arch_hash=None,
            class_names=None, input_size=None, hidden_sizes=None,
            created_at=None, device=slot.device_str,
        )

    state = await slot.acquire(current.id, current.arch_hash, current.artifact_dir)
    meta = state.artifact.meta
    return CurrentModelOut(
        artifact_id=current.id,
        val_acc=current.val_acc,
        val_loss=current.val_loss,
        arch_hash=current.arch_hash,
        class_names=meta.class_names,
        input_size=meta.input_size,
        hidden_sizes=meta.hidden_sizes,
        created_at=current.created_at,
        device=slot.device_str,
    )


@router.get('/models', response_model=list[ArtifactOut])
def list_models() -> list[ArtifactOut]:
    with Session(engine) as s:
        rows = s.exec(select(ModelArtifact).order_by(ModelArtifact.created_at.desc())).all()
    return [ArtifactOut.model_validate(r, from_attributes=True) for r in rows]


@router.post('/models/{artifact_id}/promote', response_model=ArtifactOut)
async def promote(artifact_id: str, request: Request) -> ArtifactOut:
    if request.app.state.training.is_running:
        raise HTTPException(409, 'Cannot promote during training')
    with Session(engine) as s:
        target = s.get(ModelArtifact, artifact_id)
        if target is None:
            raise HTTPException(404, 'Artifact not found')
        for row in s.exec(select(ModelArtifact).where(ModelArtifact.is_current == True)):  # noqa: E712
            row.is_current = False
            s.add(row)
        target.is_current = True
        s.add(target); s.commit(); s.refresh(target)
    # 슬롯 즉시 갱신
    await request.app.state.slot.acquire(target.id, target.arch_hash, target.artifact_dir)
    return ArtifactOut.model_validate(target, from_attributes=True)
