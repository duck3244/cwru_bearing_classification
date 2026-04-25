from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas.inference import InferenceResponse
from app.services.inference_service import (
    NoCurrentModelError,
    TrainingInProgressError,
)

router = APIRouter(prefix='/api', tags=['predict'])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post('/predict', response_model=InferenceResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> InferenceResponse:
    if not file.filename or not file.filename.lower().endswith('.mat'):
        raise HTTPException(400, 'Only .mat files are accepted')

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'File too large (>{MAX_UPLOAD_BYTES} bytes)')

    service = request.app.state.inference_service
    try:
        return await service.predict_mat_bytes(file.filename, content)
    except TrainingInProgressError:
        raise HTTPException(409, 'Training in progress; inference unavailable')
    except NoCurrentModelError:
        raise HTTPException(404, 'No current model registered')
    except ValueError as e:
        raise HTTPException(400, str(e))
