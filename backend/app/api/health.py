from fastapi import APIRouter, Request

router = APIRouter(prefix='/api', tags=['health'])


@router.get('/health')
def health(request: Request) -> dict:
    slot = request.app.state.slot
    state = slot.current
    return {
        'status': 'ok',
        'device': slot.device_str,
        'model_loaded': state is not None,
        'current_artifact_id': state.artifact_id if state else None,
        'training_running': request.app.state.training.is_running,
    }
