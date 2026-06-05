from fastapi import APIRouter, HTTPException

from api.schemas import TrainParams, TrainingStatusResponse, TaskStatusResponse
from api.services.training_service import start_training, get_training_status
from api.services.task_manager import task_manager

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/start", response_model=TaskStatusResponse)
async def training_start(params: TrainParams):
    try:
        task_id = await start_training(params.model_dump())
        return TaskStatusResponse(
            task_id=task_id,
            status="pending",
            message="Training task submitted",
            created_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{task_id}", response_model=TrainingStatusResponse)
async def training_status(task_id: str):
    status = get_training_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    return TrainingStatusResponse(**status)
