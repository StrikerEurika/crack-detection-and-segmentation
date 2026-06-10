from fastapi import APIRouter, HTTPException

from api.schemas import TaskStatusResponse
from api.services.task_manager import task_manager

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get Task Status",
    description="Polls the status of an async task (e.g. batch inference). Returns the current status (pending/running/completed/failed), progress percentage, and result data when completed. Use after submitting a batch job via POST /api/v1/predict/batch.",
    response_description="Task status with optional progress and result",
    responses={404: {"description": "Task not found"}},
)
async def get_task_status(task_id: str):
    status = task_manager.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**status)
