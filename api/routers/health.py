import torch
from fastapi import APIRouter

from api.config import settings
from api.schemas import HealthResponse, InfoResponse
from api.services.model_manager import model_manager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=model_manager.device,
        models_loaded=1 if model_manager.is_loaded else 0,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )


@router.get("/info", response_model=InfoResponse)
async def info():
    available = model_manager.list_available()
    return InfoResponse(
        project="miroscopic-crack-detection",
        version="0.1.0",
        device=model_manager.device,
        available_checkpoints=available,
        default_model=settings.default_model_path,
    )
