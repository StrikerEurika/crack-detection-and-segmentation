from fastapi import APIRouter, HTTPException

from api.schemas import LoadModelRequest, ModelInfo
from api.services.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=list[dict])
async def list_models():
    return model_manager.list_available()


@router.get("/current", response_model=ModelInfo)
async def current_model():
    info = model_manager.model_info
    if info is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return ModelInfo(**info)


@router.post("/load", response_model=ModelInfo)
async def load_model(req: LoadModelRequest):
    try:
        info = model_manager.load(
            model_path=req.model_path,
            model_type=req.model_type,
            encoder=req.encoder,
        )
        return ModelInfo(**info)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unload")
async def unload_model():
    model_manager.unload()
    return {"status": "unloaded"}
