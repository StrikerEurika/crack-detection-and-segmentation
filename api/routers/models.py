from fastapi import APIRouter, HTTPException

from api.schemas import LoadModelRequest, ModelInfo
from api.services.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=list[dict],
    summary="List Available Models",
    description="Returns a list of all available model checkpoint files found in the checkpoints directory, including file name, type (unet, unet_plusplus_v1, yolo), size, and version information.",
    response_description="List of available model checkpoints",
)
async def list_models():
    return model_manager.list_available()


@router.get(
    "/current",
    response_model=ModelInfo,
    summary="Get Current Model",
    description="Returns information about the currently loaded model, including model ID, path, type, device, and when it was loaded. Returns 404 if no model is currently loaded.",
    response_description="Currently loaded model details",
    responses={404: {"description": "No model currently loaded"}},
)
async def current_model():
    info = model_manager.model_info
    if info is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    return ModelInfo(**info)


@router.post(
    "/load",
    response_model=ModelInfo,
    summary="Load a Model",
    description="Loads a model checkpoint from the specified path. Supports UNet, UNet++, and YOLO model types. Use a version name (e.g. 'v1'), relative path (e.g. 'v1/best_model.pth'), or absolute path to the checkpoint file.",
    response_description="Loaded model details",
    responses={
        404: {"description": "Model file not found"},
        400: {"description": "Invalid model type or parameters"},
    },
)
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


@router.delete(
    "/unload",
    summary="Unload Current Model",
    description="Unloads the currently loaded model from memory, freeing GPU/CPU resources. After unloading, inference endpoints will auto-load a model on the next request.",
    response_description="Unload confirmation",
)
async def unload_model():
    model_manager.unload()
    return {"status": "unloaded"}
