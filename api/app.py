from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import health, models, inference, results, tasks
from api.services.task_manager import task_manager
from src.utils.logger import setup_logger

logger = setup_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting API server on {settings.host}:{settings.port} "
        f"(reload={settings.reload})"
    )
    await task_manager.start()
    yield
    await task_manager.stop()
    logger.info("API server stopped.")


app = FastAPI(
    title="Crack Detection API",
    description="""
AI-powered microscopic crack detection and analysis API.

## Capabilities
- **Single Image Inference** — Upload a microscopic image and detect cracks using deep learning models (UNet, UNet++, YOLO).
- **Inference + Evaluation** — Upload both image and ground-truth mask to get precision/recall/F1 metrics alongside predictions.
- **Batch Processing** — Submit multiple images for async batch inference with progress tracking.
- **Model Management** — Load, switch, and unload different model checkpoints at runtime.
- **Visualization** — Retrieve overlay, heatmap, and mask visualizations for each prediction.

## Key Features
- **Tiled Inference** — Handles large images by processing in overlapping tiles with cosine blending.
- **Marker Suppression** — Automatically removes colored annotation markers that could interfere with detection.
- **Skeletonization** — Extracts crack centerlines and estimates length/width dimensions.
- **Async Processing** — Batch jobs run in background worker threads with status polling.
""",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Crack Detection Team",
        "url": "https://github.com/anomalyco/microscopic-crack-detection",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = "/api/v1"
app.include_router(health.router, prefix=api_prefix)
app.include_router(models.router, prefix=api_prefix)
app.include_router(inference.router, prefix=api_prefix)
app.include_router(results.router, prefix=api_prefix)
app.include_router(tasks.router, prefix=api_prefix)



from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/")
async def root():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(str(html_path))



def main():
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )






