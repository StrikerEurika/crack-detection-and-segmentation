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
    description="REST API for microscopic crack detection: inference, evaluation, and result management.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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



@app.get("/")
async def root():
    return {
        "name": "Crack Detection API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


def main():
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
