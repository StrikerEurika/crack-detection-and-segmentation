from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRACK_API_", env_file=".env")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    project_root: Path = Path(__file__).resolve().parent.parent
    default_model_path: str = str(project_root / "checkpoints" / "best_model.pth")
    default_model_type: str = "auto"

    output_dir: Path = project_root / "output"
    api_results_dir: Path = output_dir / "api_results"
    data_dir: Path = project_root / "data"
    image_dir: Path = data_dir / "image"
    checkpoints_dir: Path = project_root / "checkpoints"
    dataset_dir: Path = data_dir / "crack_segmentation_dataset"

    max_upload_size_mb: int = 200
    default_tile_size_unet: int = 512
    default_tile_size_yolo: int = 640
    default_overlap_unet: int = 64
    default_overlap_yolo: int = 96
    default_threshold: float = 0.5
    default_min_area: int = 20
    default_conf_yolo: float = 0.18
    default_iou_yolo: float = 0.45

    model_class: str = ""


settings = Settings()
