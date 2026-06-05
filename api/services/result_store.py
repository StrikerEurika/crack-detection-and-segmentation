from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from api.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("api.result_store")


class ResultRecord:
    def __init__(
        self,
        result_id: str,
        image_file: str,
        image_resolution: list[int],
        crack_detected: bool,
        mean_confidence: float,
        estimated_length_pixels: int,
        estimated_average_width_pixels: float,
        crack_area_pixels: int,
        centerline_coordinates: list[list[int]],
        processing_time_ms: float,
        evaluation_metrics: Optional[dict] = None,
    ):
        self.result_id = result_id
        self.image_file = image_file
        self.image_resolution = image_resolution
        self.crack_detected = crack_detected
        self.mean_confidence = mean_confidence
        self.estimated_length_pixels = estimated_length_pixels
        self.estimated_average_width_pixels = estimated_average_width_pixels
        self.crack_area_pixels = crack_area_pixels
        self.centerline_coordinates = centerline_coordinates
        self.processing_time_ms = processing_time_ms
        self.evaluation_metrics = evaluation_metrics or {}
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "image_file": self.image_file,
            "image_resolution": self.image_resolution,
            "crack_detected": self.crack_detected,
            "mean_confidence": self.mean_confidence,
            "estimated_length_pixels": self.estimated_length_pixels,
            "estimated_average_width_pixels": self.estimated_average_width_pixels,
            "crack_area_pixels": self.crack_area_pixels,
            "centerline_coordinates": self.centerline_coordinates,
            "processing_time_ms": self.processing_time_ms,
            "evaluation_metrics": self.evaluation_metrics,
            "created_at": self.created_at.isoformat(),
        }

    def to_summary(self) -> dict:
        return {
            "result_id": self.result_id,
            "image_file": self.image_file,
            "crack_detected": self.crack_detected,
            "mean_confidence": self.mean_confidence,
            "estimated_length_pixels": self.estimated_length_pixels,
            "crack_area_pixels": self.crack_area_pixels,
            "created_at": self.created_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
        }


class ResultStore:
    def __init__(self):
        self._results: dict[str, ResultRecord] = {}
        self._store_dir = settings.api_results_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def _result_dir(self, result_id: str) -> Path:
        d = self._store_dir / result_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_visualization(
        self, result_id: str, name: str, image: np.ndarray
    ) -> Path:
        rdir = self._result_dir(result_id)
        path = rdir / f"{name}.png"
        import cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), image)
        return path

    def save_mask_visualization(
        self, result_id: str, name: str, mask: np.ndarray
    ) -> Path:
        rdir = self._result_dir(result_id)
        path = rdir / f"{name}.png"
        import cv2
        cv2.imwrite(str(path), mask)
        return path

    def save_json(self, result_id: str, data: dict) -> Path:
        rdir = self._result_dir(result_id)
        path = rdir / "result.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def get_visualization_path(self, result_id: str, name: str) -> Optional[Path]:
        path = self._store_dir / result_id / f"{name}.png"
        if path.exists():
            return path
        return None

    def get_json_path(self, result_id: str) -> Optional[Path]:
        path = self._store_dir / result_id / "result.json"
        if path.exists():
            return path
        return None

    def store(self, record: ResultRecord) -> str:
        self._results[record.result_id] = record
        self.save_json(record.result_id, record.to_dict())
        return record.result_id

    def _load_from_disk(self, result_id: str) -> Optional[ResultRecord]:
        json_path = self.get_json_path(result_id)
        if json_path is None:
            return None
        try:
            with open(json_path) as f:
                data = json.load(f)
            record = ResultRecord(
                result_id=data["result_id"],
                image_file=data["image_file"],
                image_resolution=data["image_resolution"],
                crack_detected=data["crack_detected"],
                mean_confidence=data["mean_confidence"],
                estimated_length_pixels=data["estimated_length_pixels"],
                estimated_average_width_pixels=data["estimated_average_width_pixels"],
                crack_area_pixels=data["crack_area_pixels"],
                centerline_coordinates=data["centerline_coordinates"],
                processing_time_ms=data["processing_time_ms"],
                evaluation_metrics=data.get("evaluation_metrics"),
            )
            self._results[result_id] = record
            return record
        except Exception:
            return None

    def get(self, result_id: str) -> Optional[ResultRecord]:
        record = self._results.get(result_id)
        if record is None:
            record = self._load_from_disk(result_id)
        return record

    def list_results(self, skip: int = 0, limit: int = 50) -> list[dict]:
        self._scan_disk_results()
        all_records = sorted(
            self._results.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        page = all_records[skip : skip + limit]
        return [r.to_summary() for r in page]

    def _scan_disk_results(self):
        if not self._store_dir.exists():
            return
        for d in sorted(self._store_dir.iterdir()):
            if d.is_dir() and d.name != "_uploads":
                if d.name not in self._results:
                    self._load_from_disk(d.name)

    def total(self) -> int:
        self._scan_disk_results()
        return len(self._results)


result_store = ResultStore()
