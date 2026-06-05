import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.config import settings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_path():
    images = sorted(settings.image_dir.glob("*.jpg"))
    if images:
        return images[0]
    return None


@pytest.fixture
def sample_image_bytes(sample_image_path):
    if sample_image_path:
        with open(sample_image_path, "rb") as f:
            return f.read()
    return None
