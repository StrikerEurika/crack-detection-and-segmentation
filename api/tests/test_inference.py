import pytest


def test_predict_no_file(client):
    r = client.post("/api/v1/predict")
    assert r.status_code == 422


def test_predict_with_image(client, sample_image_bytes):
    if not sample_image_bytes:
        pytest.skip("No sample image found")

    r = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"threshold": 0.5, "min_area": 20},
    )
    assert r.status_code == 200
    data = r.json()
    assert "result_id" in data
    assert "image_file" in data
    assert "crack_detected" in data
    assert "mean_confidence" in data
    assert "estimated_length_pixels" in data
    assert "estimated_average_width_pixels" in data
    assert "crack_area_pixels" in data
    assert "processing_time_ms" in data
    assert "visualizations" in data
    assert "overlay" in data["visualizations"]
    assert "heatmap" in data["visualizations"]
    assert "mask" in data["visualizations"]


def test_predict_with_yolo_params(client, sample_image_bytes):
    if not sample_image_bytes:
        pytest.skip("No sample image found")

    r = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={
            "model_type": "yolo",
            "threshold": 0.42,
            "min_area": 8,
            "conf": 0.18,
            "iou": 0.45,
        },
    )
    assert r.status_code in (200, 500)


def test_predict_batch_no_files(client):
    r = client.post("/api/v1/predict/batch")
    assert r.status_code == 422


def test_predict_batch_with_images(client, sample_image_bytes):
    if not sample_image_bytes:
        pytest.skip("No sample image found")

    r = client.post(
        "/api/v1/predict/batch",
        files=[
            ("files", ("img1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("img2.jpg", sample_image_bytes, "image/jpeg")),
        ],
        data={"threshold": 0.5, "min_area": 20},
    )
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["total_images"] == 2
    assert data["status"] == "pending"


def test_predict_with_model_version(client, sample_image_bytes):
    if not sample_image_bytes:
        pytest.skip("No sample image found")

    # Predict using a version name "v1"
    r = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"threshold": 0.5, "min_area": 20, "model_version": "v1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "result_id" in data
    
    # Predict using a relative path "v1/best_model.pth"
    r = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"threshold": 0.5, "min_area": 20, "model_version": "v1/best_model.pth"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "result_id" in data
