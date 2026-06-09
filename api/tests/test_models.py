def test_list_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "file_name" in data[0]
        assert "type" in data[0]
        assert "size_mb" in data[0]


def test_current_model_endpoint(client):
    r = client.get("/api/v1/models/current")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert "model_id" in data
        assert "model_type" in data
        assert "device" in data


def test_load_model_not_found(client):
    r = client.post(
        "/api/v1/models/load",
        json={"model_path": "/nonexistent/model.pth", "model_type": "unet"},
    )
    assert r.status_code == 404


def test_unload_model(client):
    r = client.delete("/api/v1/models/unload")
    assert r.status_code == 200
    assert r.json() == {"status": "unloaded"}


def test_list_models_fields(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "version" in data[0]
        assert "relative_path" in data[0]


def test_load_versioned_model(client):
    # Try loading by version name
    r = client.post(
        "/api/v1/models/load",
        json={"model_path": "v1", "model_type": "unet"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"] == "v1"
    
    # Try loading by relative path
    r = client.post(
        "/api/v1/models/load",
        json={"model_path": "v1/best_model.pth", "model_type": "unet"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"] == "v1"
