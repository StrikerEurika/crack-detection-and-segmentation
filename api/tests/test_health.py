def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "models_loaded" in data
    assert "torch_version" in data


def test_info(client):
    r = client.get("/api/v1/info")
    assert r.status_code == 200
    data = r.json()
    assert data["project"] == "miroscopic-crack-detection"
    assert "device" in data
    assert "available_checkpoints" in data
    assert "default_model" in data
