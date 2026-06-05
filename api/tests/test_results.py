def test_list_results_empty(client):
    r = client.get("/api/v1/results")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 0
    assert "results" in data


def test_get_nonexistent_result(client):
    r = client.get("/api/v1/results/nonexistent-id")
    assert r.status_code == 404


def test_get_nonexistent_visualization(client):
    r = client.get(
        "/api/v1/results/nonexistent-id/visualization?type=overlay"
    )
    assert r.status_code == 404


def test_get_invalid_visualization_type(client):
    r = client.get(
        "/api/v1/results/some-id/visualization?type=invalid"
    )
    assert r.status_code == 400
