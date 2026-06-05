def test_get_nonexistent_task(client):
    r = client.get("/api/v1/tasks/nonexistent-task-id")
    assert r.status_code == 404
