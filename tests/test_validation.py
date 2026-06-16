def test_top_k_zero_returns_422(client):
    response = client.post("/rag/search", json={"query": "test", "top_k": 0})
    assert response.status_code == 422


def test_empty_query_returns_422(client):
    response = client.post("/rag/search", json={"query": ""})
    assert response.status_code == 422


def test_unknown_field_returns_422(client):
    response = client.post(
        "/rag/search",
        json={"query": "mieszkanie", "unknown_field": "x"},
    )
    assert response.status_code == 422


def test_invalid_price_range_returns_422(client):
    response = client.post(
        "/rag/search",
        json={"query": "mieszkanie", "price_min": 900000, "price_max": 500000},
    )
    assert response.status_code == 422


def test_task_not_found_returns_404(client):
    response = client.get("/tasks/non-existent-task-id")
    assert response.status_code == 404
