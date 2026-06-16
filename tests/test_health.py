def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["index_loaded"] is True
    assert payload["n_documents"] > 0
    assert "embed_model" in payload
    assert "llm_model" in payload
    assert "llm_configured" in payload
