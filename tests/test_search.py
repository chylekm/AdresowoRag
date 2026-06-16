def test_search_returns_results(client):
    response = client.post(
        "/rag/search",
        json={"query": "kawalerka Białołęka", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "kawalerka Białołęka"
    assert len(payload["results"]) > 0
    assert any("Białołęka" in (hit.get("district") or "") for hit in payload["results"])


def test_search_with_filters(client):
    response = client.post(
        "/rag/search",
        json={
            "query": "mieszkanie z balkonem",
            "top_k": 5,
            "rooms": 2,
            "price_max": 900000,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    for hit in payload["results"]:
        if hit["rooms"] is not None:
            assert hit["rooms"] == 2
        if hit["price_total_zl"] is not None:
            assert hit["price_total_zl"] <= 900000
