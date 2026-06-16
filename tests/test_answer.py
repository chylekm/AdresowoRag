from app.deps import app_state
from app.services.llm import LLMResponse


def test_answer_with_mocked_llm(client, monkeypatch):
    assert app_state.settings is not None
    monkeypatch.setattr(app_state.settings, "openrouter_api_key", "test-key")

    def fake_generate(self, system_prompt, user_prompt, temperature=0.2, max_retries=3):
        return LLMResponse(
            content="Znaleziono mieszkania na Białołęce [1].",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=20,
        )

    monkeypatch.setattr("app.services.llm.LLMService.generate", fake_generate)

    response = client.post(
        "/rag/answer",
        json={"query": "jakie kawalerki są na Białołęce?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["answer"], str)
    assert len(payload["answer"]) > 0
    assert len(payload["sources"]) > 0
    assert payload["model"] == "test-model"
    assert payload["usage"]["prompt_tokens"] == 100
