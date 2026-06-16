from __future__ import annotations

from app.schemas import AnswerResponse, SearchFilters, SearchHit, SearchResponse, Source, TokenUsage
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import SearchResult, VectorStore

SYSTEM_PROMPT = (
    "Jesteś asystentem pomagającym w wyszukiwaniu mieszkań na sprzedaż w Warszawie. "
    "Odpowiadaj wyłącznie na podstawie podanych ogłoszeń, po polsku. "
    "Jeśli brakuje informacji, napisz to wprost. "
    "Cytuj źródła w formacie [1], [2] itd."
)


def _to_hit(result: SearchResult) -> SearchHit:
    return SearchHit(
        id=result.doc_id,
        score=result.score,
        district=result.district,
        rooms=result.rooms,
        area=result.area,
        price_total_zl=result.price_total_zl,
        full_address=result.full_address,
        url=result.url,
        transcription=result.text,
    )


def _to_source(result: SearchResult) -> Source:
    return Source(
        id=result.doc_id,
        score=result.score,
        district=result.district,
        url=result.url,
        transcription=result.text,
    )


class RAGService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service

    def search(self, query: str, top_k: int, filters: SearchFilters | None = None) -> SearchResponse:
        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.search(query_vector, top_k, filters)
        return SearchResponse(
            query=query,
            top_k=top_k,
            results=[_to_hit(result) for result in results],
        )

    def answer(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None = None,
        temperature: float = 0.2,
    ) -> AnswerResponse:
        if not self.llm_service.is_configured:
            raise RuntimeError("OpenRouter API key nie jest skonfigurowany")

        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.search(query_vector, top_k, filters)

        if not results:
            return AnswerResponse(
                query=query,
                answer="Nie znaleziono ogłoszeń pasujących do zapytania i filtrów.",
                sources=[],
                model=self.llm_service.settings.openrouter_model,
                usage=TokenUsage(),
            )

        context_blocks = []
        for idx, result in enumerate(results, start=1):
            context_blocks.append(
                f"[{idx}] Dzielnica: {result.district or 'brak'}\n"
                f"Adres: {result.full_address or 'brak'}\n"
                f"Pokoje: {result.rooms or 'brak'}, Metraż: {result.area or 'brak'} m2\n"
                f"Cena: {int(result.price_total_zl) if result.price_total_zl else 'brak'} zł\n"
                f"Treść: {result.text}"
            )

        user_prompt = (
            f"Pytanie użytkownika: {query}\n\n"
            "Dostępne ogłoszenia:\n"
            f"{chr(10).join(context_blocks)}\n\n"
            "Na podstawie powyższych ogłoszeń udziel zwięzłej odpowiedzi po polsku."
        )

        llm_response = self.llm_service.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        return AnswerResponse(
            query=query,
            answer=llm_response.content,
            sources=[_to_source(result) for result in results],
            model=llm_response.model,
            usage=TokenUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
            ),
        )
