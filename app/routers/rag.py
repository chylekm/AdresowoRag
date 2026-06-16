from fastapi import APIRouter, Body, HTTPException, Query

from app.deps import app_state
from app.schemas import AnswerResponse, SearchFilters, SearchResponse
from app.services.llm import LLMService
from app.services.rag import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])


def _get_rag_service() -> RAGService:
    vector_store, embedding_service, settings = app_state.ensure_ready()
    llm_service = LLMService(settings)
    return RAGService(vector_store, embedding_service, llm_service)


def _filters_from_params(
    district: str | None,
    rooms: int | None,
    price_min: float | None,
    price_max: float | None,
    area_min: float | None,
    area_max: float | None,
) -> SearchFilters:
    return SearchFilters(
        district=district,
        rooms=rooms,
        price_min=price_min,
        price_max=price_max,
        area_min=area_min,
        area_max=area_max,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Wyszukiwanie semantyczne",
    description="Przyjmuje zapytanie tekstowe (text/plain) i zwraca najbardziej podobne transkrypcje ogłoszeń.",
)
def rag_search(
    query: str = Body(
        ...,
        media_type="text/plain",
        min_length=1,
        max_length=500,
        description="Zapytanie tekstowe użytkownika",
        examples=["kawalerka na Białołęce"],
    ),
    top_k: int = Query(default=5, ge=1, le=20, description="Liczba zwracanych transkrypcji"),
    district: str | None = Query(default=None, max_length=100),
    rooms: int | None = Query(default=None, ge=1, le=10),
    price_min: float | None = Query(default=None, ge=0),
    price_max: float | None = Query(default=None, ge=0),
    area_min: float | None = Query(default=None, ge=0),
    area_max: float | None = Query(default=None, ge=0),
) -> SearchResponse:
    try:
        filters = _filters_from_params(
            district, rooms, price_min, price_max, area_min, area_max
        )
        rag_service = _get_rag_service()
        return rag_service.search(
            query=query.strip(),
            top_k=top_k,
            filters=filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Błąd przetwarzania wyszukiwania") from exc


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Odpowiedź RAG",
    description="Przyjmuje pytanie tekstowe (text/plain), wyszukuje transkrypcje i generuje odpowiedź z LLM.",
)
def rag_answer(
    query: str = Body(
        ...,
        media_type="text/plain",
        min_length=1,
        max_length=500,
        description="Pytanie tekstowe użytkownika",
        examples=["jakie mieszkania blisko metra mają balkon?"],
    ),
    top_k: int = Query(default=5, ge=1, le=20),
    temperature: float = Query(default=0.2, ge=0.0, le=1.0),
    district: str | None = Query(default=None, max_length=100),
    rooms: int | None = Query(default=None, ge=1, le=10),
    price_min: float | None = Query(default=None, ge=0),
    price_max: float | None = Query(default=None, ge=0),
    area_min: float | None = Query(default=None, ge=0),
    area_max: float | None = Query(default=None, ge=0),
) -> AnswerResponse:
    settings = app_state.settings
    if settings is None or not settings.llm_configured:
        raise HTTPException(
            status_code=503,
            detail="OpenRouter API key nie jest skonfigurowany. Ustaw OPENROUTER_API_KEY.",
        )

    try:
        filters = _filters_from_params(
            district, rooms, price_min, price_max, area_min, area_max
        )
        rag_service = _get_rag_service()
        return rag_service.answer(
            query=query.strip(),
            top_k=top_k,
            filters=filters,
            temperature=temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "OpenRouter" in message:
            raise HTTPException(status_code=502, detail=message) from exc
        raise HTTPException(status_code=500, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Błąd generowania odpowiedzi przez LLM") from exc
