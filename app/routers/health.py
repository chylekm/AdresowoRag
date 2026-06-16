from fastapi import APIRouter

from app.deps import app_state
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = app_state.settings
    vector_store = app_state.vector_store
    index_loaded = bool(vector_store and vector_store.is_loaded)
    n_documents = vector_store.n_documents if vector_store else 0

    return HealthResponse(
        status="ok" if index_loaded else "degraded",
        version=settings.app_version if settings else "unknown",
        index_loaded=index_loaded,
        n_documents=n_documents,
        embed_model=settings.embed_model if settings else "unknown",
        llm_model=settings.openrouter_model if settings else "unknown",
        llm_configured=settings.llm_configured if settings else False,
    )
