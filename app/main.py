from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse

from app.config import get_settings
from app.deps import app_state
from app.errors import (
    attach_request_id_middleware,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.routers import admin, health, rag
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


def _initialize_index() -> None:
    settings = get_settings()
    app_state.settings = settings
    app_state.embedding_service = EmbeddingService(settings.embed_model)
    app_state.vector_store = VectorStore(settings.index_dir)

    faiss_path = settings.faiss_index_path
    meta_path = settings.meta_path

    if not settings.data_path.exists():
        raise FileNotFoundError(f"Niepoprawny plik danych: {settings.data_path}")

    if app_state.vector_store.exists(faiss_path, meta_path):
        app_state.vector_store.load(faiss_path, meta_path)
        return

    app_state.vector_store.build_from_csv(
        csv_path=settings.data_path,
        embedding_service=app_state.embedding_service,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app_state.vector_store.save(faiss_path, meta_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _initialize_index()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Adresowo RAG API",
        description="REST API do wyszukiwania semantycznego i odpowiedzi RAG nad ogłoszeniami mieszkań w Warszawie.",
        version=settings.app_version,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    attach_request_id_middleware(application)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    application.include_router(health.router)
    application.include_router(rag.router)
    application.include_router(admin.router)

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return application


app = create_app()
