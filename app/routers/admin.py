from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.deps import app_state
from app.schemas import ReindexRequest, TaskAcceptedResponse, TaskStatusResponse
from app.services.embeddings import EmbeddingService

router = APIRouter(tags=["admin"])


def _run_reindex(task_id: str, force: bool) -> None:
    record = app_state.task_registry.get(task_id)
    record.status = "running"
    record.message = "Budowanie indeksu w toku"

    try:
        settings = app_state.settings
        vector_store = app_state.vector_store
        embedding_service = app_state.embedding_service
        if settings is None or vector_store is None or embedding_service is None:
            raise RuntimeError("Aplikacja nie jest gotowa do reindeksacji")

        faiss_path = settings.faiss_index_path
        meta_path = settings.meta_path
        if vector_store.exists(faiss_path, meta_path) and not force:
            vector_store.load(faiss_path, meta_path)
            record.status = "done"
            record.message = "Indeks już istnieje, załadowano istniejący"
            record.n_documents = vector_store.n_documents
            return

        vector_store.build_from_csv(
            csv_path=settings.data_path,
            embedding_service=embedding_service,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vector_store.save(faiss_path, meta_path)
        record.status = "done"
        record.message = "Indeks został przebudowany"
        record.n_documents = vector_store.n_documents
    except Exception as exc:
        record.status = "failed"
        record.message = str(exc)


@router.post("/admin/reindex", status_code=202, response_model=TaskAcceptedResponse)
def reindex(request: ReindexRequest, background_tasks: BackgroundTasks) -> TaskAcceptedResponse:
    try:
        app_state.ensure_ready()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record = app_state.task_registry.create()
    background_tasks.add_task(_run_reindex, record.task_id, request.force)
    return TaskAcceptedResponse(task_id=record.task_id)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str) -> TaskStatusResponse:
    try:
        record = app_state.task_registry.get(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Brak zadania o podanym ID") from exc

    return TaskStatusResponse(
        task_id=record.task_id,
        status=record.status,
        message=record.message,
        n_documents=record.n_documents,
    )
