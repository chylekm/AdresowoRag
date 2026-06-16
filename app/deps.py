from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.config import Settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass
class TaskRecord:
    task_id: str
    status: Literal["pending", "running", "done", "failed"]
    message: str | None = None
    n_documents: int | None = None


@dataclass
class TaskRegistry:
    tasks: dict[str, TaskRecord] = field(default_factory=dict)

    def create(self) -> TaskRecord:
        task_id = str(uuid.uuid4())
        record = TaskRecord(task_id=task_id, status="pending")
        self.tasks[task_id] = record
        return record

    def get(self, task_id: str) -> TaskRecord:
        if task_id not in self.tasks:
            raise KeyError(task_id)
        return self.tasks[task_id]


class AppState:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.vector_store: VectorStore | None = None
        self.embedding_service: EmbeddingService | None = None
        self.task_registry = TaskRegistry()

    def ensure_ready(self) -> tuple[VectorStore, EmbeddingService, Settings]:
        if self.vector_store is None or self.embedding_service is None or self.settings is None:
            raise RuntimeError("Aplikacja nie jest jeszcze gotowa")
        if not self.vector_store.is_loaded:
            raise RuntimeError("Indeks wektorowy nie jest załadowany")
        return self.vector_store, self.embedding_service, self.settings


app_state = AppState()
