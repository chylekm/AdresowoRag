from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district: str | None = Field(default=None, max_length=100)
    rooms: int | None = Field(default=None, ge=1, le=10)
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    area_min: float | None = Field(default=None, ge=0)
    area_max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SearchFilters":
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min nie może być większe niż price_max")
        if (
            self.area_min is not None
            and self.area_max is not None
            and self.area_min > self.area_max
        ):
            raise ValueError("area_min nie może być większe niż area_max")
        return self


class SearchHit(BaseModel):
    id: int
    score: float
    district: str | None = None
    rooms: int | None = None
    area: float | None = None
    price_total_zl: float | None = None
    full_address: str | None = None
    url: str | None = None
    transcription: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchHit]


class Source(BaseModel):
    id: int
    score: float
    district: str | None = None
    url: str | None = None
    transcription: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    model: str
    usage: TokenUsage


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    index_loaded: bool
    n_documents: int
    embed_model: str
    llm_model: str
    llm_configured: bool


class ReindexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: bool = False


class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: Literal["pending"] = "pending"
    message: str = "Zadanie przyjęte do przetwarzania"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["pending", "running", "done", "failed"]
    message: str | None = None
    n_documents: int | None = None


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
