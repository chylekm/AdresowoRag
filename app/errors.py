from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorResponse


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(detail=str(exc.detail), request_id=request_id)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(detail=str(exc.errors()), request_id=request_id)
    return JSONResponse(status_code=422, content=payload.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = ErrorResponse(
        detail="Wewnętrzny błąd przetwarzania",
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


def attach_request_id_middleware(app):
    async def middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        return await call_next(request)

    app.middleware("http")(middleware)
