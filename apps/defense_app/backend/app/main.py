from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.database import DefenseAppError

app = FastAPI(title="Fashion Defense App API")
app.include_router(api_router)


@app.exception_handler(DefenseAppError)
async def defense_app_error_handler(
    request: Request, exc: DefenseAppError
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "validation_error", "message": "请求参数不合法"}},
    )
