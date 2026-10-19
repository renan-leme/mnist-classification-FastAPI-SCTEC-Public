"""FastAPI entry point: python -m uvicorn app.main:app --reload."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routers import router
from app.services import PredictionService


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_pipeline.joblib"


def create_app(model_path: Path = MODEL_PATH):
    """Build the application without loading the artifact at import time."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = PredictionService()
        service.load(model_path)  # One model load per application process.
        application.state.prediction_service = service
        try:
            yield
        finally:
            service.clear()

    application = FastAPI(
        title="MNIST Classification API",
        description="Option A: reuse the complete MNIST notebook and serve a trained classifier.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        # Avoid echoing input values. Even non-finite values such as NaN must
        # produce a serializable 422 response instead of crashing the server.
        details = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    return application


app = create_app()
