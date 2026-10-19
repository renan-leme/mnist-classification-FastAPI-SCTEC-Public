"""Map HTTP requests to the prediction service and documented responses."""

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from app.schemas import ErrorResponse, HealthResponse, PredictionRequest, PredictionResponse
from app.services import InvalidImageError, ModelNotLoadedError


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health", response_model=HealthResponse, tags=["Monitoring"],
    responses={503: {"model": HealthResponse, "description": "Model unavailable"}},
)
def health(request: Request, response: Response):
    """Check memory state without opening the model file again."""
    service = request.app.state.prediction_service
    loaded = service.pipeline is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=service.metadata.get("model_name"),
    )


@router.post(
    "/predict", response_model=PredictionResponse, tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Image contains no digit"},
        500: {"model": ErrorResponse, "description": "Unexpected inference error"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    },
)
def predict(payload: PredictionRequest, request: Request):
    """Recognize one digit after Pydantic validates the complete input."""
    service = request.app.state.prediction_service
    try:
        result = service.predict(payload.pixels)
        # Validate the output too, before returning it to the caller.
        return PredictionResponse(**result)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelNotLoadedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        # Log the traceback on the server without exposing internal details.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.") from error
