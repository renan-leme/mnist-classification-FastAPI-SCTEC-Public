"""Explicit request and response contracts for the MNIST API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# Strict numeric validation rejects strings and booleans instead of converting
# them silently. Both JSON integers and floats are valid pixel intensities.
Pixel = Annotated[float, Field(strict=True, ge=0, le=255, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """One flattened 28 x 28 grayscale image, before normalization."""

    model_config = ConfigDict(strict=True, extra="forbid")
    pixels: list[Pixel] = Field(
        min_length=784,
        max_length=784,
        description=(
            "Exactly 784 grayscale pixels in row-major order, each from 0 to 255. "
            "Use a dark background and bright digit, as in MNIST. "
            "Copy examples/predict_digit_7.json for a complete valid request."
        ),
    )


class PredictionResponse(BaseModel):
    """A digit and the classifier's probability distribution."""

    model_config = ConfigDict(strict=True, extra="forbid")
    prediction: int = Field(ge=0, le=9, description="Predicted digit.")
    confidence: Probability = Field(description="Maximum class probability; not a guarantee of correctness.")
    probabilities: list[Probability] = Field(
        min_length=10, max_length=10,
        description="Probabilities in digit order: 0, 1, 2, ..., 9.",
    )
    model_name: str = Field(min_length=1, description="Classifier used by the saved pipeline.")


class HealthResponse(BaseModel):
    """Report whether the model is available in this application's memory."""

    model_config = ConfigDict(strict=True, extra="forbid")
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = Field(default=None, min_length=1)


class ErrorResponse(BaseModel):
    """Human-readable errors for failed inference requests."""

    detail: str = Field(min_length=1)
