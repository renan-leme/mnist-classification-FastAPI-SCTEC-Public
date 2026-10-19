"""Load a trained pipeline once and perform inference without HTTP concerns."""

from pathlib import Path

import joblib
import numpy as np


class InvalidImageError(ValueError):
    """The image is structurally valid but contains no distinguishable digit."""


class ModelNotLoadedError(RuntimeError):
    """The service cannot predict before startup or after shutdown."""


class PredictionService:
    """Own the in-memory pipeline for one application process."""

    def __init__(self):
        self.pipeline = None
        self.metadata = {}

    def load(self, model_path: Path):
        """Called by lifespan only; never call this from a prediction route."""
        if not model_path.is_file():
            raise RuntimeError(
                "Model artifact not found. Run 'python -m scripts.train_model' first."
            )
        # Only load this project's trusted artifact. Joblib is not an upload format.
        artifact = joblib.load(model_path)
        pipeline = artifact["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        if classifier.n_features_in_ != 784 or not np.array_equal(
            classifier.classes_, np.arange(10)
        ):
            raise RuntimeError("The model must accept 784 pixels and recognize digits 0-9.")
        self.pipeline = pipeline
        self.metadata = artifact["metadata"]

    def clear(self):
        """Release references when the application shuts down."""
        self.pipeline = None
        self.metadata = {}

    def predict(self, pixels: list[float]):
        """Use the already-loaded pipeline; there is no disk access here."""
        if self.pipeline is None:
            raise ModelNotLoadedError("The prediction model is unavailable.")
        image = np.asarray(pixels, dtype=np.float32).reshape(1, 784)
        if float(np.ptp(image)) == 0:
            raise InvalidImageError("The image is uniform. Provide an image containing a digit.")

        # Normalization is inside the saved pipeline, so raw input is divided
        # by 255 exactly once. One probability calculation is enough to predict.
        probabilities = self.pipeline.predict_proba(image)[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        return {
            "prediction": int(classes[int(np.argmax(probabilities))]),
            "confidence": float(np.max(probabilities)),
            "probabilities": probabilities.astype(float).tolist(),
            "model_name": self.metadata["model_name"],
        }
