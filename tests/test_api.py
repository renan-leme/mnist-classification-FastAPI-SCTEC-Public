"""Check the real saved model, strict validation, errors, and lifespan behavior."""

from hashlib import sha256
from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

from app.main import MODEL_PATH, create_app
import app.services as services


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples" / "predict_digit_7.json").read_text())


@pytest.fixture
def client():
    # Entering the context runs startup; leaving it runs shutdown.
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_matches_exported_example(client, payload):
    expected = json.loads((ROOT / "examples" / "expected_prediction.json").read_text())
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == expected["prediction"]
    assert result["model_name"] == expected["model_name"]
    assert result["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-6)
    assert len(result["probabilities"]) == 10
    assert sum(result["probabilities"]) == pytest.approx(1, abs=1e-6)
    assert result["confidence"] == max(result["probabilities"])


@pytest.mark.parametrize("length", [0, 783, 785])
def test_wrong_pixel_count_returns_422(client, length):
    assert client.post("/predict", json={"pixels": [0] * length}).status_code == 422


@pytest.mark.parametrize("bad_pixel", ["12", "wrong", True, None, -1, 256, {}, []])
def test_bad_pixel_is_rejected_before_inference(client, payload, bad_pixel, monkeypatch):
    def unexpected_inference(*args):
        pytest.fail("Invalid data reached the model.")

    monkeypatch.setattr(client.app.state.prediction_service, "predict", unexpected_inference)
    payload["pixels"][0] = bad_pixel
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", "pixels"]


@pytest.mark.parametrize("bad_pixel", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pixels_return_serializable_422(client, payload, bad_pixel):
    payload["pixels"][0] = bad_pixel
    # Python's JSON reader accepts these nonstandard tokens; our API rejects them.
    response = client.post("/predict", content=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_unknown_fields_are_rejected(client, payload):
    payload["unexpected"] = "not part of the contract"
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("body", [{}, [], {"pixels": [[0] * 28] * 28}])
def test_invalid_body_shape_returns_422(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_malformed_json_returns_422(client):
    response = client.post("/predict", content="{",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("pixel", [0, 255])
def test_uniform_image_returns_400(client, pixel):
    response = client.post("/predict", json={"pixels": [pixel] * 784})
    assert response.status_code == 400
    assert "uniform" in response.json()["detail"]


def test_model_is_loaded_once_and_released_on_shutdown(payload, monkeypatch):
    actual_load = services.joblib.load
    loads = []

    def counted_load(path):
        loads.append(path)
        return actual_load(path)

    monkeypatch.setattr(services.joblib, "load", counted_load)
    application = create_app()
    assert loads == []  # Importing/creating the app must not read the artifact.
    with TestClient(application) as test_client:
        for _ in range(3):
            assert test_client.get("/health").status_code == 200
            assert test_client.post("/predict", json=payload).status_code == 200
        assert loads == [MODEL_PATH]
    assert application.state.prediction_service.pipeline is None


def test_unavailable_model_returns_503(client, payload):
    client.app.state.prediction_service.clear()
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert client.post("/predict", json=payload).status_code == 503


def test_unexpected_inference_error_returns_safe_500(client, payload, monkeypatch):
    def failed_prediction(*args):
        raise RuntimeError("private internal error")

    monkeypatch.setattr(client.app.state.prediction_service.pipeline,
                        "predict_proba", failed_prediction)
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed. Please try again."}
    # The process should still answer the next request.
    assert client.get("/health").status_code == 200


def test_invalid_model_output_is_not_returned_to_user(client, payload, monkeypatch):
    monkeypatch.setattr(client.app.state.prediction_service, "predict",
                        lambda pixels: {"prediction": 99})
    assert client.post("/predict", json=payload).status_code == 500


def test_missing_artifact_fails_startup_with_training_instructions(tmp_path):
    with pytest.raises(RuntimeError, match="python -m scripts.train_model"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass


def test_swagger_and_openapi_are_available(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]
    assert "422" in schema["paths"]["/predict"]["post"]["responses"]


def test_original_notebook_matches_training_record():
    notebook = ROOT / "mnist_classification_project.ipynb"
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert sha256(notebook.read_bytes()).hexdigest() == metadata["notebook_sha256"]
