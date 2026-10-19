# MNIST Classification with FastAPI

This project turns a trained handwritten-digit classifier into a small REST API.
An external application sends one 28 x 28 grayscale digit as 784 pixel values
and receives the predicted digit, class probabilities, and model name.

This is **Option A** of the SCTEC assignment: reuse and improve a model from a
previous course project. The complete original
[`mnist_classification_project.ipynb`](mnist_classification_project.ipynb) is
preserved byte for byte. The application adds model persistence, a modular API,
strict validation, lifecycle management, and documentation.

## Original model and dataset

The notebook uses [OpenML MNIST, version 1](https://www.openml.org/d/554): 70,000
grayscale images of handwritten digits 0-9, with 28 x 28 pixels per image.
Pixels are divided by 255 and converted to `float32`.

The stratified split uses `random_state=42`:

| Subset | Images | Purpose |
| --- | ---: | --- |
| Training | 49,000 | Fit candidate models |
| Validation | 7,000 | Select hyperparameters and the API model family |
| Test | 14,000 | Report final performance |

The notebook compares KNN, Random Forest, and MLP. After tuning, the final models
are refitted on the 56,000 development samples. MLP also reserves an internal
10% validation fraction for early stopping.

The original notebook ranks its final models using test metrics. To preserve
that historical project without using the test set to choose the API model,
the export script selects the family by **validation accuracy before the
notebook's final training and test cells execute**. Both rankings are recorded
in [`models/metadata.json`](models/metadata.json).

The saved artifact includes the fitted classifier and normalization in one
Scikit-learn pipeline. It is included in Git, so running the API does not require
downloading MNIST or retraining.

## Technologies

Validated with **Python 3.11.7**. Use Python 3.11 to reproduce this environment.
All direct dependencies are pinned in [`requirements.txt`](requirements.txt).

| Purpose | Libraries and versions |
| --- | --- |
| API | FastAPI 0.147.0, Pydantic 2.14.1, Uvicorn 0.54.0 |
| Machine learning | Scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17.1, Joblib 1.6.0 |
| Notebook and analysis | Pandas 2.3.3, Matplotlib 3.10.8, Jupyter 1.1.1, IPython 9.17.1, Tabulate 0.10.0 |
| Images | Pillow 12.3.0, OpenCV 4.13.0.92 |
| Tests | Pytest 9.1.1, HTTPX 0.28.1 |

## Install and run

Clone the public repository:

```bash
git clone https://github.com/renan-leme/mnist-classification-FastAPI-SCTEC-Public.git
cd mnist-classification-FastAPI-SCTEC-Public
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

On Linux or macOS, with Python 3.11 installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Run commands from the repository root. Directly using the virtual environment's
Python on Windows avoids PowerShell activation-policy problems.

Open [Swagger UI](http://127.0.0.1:8000/docs) to try the API.
The [OpenAPI schema](http://127.0.0.1:8000/openapi.json) is also available.
Stop the server with `Ctrl+C`.

For a regular ASGI deployment without the development reloader:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This starts a service on the machine where the command runs. No public cloud
URL is claimed. Each Uvicorn worker loads its own copy of the model once.

## API usage

### GET /health

Returns HTTP 200 when the service and its in-memory model are available:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "MLP"
}
```

If the in-memory model is unavailable, this endpoint returns HTTP 503. If the
artifact is missing at startup, the server fails with a message explaining how
to train it; it does not pretend to be ready.

### POST /predict

The body contains exactly one field, `pixels`: a flat list of **784 numeric
values between 0 and 255**, in row-major order. The digit must be bright against
a dark background. Normalization is applied inside the pipeline; do not divide
the example pixels by 255 before submitting them.

The complete, valid JSON payload is
[`examples/predict_digit_7.json`](examples/predict_digit_7.json). Its source image
is [`examples/digit_7.png`](examples/digit_7.png), a real MNIST test example.

On PowerShell:

```powershell
$body = Get-Content -Raw examples/predict_digit_7.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body
```

With curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/predict_digit_7.json
```

For Swagger, expand `POST /predict`, click **Try it out**, replace the generated
body with the complete example JSON, and click **Execute**.

The response contains `prediction` (0-9), `confidence` (0-1), ten
`probabilities` in digit order 0-9, and `model_name`. The full response for the
included example is in
[`examples/expected_prediction.json`](examples/expected_prediction.json).
Probabilities describe the classifier's output; they do not guarantee correctness.

### Validation and errors

| Status | Meaning | Example |
| --- | --- | --- |
| 200 | Successful health check or prediction | Complete MNIST example |
| 400 | Structurally valid image with no distinguishable digit | 784 identical pixels |
| 422 | Invalid request, rejected before inference | Wrong length, a string, boolean, null, out-of-range or non-finite pixel, or an extra field |
| 500 | Unexpected inference/output error, safely reported | Internal model failure |
| 503 | Model unavailable in memory | Service is not ready |

Minimal invalid request, which returns HTTP 422:

```json
{"pixels": [0, 255]}
```

Expected 400/500/503 responses have a `detail` string. A 422 response has a
`detail` list describing the invalid fields. Internal tracebacks are logged on
the server and are not sent to API clients.

## Reproduce training

After installing the requirements, run:

```bash
python -m scripts.train_model
```

On Windows, use `.\.venv\Scripts\python.exe -m scripts.train_model` if the
environment is not activated.

This command executes **every original code cell in order**, including EDA,
the tuning/training of all three models, metrics, confusion matrices, OOD
analysis, external-image handling, and documentation generation. It uses the
original cell source, without editing the notebook. This requires an Internet
connection on the first run and more time/memory than serving the API.

The notebook runs with `build/training/` as its working directory. Its generated
README, requirements, figures, and notes stay there, protecting this API's
source documentation. The download cache is also kept there. Logs are saved to
`build/training/notebook.log`.

The script then writes:

- `models/mnist_pipeline.joblib`: normalization and the fitted classifier;
- `models/metadata.json`: actual validation/test metrics, parameters, versions,
  split sizes, and the original notebook's SHA-256;
- `examples/`: a complete request, source image, and expected response.

Before reporting success, it reloads the saved artifact and checks that its
predictions match the original classifier on **all 14,000 test images**.

To repeat the notebook's external-photograph experiment, place your own images
in [`data/custom_digits/`](data/custom_digits/README.md) before training.
The historical notebook contains saved outputs for nine photographs; the
original photograph files were not supplied. The new execution reports how
many are actually present, without fabricating photographs or results.

Opening the original notebook and running its cells manually from the root can
overwrite the root README and requirements, because that is its original
behavior. Prefer the training command above.

## Project structure

```text
app/
  main.py                 # Application, lifespan, and validation error handler
  routers.py              # GET /health and POST /predict
  schemas.py              # Pydantic input/output contracts
  services.py             # Model loading and inference; no HTTP dependencies
  preprocessing.py        # The notebook's float32 / 255 normalization
scripts/
  train_model.py          # Execute the complete notebook and export the pipeline
  demo_api.py             # Demonstrate a running server
models/
  mnist_pipeline.joblib    # Included trained artifact
  metadata.json           # Reproducible metrics and source hash
examples/                 # Complete request, expected response, and source image
tests/test_api.py         # Integration and lifecycle tests
docs/                     # Results, project plan, and video/submission guides
data/custom_digits/       # Optional original handwritten photographs
mnist_classification_project.ipynb
projeto.pdf
requirements.txt
```

## Verification

```bash
python -m pytest -q
python -m pip check
```

The tests use the real artifact and verify valid predictions, output contracts,
strict 422 validation, uniform-image 400 errors, safe 500 errors, unavailable
model responses, Swagger, missing-artifact startup, notebook preservation,
and exactly one model load across multiple requests.

With the server running in a separate terminal, demonstrate success and errors:

```bash
python -m scripts.demo_api
```

## Git workflow and project planning

The implementation uses `develop` as the integration branch and separate
feature branches for setup, model export, API, tests, and documentation. Feature
branches are retained after integration, as requested. The final version is
merged into `main`. See [`docs/project_plan.md`](docs/project_plan.md) for the
task order, branch purposes, and requirement mapping.

## Results and limitations

See [`docs/results.md`](docs/results.md) and
[`models/metadata.json`](models/metadata.json) for the executed results.
The shipped MLP achieved **97.3071% test accuracy** and **0.973047 weighted F1**.

MNIST is a clean benchmark. Photographs, unusual handwriting, rotations,
multiple digits, and non-digit images can behave differently. The original
notebook's nine external-image outputs contained four correct classifications,
based on the file labels; this small historical sample is not a new evaluation.
The API validates the pixel contract and rejects uniform images; it does not
detect every non-digit or out-of-distribution image. This is an educational
single-digit classifier, not a general OCR service.

## Video and submission evidence

**Video status: pending student recording and Google Drive link.**

Follow [`docs/video_script.md`](docs/video_script.md) for a presentation of less
than ten minutes. Record yourself with your face visible and demonstrate your
own understanding. The assignment prohibits AI-generated presentation videos.
After uploading, set Google Drive access to **Anyone with the link / Viewer**
and add the real link here before submission.

Public repository:
[mnist-classification-FastAPI-SCTEC-Public](https://github.com/renan-leme/mnist-classification-FastAPI-SCTEC-Public).

Submit **both the GitHub and video links** in the AVA task by
**3 November 2026 at 22:00**, using the deadline's timezone as confirmed by the
course. The PDF does not specify a timezone. Follow
[`docs/submission_checklist.md`](docs/submission_checklist.md), and do not change
the repository after submission until the grade is issued.

The assignment also restricts AI assistance and assesses the student's
understanding and authorship. Review its rules in `projeto.pdf` and make sure
you can explain and take responsibility for every submitted part.

## Technical references

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/): load shared resources at startup.
- [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/): reject unwanted type conversions.
- [Scikit-learn model persistence](https://scikit-learn.org/1.8/model_persistence.html): save models and use matching environments.

## Author

Renan de Brito Leme
