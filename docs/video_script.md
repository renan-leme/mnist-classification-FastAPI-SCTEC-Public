# Student video script (target: 8-9 minutes; maximum: 10 minutes)

Record the presentation yourself, with your face visible and good lighting.
Use your own words and demonstrate your understanding. Do not use AI-generated
video. This guide is a checklist and timing suggestion, not evidence of a
completed recording.

## 0:00-0:50 - Objective and Option A

- Introduce yourself and the project.
- Explain that the API recognizes one handwritten digit from a 28 x 28 image.
- Explain Option A: the original MNIST project was reused in full.
- Show the original notebook and explain that its source was preserved.

## 0:50-2:00 - Data, training, and model choice

- Explain 70,000 MNIST images, 784 pixels, and ten classes.
- Explain 49,000 training, 7,000 validation, and 14,000 test samples.
- Explain float32 normalization by 255 and why no data-dependent fitting is needed.
- Briefly compare KNN, Random Forest, and MLP.
- Show `models/metadata.json` and `docs/results.md` for actual metrics.
- Explain that the API family is selected by validation before final testing.
- Distinguish this from the original notebook's historical test-based ranking.

## 2:00-3:00 - Task organization and Git

- Show `docs/project_plan.md` and describe the order of work.
- Show `git log --graph --oneline --all` and `git branch -a`.
- Explain the purposes of `develop`, `main`, and each feature branch.
- Explain the concise commits and the preservation of feature branches.

## 3:00-4:10 - Installation and execution

Show these steps from the repository root (use the Windows equivalents in the
README if the virtual environment is not activated):

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

- Explain that the trained artifact is included, so API startup does not train.
- Explain that `python -m scripts.train_model` reproduces the full notebook in
  `build/training/`, keeping generated files away from the root documentation.
- Do not wait for a full retraining during the ten-minute recording.

## 4:10-5:20 - Architecture and lifespan

- Show `schemas.py`: input, output, pixel length, numeric types, and range.
- Show `services.py`: in-memory pipeline and inference.
- Show `routers.py`: health and prediction endpoints.
- Show `main.py`: `@asynccontextmanager`, startup loading, `yield`, and shutdown cleanup.
- Explain why the model is never reloaded from disk for each request.

## 5:20-7:10 - Working demonstration

1. Open `http://127.0.0.1:8000/docs`.
2. Execute `GET /health`; show HTTP 200 and `model_loaded: true`.
3. Execute `POST /predict` with the complete JSON in
   `examples/predict_digit_7.json`; show the predicted digit and probabilities.
4. Send `{"pixels": [0, 255]}`; show HTTP 422 and explain validation.
5. Run `python -m scripts.demo_api` to show a uniform image returning HTTP 400.
6. Explain that unexpected internal failures return a safe HTTP 500 message;
   do not intentionally corrupt or delete the submitted artifact.

## 7:10-8:10 - Tests and limitations

- Run `python -m pytest -q` and show the actual result.
- Explain the test that counts one artifact load across multiple requests.
- Explain that high MNIST accuracy does not guarantee photographic accuracy.
- Mention the original external-image and hidden-class experiments.
- Explain that model confidence is not a guarantee of correctness.
- Suggest a focused future improvement, such as better photographic
  preprocessing or robustness evaluation, without claiming it is implemented.

## 8:10-8:50 - Delivery

- Show the public repository's `main` branch and the README.
- State what you learned and justify the simple architecture.
- Confirm that both the repository and the recorded video links will be sent
  through AVA before the deadline.

## After recording

1. Upload the real recording to Google Drive.
2. Set access to **Anyone with the link / Viewer**.
3. Verify the link without being signed in.
4. Replace the pending-video status in README with the actual link and commit
   that update **before** submitting in AVA.
5. Submit both links and stop modifying the repository until grading.
