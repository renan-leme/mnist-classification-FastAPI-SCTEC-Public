# Project plan and requirement mapping

## Scope

Option A: preserve the complete previous MNIST notebook and add the minimum
software needed to serve one trained classifier through FastAPI. No frontend,
database, authentication system, container platform, or unrelated feature is
required for this implementation.

## Task order and branches

| Order | Branch | Responsibility |
| --- | --- | --- |
| 1 | `feature/project-setup` | Dependencies, ignore rules, remove local environments from tracking |
| 2 | `feature/model-export` | Execute the unchanged notebook and export model, preprocessing, and metrics |
| 3 | `feature/fastapi` | Schemas, services, routers, lifespan, and error responses |
| 4 | `feature/api-tests` | Real-model integration tests and HTTP demonstration |
| 5 | `feature/documentation` | English README, results, video script, and submission checklist |

`develop` integrates the completed steps. `main` receives the final working
version. Keep the feature branches so the evaluator can inspect each stage.
Use concise imperative commit messages describing the actual changes.

## Assignment requirements

| Requirement | Implementation or evidence |
| --- | --- |
| Reuse an earlier model, Option A | Original MNIST notebook retained in full |
| Training, validation, and test | Original stratified 70/10/20 split, recorded in metadata |
| Appropriate preprocessing | Original float32 normalization, included in the persisted pipeline |
| Serialized trained artifact | `models/mnist_pipeline.joblib` |
| Documented metrics | `models/metadata.json`, `docs/results.md` |
| Modular structure | `app/main.py`, `routers.py`, `schemas.py`, `services.py` |
| Load once using lifespan | `@asynccontextmanager` in `app/main.py` |
| No model reload per request | Routes use `app.state.prediction_service`; load-count integration test |
| Pydantic input and output | `PredictionRequest`, `PredictionResponse`, and `HealthResponse` |
| Field restrictions and HTTP 422 | Pixel type/range/length restrictions, finite values, forbidden extra fields |
| GET /health | Reports actual in-memory availability |
| POST /predict | Returns one digit and ten class probabilities |
| HTTP error handling | Tested 400, 422, 500, and 503 paths |
| Reproducibility | Versioned requirements, Python version, notebook hash, fixed seed |
| Relative project paths | Paths resolved from source files; no machine-specific paths in source |
| Swagger examples | `/docs`, complete request JSON and expected response in `examples/` |
| ASGI execution/deployment | Uvicorn commands in README; no public hosting URL is claimed |
| GitHub workflow | Integration and feature branches retained; final code on main |
| README | Purpose, origin, technologies, installation, API, limitations, and evidence status |
| Student presentation | Prepared script; student must record and upload the video |
| AVA submission | Student must submit both real links and freeze repository updates |

## Technical choices to explain

- A single numeric-image endpoint keeps the required JSON contract explicit.
- The complete original notebook is used for training and analysis rather than
  replacing it with a new model implementation.
- API family selection uses validation accuracy before final testing. The
  historical notebook's test-based ranking is preserved but not used for the
  API selection.
- Persisting preprocessing with the classifier avoids differences between
  training and serving and avoids applying normalization twice.
- The service owns model operations; routes only translate HTTP requests and
  errors. Loading happens once per process during lifespan startup.
- Pydantic rejects invalid inputs before they reach the classifier.
- Unexpected errors return a safe message while server logs retain the traceback.

## Completion limits

The code, artifact, documentation, and checks can be prepared in the repository.
The student must personally record the required video, add its real Google Drive
link, verify public access, submit the links in AVA, and follow the assignment's
authorship/AI rules. A grade cannot be guaranteed by automated checks.
