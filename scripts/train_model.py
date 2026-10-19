"""Execute the complete original notebook and export a ready-to-use pipeline.

Run from the repository root: python -m scripts.train_model
The original notebook is read, never edited. Its generated files go to build/.
"""

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import shutil

import joblib
import matplotlib
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.preprocessing import normalize_pixels


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "mnist_classification_project.ipynb"
WORK_DIR = ROOT / "build" / "training"


def run_notebook():
    """Run every original code cell, in order, inside an isolated output folder."""
    notebook_bytes = NOTEBOOK_PATH.read_bytes()
    notebook = json.loads(notebook_bytes)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Keep downloads and notebook-generated README files outside the source tree.
    os.environ["SCIKIT_LEARN_DATA"] = str(WORK_DIR / "cache")
    matplotlib.use("Agg")  # Save the original plots without opening GUI windows.
    custom_images = ROOT / "data" / "custom_digits"
    if custom_images.exists():
        shutil.copytree(custom_images, WORK_DIR / "data" / "custom_digits",
                        dirs_exist_ok=True)

    namespace = {"__name__": "__main__"}
    selected_model = None
    validation_scores = {}
    previous_directory = Path.cwd()
    try:
        os.chdir(WORK_DIR)
        with (WORK_DIR / "notebook.log").open("w", encoding="utf-8") as log:
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                print(f"Executing original notebook cell {index}/{len(notebook['cells'])}",
                      flush=True)
                source = "".join(cell["source"])
                with redirect_stdout(log), redirect_stderr(log):
                    # Execute the original source, including EDA, all three models,
                    # OOD analysis, custom-image handling, and documentation cells.
                    exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"),
                         namespace)

                tuning_tables = {
                    "KNN": "knn_tuning_df",
                    "Random Forest": "random_forest_tuning_df",
                    "MLP": "mlp_tuning_df",
                }
                if selected_model is None and all(
                    table in namespace for table in tuning_tables.values()
                ):
                    # Decide before the original final-training/test cells run.
                    # The notebook's own test-based ranking is left intact for
                    # historical comparison; the API only uses validation scores.
                    validation_scores = {
                        name: float(namespace[table].iloc[0]["validation_accuracy"])
                        for name, table in tuning_tables.items()
                    }
                    selected_model = max(validation_scores, key=validation_scores.get)
                    print(f"API model selected using validation only: {selected_model}",
                          flush=True)
    finally:
        os.chdir(previous_directory)

    if NOTEBOOK_PATH.read_bytes() != notebook_bytes:
        raise RuntimeError("The original notebook changed during training.")
    if selected_model is None:
        raise RuntimeError("The notebook did not produce the expected tuning tables.")
    return namespace, selected_model, validation_scores, sha256(notebook_bytes).hexdigest()


def export_model(namespace, selected_model, validation_scores, notebook_hash):
    """Wrap the fitted classifier with its normalization and record real metrics."""
    model = namespace["final_models"][selected_model]
    pipeline = Pipeline([
        ("normalize", FunctionTransformer(normalize_pixels)),
        ("classifier", model),
    ])
    comparison = namespace["model_comparison_df"].to_dict(orient="records")
    metrics = next(row for row in comparison if row["model"] == selected_model)
    metadata = {
        "model_name": selected_model,
        "model_version": "1.0.0",
        "selection_rule": "Highest validation accuracy, chosen before final test evaluation",
        "validation_accuracy": validation_scores,
        "selected_parameters": namespace["selected_parameters"][selected_model],
        "test_metrics": metrics,
        "model_comparison": comparison,
        "original_notebook_best_model": namespace["best_model_name"],
        "notebook_sha256": notebook_hash,
        "dataset": "OpenML mnist_784, version 1",
        "split": {"train": 49000, "validation": 7000, "test": 14000},
        "final_fit": "56,000 development samples; MLP early stopping holds out 10% internally",
        "input": "784 row-major grayscale pixels, each from 0 to 255",
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in
                     ["numpy", "scipy", "scikit-learn", "joblib"]},
        "custom_images_processed": len(namespace["custom_image_paths"]),
    }

    models_dir = ROOT / "models"
    examples_dir = ROOT / "examples"
    models_dir.mkdir(exist_ok=True)
    examples_dir.mkdir(exist_ok=True)
    artifact_path = models_dir / "mnist_pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, artifact_path, compress=3)

    # Check the artifact after reloading it, not just the in-memory classifier.
    restored = joblib.load(artifact_path)["pipeline"]
    np.testing.assert_array_equal(
        restored.predict(namespace["X_test_raw"]),
        namespace["final_predictions"][selected_model],
    )
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Include a real MNIST request so the evaluator can immediately try Swagger.
    sample_index = int(np.flatnonzero(namespace["y_test"] == 7)[0])
    pixels = namespace["X_test_raw"][sample_index].astype(int)
    request = {"pixels": pixels.tolist()}
    probabilities = restored.predict_proba([pixels])[0]
    prediction = int(model.classes_[int(np.argmax(probabilities))])
    expected = {
        "prediction": prediction,
        "confidence": float(probabilities.max()),
        "probabilities": probabilities.tolist(),
        "model_name": selected_model,
    }
    (examples_dir / "predict_digit_7.json").write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )
    (examples_dir / "expected_prediction.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )
    Image.fromarray(pixels.astype(np.uint8).reshape(28, 28)).save(
        examples_dir / "digit_7.png"
    )
    print(f"Saved {artifact_path.relative_to(ROOT)} ({artifact_path.stat().st_size:,} bytes)")
    print(f"Test accuracy: {metrics['accuracy']:.6f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.6f}")
    print("Reloaded pipeline reproduced all 14,000 original test predictions.")
    print(f"Notebook output and log: {WORK_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    export_model(*run_notebook())
