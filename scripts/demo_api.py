"""Demonstrate the running API: python -m scripts.demo_api."""

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import argparse
import json


def call_api(base_url, path, payload=None):
    """Return the status and JSON body, including expected HTTP error responses."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(base_url.rstrip("/") + path, data=body,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    example = json.loads((root / "examples" / "predict_digit_7.json").read_text())
    cases = [
        ("Health", "/health", None, 200),
        ("Valid prediction", "/predict", example, 200),
        ("Wrong number of pixels", "/predict", {"pixels": [0]}, 422),
        ("Uniform image", "/predict", {"pixels": [0] * 784}, 400),
    ]
    for name, path, payload, expected_status in cases:
        status, result = call_api(args.url, path, payload)
        print(f"\n{name}: HTTP {status}")
        print(json.dumps(result, indent=2))
        if status != expected_status:
            raise SystemExit(f"Expected HTTP {expected_status}, received {status}.")


if __name__ == "__main__":
    main()
