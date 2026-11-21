
"""
Day 14 - Automated Flask Application Tests

ML-Based Network Intrusion Detection System

Tests:
1. Homepage loads
2. Missing uploaded file
3. Unsupported file extension
4. Empty CSV
5. Missing required features
6. Nonnumeric features
7. Download without predictions
8. Oversized upload
9. Successful prediction and CSV download
"""

import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ==================================================
# STEP 1: Project configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "FLASK_SECRET_KEY",
    "local-automated-testing-only-key"
)

import app as app_module

app = app_module.app


# ==================================================
# STEP 2: Flask test client fixture
# ==================================================

@pytest.fixture
def client():
    """Create a Flask test client."""

    app.config.update(TESTING=True)

    with app.test_client() as test_client:
        yield test_client


# ==================================================
# TEST 1: Homepage loads
# ==================================================

def test_homepage_loads(client):
    response = client.get("/")

    assert response.status_code == 200


# ==================================================
# TEST 2: Missing uploaded file
# ==================================================

def test_missing_file(client):
    response = client.post("/", data={})

    assert response.status_code == 200
    assert b"Please select a CSV file." in response.data


# ==================================================
# TEST 3: Unsupported file extension
# ==================================================

def test_unsupported_file(client):
    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"invalid file content"),
                "test.txt"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Only CSV files are supported." in response.data


# ==================================================
# TEST 4: Empty CSV
# ==================================================

def test_empty_csv(client):
    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"Feature1,Feature2\n"),
                "empty.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"The uploaded CSV file is empty." in response.data


# ==================================================
# TEST 5: Missing required features
# ==================================================

def test_missing_required_features(client):
    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"Feature1,Feature2\n1,2\n"),
                "missing_features.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Missing required features:" in response.data


# ==================================================
# TEST 6: Nonnumeric features
# ==================================================

def test_nonnumeric_features(client):
    row = {
        name: 0
        for name in app_module.feature_names
    }

    row[app_module.feature_names[0]] = "invalid_text"

    csv_content = pd.DataFrame(
        [row]
    ).to_csv(index=False)

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(csv_content.encode("utf-8")),
                "nonnumeric.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Unable to process this CSV file" not in response.data
    assert b"invalid_text" in response.data


# ==================================================
# TEST 7: Download without predictions
# ==================================================

def test_download_without_predictions(client):
    response = client.get("/download")

    assert response.status_code == 404
    assert b"No prediction results available." in response.data


# ==================================================
# TEST 8: Oversized upload
# ==================================================

def test_oversized_upload(client):
    oversized_content = b"x" * (
        10 * 1024 * 1024 + 1
    )

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(oversized_content),
                "oversized.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 413
    assert b"File is too large." in response.data


# ==================================================
# TEST 9: Successful prediction and CSV download
# ==================================================

def test_successful_prediction_and_download(
    client,
    monkeypatch
):
    """
    Verify successful predictions and CSV download
    without changing the actual trained XGBoost model.
    """

    class FakeModel:
        def predict(self, features):
            assert len(features) == 2
            return np.array([0, 1])

    # Temporarily replace the model during this test.
    monkeypatch.setattr(
        app_module,
        "model",
        FakeModel()
    )

    # Use a dedicated temporary export directory.
    # On Windows, cleanup may encounter locked files.
    with tempfile.TemporaryDirectory(
        prefix="ids_test_",
        dir=PROJECT_ROOT,
        ignore_cleanup_errors=True
    ) as temp_dir:

        monkeypatch.setattr(
            app_module,
            "EXPORT_FOLDER",
            Path(temp_dir)
        )

        # Generate two network traffic records.
        row = {
            name: 0
            for name in app_module.feature_names
        }

        csv_content = pd.DataFrame(
            [row, row]
        ).to_csv(index=False)

        # Upload CSV for prediction.
        response = client.post(
            "/",
            data={
                "file": (
                    BytesIO(csv_content.encode("utf-8")),
                    "valid_test.csv"
                )
            },
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        assert b"BENIGN" in response.data
        assert b"ATTACK" in response.data

        # Download and buffer the prediction CSV.
        download = client.get(
            "/download",
            buffered=True
        )

        try:
            assert download.status_code == 200
            assert download.mimetype == "text/csv"

            exported = pd.read_csv(
                BytesIO(download.data)
            )

            assert len(exported) == 2

            assert exported["Predicted_Label"].tolist() == [
                "BENIGN",
                "ATTACK"
            ]

            assert exported["Predicted_Class"].tolist() == [
                0,
                1
            ]

        finally:
            # Release response resources before cleanup.
            download.close()
            response.close()
