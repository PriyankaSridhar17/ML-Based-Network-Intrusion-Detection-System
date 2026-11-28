
"""
DAY 21 - AUTOMATED FLASK APPLICATION TESTS

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
10. Duplicate original CSV headers
11. Reserved prediction columns (2 cases)
12. CSV export formula protection (5 cases)
13. Export expiration and safe cleanup
14. Expired download
15. Session cookie security
16. Download cache headers

Total pytest cases: 21

The real trained XGBoost model is not modified.
"""

import os
import sys
import tempfile
import time

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ==================================================
# STEP 1 - PROJECT CONFIGURATION
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
# STEP 2 - FLASK TEST CLIENT FIXTURE
# ==================================================


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Create an isolated Flask test client.

    Every test uses a temporary export directory,
    protecting real prediction files.
    """

    monkeypatch.setattr(
        app_module,
        "EXPORT_FOLDER",
        tmp_path
    )

    monkeypatch.setitem(
        app.config,
        "EXPORT_CLEANUP_ENABLED",
        True
    )

    monkeypatch.setitem(
        app.config,
        "TESTING",
        True
    )

    with app.test_client() as test_client:
        yield test_client


# ==================================================
# TEST 1 - HOMEPAGE LOADS
# ==================================================

def test_homepage_loads(client):

    response = client.get("/")

    assert response.status_code == 200


# ==================================================
# TEST 2 - MISSING UPLOADED FILE
# ==================================================

def test_missing_file(client):

    response = client.post(
        "/",
        data={}
    )

    assert response.status_code == 200

    assert (
        b"Please select a CSV file."
        in response.data
    )


# ==================================================
# TEST 3 - UNSUPPORTED FILE EXTENSION
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

    assert (
        b"Only CSV files are supported."
        in response.data
    )


# ==================================================
# TEST 4 - EMPTY CSV
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

    assert (
        b"The uploaded CSV file is empty."
        in response.data
    )


# ==================================================
# TEST 5 - MISSING REQUIRED FEATURES
# ==================================================

def test_missing_required_features(client):

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(
                    b"Feature1,Feature2\n1,2\n"
                ),
                "missing_features.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    assert (
        b"Missing required features:"
        in response.data
    )


# ==================================================
# TEST 6 - NONNUMERIC FEATURES
# ==================================================

def test_nonnumeric_features(client):

    row = {
        name: 0
        for name in app_module.feature_names
    }

    row[
        app_module.feature_names[0]
    ] = "invalid_text"

    csv_content = pd.DataFrame(
        [row]
    ).to_csv(index=False)

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "nonnumeric.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    assert (
        b"Unable to process this CSV file"
        not in response.data
    )

    assert (
        b"invalid_text"
        in response.data
    )


# ==================================================
# TEST 7 - DOWNLOAD WITHOUT PREDICTIONS
# ==================================================

def test_download_without_predictions(client):

    response = client.get("/download")

    assert response.status_code == 404

    assert (
        b"No prediction results available."
        in response.data
    )


# ==================================================
# TEST 8 - OVERSIZED UPLOAD
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

    assert (
        b"File is too large."
        in response.data
    )


# ==================================================
# TEST 9 - SUCCESSFUL PREDICTION AND DOWNLOAD
# ==================================================

def test_successful_prediction_and_download(
    client,
    monkeypatch
):
    """
    Verify successful prediction and CSV download
    without modifying the real XGBoost model.
    """

    class FakeModel:

        def predict(self, features):

            assert len(features) == 2

            return np.array([0, 1])

    monkeypatch.setattr(
        app_module,
        "model",
        FakeModel()
    )

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

        row = {
            name: 0
            for name in app_module.feature_names
        }

        csv_content = pd.DataFrame(
            [row, row]
        ).to_csv(index=False)

        response = client.post(
            "/",
            data={
                "file": (
                    BytesIO(
                        csv_content.encode("utf-8")
                    ),
                    "valid_test.csv"
                )
            },
            content_type="multipart/form-data"
        )

        assert response.status_code == 200

        assert b"BENIGN" in response.data
        assert b"ATTACK" in response.data

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

            assert (
                exported["Predicted_Label"].tolist()
                == ["BENIGN", "ATTACK"]
            )

            assert (
                exported["Predicted_Class"].tolist()
                == [0, 1]
            )

        finally:
            download.close()
            response.close()


# ==================================================
# TEST 10 - DUPLICATE ORIGINAL CSV HEADERS
# ==================================================

def test_duplicate_original_csv_headers(client):

    csv_content = (
        "Flow Duration,Flow Duration,"
        "Average Packet Size\n"
        "100,200,50\n"
    )

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "duplicate_headers.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    assert (
        b"Duplicate column names found."
        in response.data
    )


# ==================================================
# TEST 11 - RESERVED PREDICTION COLUMNS
# ==================================================

@pytest.mark.parametrize(
    "reserved_column",
    [
        "Predicted_Label",
        "Predicted_Class"
    ]
)
def test_reserved_prediction_columns(
    client,
    reserved_column
):

    row = {
        name: 0
        for name in app_module.feature_names
    }

    row[reserved_column] = "existing_value"

    csv_content = pd.DataFrame(
        [row]
    ).to_csv(index=False)

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "reserved_columns.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    assert (
        b"CSV contains reserved prediction columns:"
        in response.data
    )

    assert (
        reserved_column.encode("utf-8")
        in response.data
    )


# ==================================================
# TEST 12 - CSV EXPORT FORMULA PROTECTION
# ==================================================

@pytest.mark.parametrize(
    "input_value,expected_value",
    [
        ("=1+1", "'=1+1"),
        ("+SUM(1,2)", "'+SUM(1,2)"),
        ("@SUM(1,2)", "'@SUM(1,2)"),
        ("  =1+1", "'  =1+1"),
        ("Normal traffic", "Normal traffic"),
    ]
)
def test_csv_export_formula_protection(
    input_value,
    expected_value
):

    data = pd.DataFrame({
        "Notes": [input_value],
        "Flow Duration": [120],
        "Negative Numeric": [-5]
    })

    protected = app_module.protect_csv_export(
        data
    )

    assert (
        protected.loc[0, "Notes"]
        == expected_value
    )

    assert (
        protected.loc[0, "Flow Duration"]
        == 120
    )

    assert (
        protected.loc[0, "Negative Numeric"]
        == -5
    )

    assert (
        data.loc[0, "Notes"]
        == input_value
    )


# ==================================================
# TEST 13 - EXPORT EXPIRATION AND SAFE CLEANUP
# ==================================================

def test_export_expiration_and_cleanup(tmp_path):
    """
    Verify expired generated exports are removed,
    recent exports remain, and unrelated files
    are preserved.
    """

    now = time.time()

    expired = tmp_path / ("a" * 32 + ".csv")
    recent = tmp_path / ("b" * 32 + ".csv")
    unrelated = (
        tmp_path / "intrusion_detection_results.csv"
    )

    for path in (expired, recent, unrelated):
        path.write_text(
            "Flow Duration,Predicted_Label\n"
            "100,BENIGN\n",
            encoding="utf-8"
        )

    old_time = now - 25 * 60 * 60

    os.utime(expired, (old_time, old_time))
    os.utime(unrelated, (old_time, old_time))

    assert app_module.is_export_expired(
        expired,
        now=now
    )

    assert not app_module.is_export_expired(
        recent,
        now=now
    )

    preview = app_module.cleanup_expired_exports(
        folder=tmp_path,
        now=now,
        dry_run=True
    )

    assert preview["expired_found"] == 1
    assert preview["deleted"] == 0
    assert expired.exists()

    result = app_module.cleanup_expired_exports(
        folder=tmp_path,
        now=now
    )

    assert result["deleted"] == 1

    assert not expired.exists()
    assert recent.exists()
    assert unrelated.exists()


# ==================================================
# TEST 14 - EXPIRED DOWNLOAD
# ==================================================

def test_expired_download(
    client,
    monkeypatch,
    tmp_path
):
    """
    Verify expired prediction downloads return 404.
    """

    monkeypatch.setattr(
        app_module,
        "EXPORT_FOLDER",
        tmp_path
    )

    export_id = "c" * 32

    export_path = (
        tmp_path / f"{export_id}.csv"
    )

    export_path.write_text(
        "Predicted_Label\nATTACK\n",
        encoding="utf-8"
    )

    old_time = time.time() - 25 * 60 * 60

    os.utime(
        export_path,
        (old_time, old_time)
    )

    with client.session_transaction() as sess:
        sess["export_id"] = export_id

    response = client.get("/download")

    assert response.status_code == 404

    assert (
        b"expired"
        in response.data.lower()
    )

    response.close()


# ==================================================
# TEST 15 - SESSION COOKIE SETTINGS
# ==================================================

def test_session_cookie_security():
    """
    Verify local demonstration cookie settings.
    """

    assert (
        app.config["SESSION_COOKIE_HTTPONLY"]
        is True
    )

    assert (
        app.config["SESSION_COOKIE_SAMESITE"]
        == "Lax"
    )

    assert (
        app.config["SESSION_COOKIE_SECURE"]
        is False
    )


# ==================================================
# TEST 16 - DOWNLOAD CACHE HEADERS
# ==================================================

def test_download_cache_headers(
    client,
    monkeypatch,
    tmp_path
):
    """
    Verify successful prediction downloads
    include no-cache response headers.
    """

    monkeypatch.setattr(
        app_module,
        "EXPORT_FOLDER",
        tmp_path
    )

    export_id = "d" * 32

    export_path = (
        tmp_path / f"{export_id}.csv"
    )

    export_path.write_text(
        "Predicted_Label\nBENIGN\n",
        encoding="utf-8"
    )

    with client.session_transaction() as sess:
        sess["export_id"] = export_id

    response = client.get(
        "/download",
        buffered=True
    )

    try:

        assert response.status_code == 200

        assert (
            "no-store"
            in response.headers["Cache-Control"]
        )

        assert (
            response.headers["Pragma"]
            == "no-cache"
        )

        assert (
            response.headers["X-Content-Type-Options"]
            == "nosniff"
        )

    finally:
        response.close()


# ==================================================
# TEST 17 - END-TO-END PROTECTED CSV DOWNLOAD
# ==================================================

def test_protected_csv_download_end_to_end(
    client, monkeypatch, tmp_path
):
    """
    Verify formula-like text remains protected
    in the actual downloaded prediction CSV.
    """

    class FakeModel:
        def predict(self, features):
            assert len(features) == 1
            assert len(features.columns) == 78
            return np.array([1])

    monkeypatch.setattr(
        app_module, "model", FakeModel()
    )

    monkeypatch.setattr(
        app_module, "EXPORT_FOLDER", tmp_path
    )

    row = {
        name: 0
        for name in app_module.feature_names
    }

    row["Notes"] = "=1+1"

    csv_content = pd.DataFrame(
        [row]
    ).to_csv(index=False)

    upload = client.post(
        "/",
        data={
            "file": (
                BytesIO(csv_content.encode("utf-8")),
                "formula_test.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert upload.status_code == 200
    assert b"ATTACK" in upload.data

    download = client.get(
        "/download", buffered=True
    )

    try:
        assert download.status_code == 200

        exported = pd.read_csv(
            BytesIO(download.data)
        )

        assert exported.loc[0, "Notes"] == "'=1+1"
        assert exported.loc[0, "Predicted_Label"] == "ATTACK"
        assert exported.loc[0, "Predicted_Class"] == 1

        # Confirm the actual CSV bytes contain
        # the protected text.
        assert b"'=1+1" in download.data

    finally:
        download.close()
        upload.close()
