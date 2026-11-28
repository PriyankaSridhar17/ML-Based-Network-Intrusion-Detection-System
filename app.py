
"""
ML-Based Network Intrusion Detection System
Flask Dashboard - Day 21 Security Improvements

Author: Priyanka Sridhar

Features:
- CSV upload and validation
- XGBoost network intrusion predictions
- BENIGN and ATTACK statistics
- Downloadable prediction results
- Duplicate header protection
- Reserved prediction column protection
- CSV formula-injection mitigation
- 24-hour export expiration
- Safe export cleanup
- Session cookie security
- Download cache protection

Local academic demonstration application.
Not intended for public production deployment.
"""

import csv
import os
import re
import secrets
import time

from io import BytesIO, StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
)


# ==================================================
# STEP 1 - PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.joblib"

EXPORT_FOLDER = BASE_DIR / "exports"
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Never hardcode a real application secret.
app.secret_key = os.environ["FLASK_SECRET_KEY"]

app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",

    # False for local HTTP demonstration.
    # Set True when deploying behind HTTPS.
    SESSION_COOKIE_SECURE=False,

    # Keep cleanup disabled until regression
    # tests have been completed.
    EXPORT_CLEANUP_ENABLED=True,
)

MAX_EXPORT_ROWS = 50000

EXPORT_RETENTION_SECONDS = 24 * 60 * 60

RESERVED_COLUMNS = {
    "Predicted_Label",
    "Predicted_Class",
}

# Application-generated export IDs contain
# exactly 32 lowercase hexadecimal characters.
EXPORT_FILENAME_PATTERN = re.compile(
    r"^[0-9a-f]{32}\.csv$"
)


# ==================================================
# STEP 2 - LOAD TRUSTED TRAINED MODEL
# ==================================================

print("Loading trained XGBoost model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

# Only load trusted local joblib files.
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_names = list(saved_model["features"])

print("XGBoost model loaded successfully!")
print("Number of required features:", len(feature_names))


# ==================================================
# STEP 3 - VALIDATE ORIGINAL CSV HEADERS
# ==================================================

def validate_csv_headers(content):
    """
    Validate original CSV headers before pandas
    automatically renames duplicate names.
    """

    if not content:
        raise ValueError(
            "The uploaded CSV file is empty."
        )

    try:
        decoded = content.decode("utf-8-sig")

        reader = csv.reader(
            StringIO(decoded),
            strict=True
        )

        original_headers = next(reader, None)

    except UnicodeDecodeError as exc:
        raise ValueError(
            "CSV file must use UTF-8 encoding."
        ) from exc

    except csv.Error as exc:
        raise ValueError(
            "Invalid CSV header format."
        ) from exc

    if not original_headers:
        raise ValueError(
            "The uploaded CSV file is empty."
        )

    normalized_headers = [
        header.strip()
        for header in original_headers
    ]

    if any(not header for header in normalized_headers):
        raise ValueError(
            "CSV contains empty column names."
        )

    if len(normalized_headers) != len(
        set(normalized_headers)
    ):
        raise ValueError(
            "Duplicate column names found."
        )

    return normalized_headers


# ==================================================
# STEP 4 - PROTECT EXPORTED CSV TEXT
# ==================================================

def protect_csv_export(data):
    """
    Reduce spreadsheet formula-injection risks.

    Operates on a copy of the data.
    Does not modify the model's input features.

    This is a mitigation, not a guarantee of
    safety in every spreadsheet application.
    """

    safe_data = data.copy()

    dangerous_prefixes = (
        "=",
        "+",
        "-",
        "@",
    )

    def protect_value(value):
        if not isinstance(value, str):
            return value

        normalized = value.lstrip(" \t\r\n")

        if normalized.startswith(dangerous_prefixes):
            return "'" + value

        return value

    for column in safe_data.columns:
        if (
            pd.api.types.is_object_dtype(
                safe_data[column]
            )
            or pd.api.types.is_string_dtype(
                safe_data[column]
            )
        ):
            safe_data[column] = (
                safe_data[column].map(protect_value)
            )

    return safe_data


# ==================================================
# STEP 5 - EXPORT FILE VALIDATION
# ==================================================

def is_generated_export(path):
    """
    Return True only for regular files whose
    names match the application-generated format.

    Symlinks are excluded.
    """

    path = Path(path)

    return (
        EXPORT_FILENAME_PATTERN.fullmatch(path.name)
        is not None
        and path.is_file()
        and not path.is_symlink()
    )


def is_export_expired(path, now=None):
    """
    Determine whether a generated export has
    exceeded the 24-hour retention period.
    """

    path = Path(path)

    if not is_generated_export(path):
        return False

    if now is None:
        now = time.time()

    try:
        modified_time = path.stat().st_mtime

    except OSError:
        app.logger.warning(
            "Unable to inspect export modification time"
        )
        return False

    return (
        now - modified_time
        >= EXPORT_RETENTION_SECONDS
    )


# ==================================================
# STEP 6 - SAFE EXPORT CLEANUP
# ==================================================

def cleanup_expired_exports(
    folder=None,
    now=None,
    dry_run=False
):
    """
    Remove expired application-generated CSV files.

    Safeguards:
    - Only filenames matching 32 hex characters.
    - Ignore unrelated files.
    - Ignore symbolic links.
    - Ignore files younger than 24 hours.
    - Handle Windows file-locking errors.

    Returns a summary of cleanup activity.
    """

    if folder is None:
        folder = EXPORT_FOLDER

    folder = Path(folder)

    if now is None:
        now = time.time()

    summary = {
        "expired_found": 0,
        "deleted": 0,
        "errors": 0,
    }

    if not folder.is_dir():
        return summary

    for path in folder.iterdir():

        if not is_generated_export(path):
            continue

        if not is_export_expired(path, now=now):
            continue

        summary["expired_found"] += 1

        if dry_run:
            continue

        try:
            path.unlink()
            summary["deleted"] += 1

        except OSError:
            summary["errors"] += 1

            app.logger.warning(
                "Unable to delete expired export: %s",
                path.name
            )

    return summary


def maybe_cleanup_exports():
    """
    Run cleanup only when explicitly enabled.

    During initial Day 21 testing, this remains
    disabled to preserve existing local files.
    """

    if not app.config.get(
        "EXPORT_CLEANUP_ENABLED",
        False
    ):
        return

    try:
        cleanup_expired_exports()

    except OSError:
        app.logger.exception(
            "Export cleanup failed"
        )


# ==================================================
# STEP 7 - DASHBOARD HOMEPAGE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a CSV file."

        elif not uploaded_file.filename.lower().endswith(
            ".csv"
        ):
            error = "Only CSV files are supported."

        else:

            try:

                # ----------------------------------
                # Read and validate uploaded CSV
                # ----------------------------------

                content = uploaded_file.read()

                validate_csv_headers(content)

                data = pd.read_csv(BytesIO(content))

                if data.empty:
                    raise ValueError(
                        "The uploaded CSV file is empty."
                    )

                data.columns = data.columns.str.strip()

                if data.columns.duplicated().any():
                    raise ValueError(
                        "Duplicate column names found."
                    )

                # ----------------------------------
                # Reject reserved prediction columns
                # ----------------------------------

                conflicting_columns = (
                    RESERVED_COLUMNS.intersection(
                        data.columns
                    )
                )

                if conflicting_columns:
                    raise ValueError(
                        "CSV contains reserved prediction columns: "
                        + ", ".join(
                            sorted(conflicting_columns)
                        )
                    )

                # ----------------------------------
                # Validate row count
                # ----------------------------------

                if len(data) > MAX_EXPORT_ROWS:
                    raise ValueError(
                        "Maximum 50,000 rows are supported "
                        "per upload."
                    )

                print("\nCSV uploaded successfully!")
                print("Total network flows:", len(data))

                # ----------------------------------
                # Validate required model features
                # ----------------------------------

                missing_features = [
                    name
                    for name in feature_names
                    if name not in data.columns
                ]

                if missing_features:
                    raise ValueError(
                        "Missing required features: "
                        + ", ".join(
                            missing_features[:10]
                        )
                    )

                features = data.loc[
                    :, feature_names
                ].copy()

                for column in feature_names:
                    features[column] = pd.to_numeric(
                        features[column],
                        errors="raise"
                    )

                if not np.isfinite(
                    features.to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        "CSV contains missing or infinite "
                        "values in required features."
                    )

                # ----------------------------------
                # Predict network traffic
                # ----------------------------------

                print("Analyzing network traffic...")

                predictions = np.asarray(
                    model.predict(features)
                ).astype(int)

                if (
                    len(predictions) != len(data)
                    or not np.isin(
                        predictions, [0, 1]
                    ).all()
                ):
                    raise ValueError(
                        "Unexpected model prediction labels."
                    )

                # ----------------------------------
                # Calculate dashboard statistics
                # ----------------------------------

                total_count = len(predictions)

                benign_count = int(
                    np.sum(predictions == 0)
                )

                attack_count = int(
                    np.sum(predictions == 1)
                )

                benign_percentage = round(
                    benign_count / total_count * 100,
                    2
                )

                attack_percentage = round(
                    attack_count / total_count * 100,
                    2
                )

                results = {
                    "total": total_count,
                    "benign": benign_count,
                    "attacks": attack_count,
                    "benign_percent": benign_percentage,
                    "attack_percent": attack_percentage,
                    "chart_labels": [
                        "BENIGN", "ATTACK"
                    ],
                    "chart_values": [
                        benign_count, attack_count
                    ],
                    "attack_detected": attack_count > 0,
                }

                # ----------------------------------
                # Prepare protected prediction export
                # ----------------------------------

                export_data = protect_csv_export(data)

                export_data["Predicted_Label"] = np.where(
                    predictions == 1,
                    "ATTACK",
                    "BENIGN"
                )

                export_data["Predicted_Class"] = (
                    predictions
                )

                export_id = secrets.token_hex(16)

                export_path = (
                    EXPORT_FOLDER / f"{export_id}.csv"
                )

                export_data.to_csv(
                    export_path,
                    index=False
                )

                session["export_id"] = export_id

                results["download_available"] = True

                # Cleanup is disabled until tests
                # confirm that it behaves safely.
                maybe_cleanup_exports()

                print(
                    "Prediction CSV saved successfully!"
                )

                print("\nINTRUSION DETECTION RESULTS")
                print("----------------------------")
                print("Total flows:", total_count)
                print("BENIGN flows:", benign_count)
                print("ATTACK flows:", attack_count)
                print(
                    "BENIGN percentage:",
                    benign_percentage
                )
                print(
                    "ATTACK percentage:",
                    attack_percentage
                )
                print(
                    "\nNetwork traffic analysis completed!"
                )

            except (
                ValueError,
                TypeError,
                pd.errors.ParserError,
                UnicodeError
            ) as exc:

                error = str(exc)

            except Exception:

                app.logger.exception(
                    "Prediction processing failed"
                )

                error = (
                    "Unable to process this CSV file. "
                    "Please check its format."
                )

    return render_template(
        "index.html",
        results=results,
        error=error
    )


# ==================================================
# STEP 8 - DOWNLOAD PREDICTION RESULTS
# ==================================================

@app.route("/download")
def download_results():

    export_id = session.get("export_id")

    if (
        not isinstance(export_id, str)
        or re.fullmatch(
            r"[0-9a-f]{32}",
            export_id
        ) is None
    ):
        return (
            "No prediction results available.",
            404
        )

    export_path = (
        EXPORT_FOLDER / f"{export_id}.csv"
    )

    if not is_generated_export(export_path):
        return (
            "No prediction results available.",
            404
        )

    # Reject expired downloads even while
    # automatic cleanup is disabled.
    if is_export_expired(export_path):

        session.pop("export_id", None)

        maybe_cleanup_exports()

        return (
            "Prediction results have expired.",
            404
        )

    response = send_file(
        export_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=(
            "intrusion_detection_results.csv"
        ),
        conditional=False
    )

    # Prevent browser and intermediary caching.
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, private"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"

    maybe_cleanup_exports()

    return response


# ==================================================
# STEP 9 - OVERSIZED UPLOAD HANDLING
# ==================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        results=None,
        error=(
            "File is too large. "
            "Maximum size is 10 MB."
        )
    ), 413


# ==================================================
# STEP 10 - START LOCAL FLASK DASHBOARD
# ==================================================

if __name__ == "__main__":

    print(
        "\nStarting Network Intrusion "
        "Detection Dashboard..."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
