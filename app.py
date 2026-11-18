
import os
import secrets
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session
)

# --------------------------------------------------
# STEP 1: Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.joblib"

EXPORT_FOLDER = BASE_DIR / "exports"
EXPORT_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)

# Read the Flask secret key from an environment variable.
# Do not hardcode passwords or secret keys in source code.
app.secret_key = os.environ["FLASK_SECRET_KEY"]

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Maximum number of prediction rows per upload
MAX_EXPORT_ROWS = 50000


# --------------------------------------------------
# STEP 2: Load trained XGBoost model
# --------------------------------------------------

print("Loading trained XGBoost model...")

# Only load trusted model files.
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_names = list(saved_model["features"])

print("XGBoost model loaded successfully!")
print("Number of required features:", len(feature_names))


# --------------------------------------------------
# STEP 3: Dashboard homepage
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    results = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:

            error = "Please select a CSV file."

        elif not uploaded_file.filename.lower().endswith(".csv"):

            error = "Only CSV files are supported."

        else:

            try:

                # ----------------------------------
                # STEP 4: Read CSV file
                # ----------------------------------

                content = uploaded_file.read()

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

                if len(data) > MAX_EXPORT_ROWS:
                    raise ValueError(
                        "Maximum 50,000 rows are supported "
                        "per upload."
                    )

                print("\nCSV uploaded successfully!")
                print("Total network flows:", len(data))

                # ----------------------------------
                # STEP 5: Validate model features
                # ----------------------------------

                missing_features = [
                    name for name in feature_names
                    if name not in data.columns
                ]

                if missing_features:

                    raise ValueError(
                        "Missing required features: "
                        + ", ".join(missing_features[:10])
                    )

                features = data.loc[:, feature_names].copy()

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
                # STEP 6: Predict network traffic
                # ----------------------------------

                print("Analyzing network traffic...")

                predictions = np.asarray(
                    model.predict(features)
                ).astype(int)

                if not np.isin(
                    predictions, [0, 1]
                ).all():

                    raise ValueError(
                        "Unexpected model prediction labels."
                    )

                # ----------------------------------
                # STEP 7: Count predictions
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

                # ----------------------------------
                # STEP 8: Prepare dashboard results
                # ----------------------------------

                results = {
                    "total": total_count,
                    "benign": benign_count,
                    "attacks": attack_count,
                    "benign_percent": benign_percentage,
                    "attack_percent": attack_percentage,

                    # Chart data
                    "chart_labels": [
                        "BENIGN",
                        "ATTACK"
                    ],
                    "chart_values": [
                        benign_count,
                        attack_count
                    ],

                    # Attack alert
                    "attack_detected": attack_count > 0
                }

                # ----------------------------------
                # STEP 9: Export individual predictions
                # ----------------------------------

                export_data = data.copy()

                export_data["Predicted_Label"] = np.where(
                    predictions == 1,
                    "ATTACK",
                    "BENIGN"
                )

                export_data["Predicted_Class"] = predictions

                # Generate a unique export identifier
                export_id = secrets.token_hex(16)

                export_path = (
                    EXPORT_FOLDER /
                    f"{export_id}.csv"
                )

                export_data.to_csv(
                    export_path,
                    index=False
                )

                # Associate the export with this session
                session["export_id"] = export_id

                results["download_available"] = True

                print("Prediction CSV saved successfully!")

                # ----------------------------------
                # STEP 10: Print results
                # ----------------------------------

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

                print("\nNetwork traffic analysis completed!")

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


# --------------------------------------------------
# STEP 11: Download prediction results
# --------------------------------------------------

@app.route("/download")
def download_results():

    export_id = session.get("export_id")

    if not export_id:
        return "No prediction results available.", 404

    export_path = (
        EXPORT_FOLDER /
        f"{export_id}.csv"
    )

    if not export_path.is_file():
        return "No prediction results available.", 404

    return send_file(
        export_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="intrusion_detection_results.csv"
    )


# --------------------------------------------------
# STEP 12: Handle oversized uploads
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        results=None,
        error="File is too large. Maximum size is 10 MB."
    ), 413


# --------------------------------------------------
# STEP 13: Start Flask
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "\nStarting Network Intrusion Detection Dashboard..."
    )

    # Local development only.
    # Do not enable debug mode in production.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
