
# ML-Based Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (NIDS)
developed using Python, Scikit-learn, XGBoost, and Flask.

The project analyzes network traffic flow features and classifies
traffic as **BENIGN** or **ATTACK**.

It includes data preprocessing, machine learning model comparison,
performance evaluation, automated testing, and an interactive
Flask web dashboard.

**Project Status:** Functional local prototype with automated
application testing completed.

**Author:** Priyanka Sridhar

**GitHub Repository:**
https://github.com/PriyankaSridhar17/ML-Based-Network-Intrusion-Detection-System

---

## 1. Project Overview

Network Intrusion Detection Systems help identify potentially
malicious activity by analyzing network traffic.

This project uses supervised machine learning to classify
network traffic records into two categories:

- **BENIGN (0):** Traffic predicted to be legitimate.
- **ATTACK (1):** Traffic predicted to be potentially malicious.

The system analyzes pre-extracted network flow features
from CSV files rather than capturing live network packets.

Three machine learning classifiers were implemented and
compared:

1. Decision Tree
2. Random Forest
3. XGBoost

XGBoost was selected as the project's best-performing
baseline based on the saved evaluation results.

A Flask web dashboard provides a user-friendly interface
for uploading CSV files, reviewing predictions, and
downloading results.

---

## 2. Project Objectives

- Analyze network traffic using machine learning.
- Prepare and clean the CICIDS2017 dataset.
- Perform exploratory data analysis.
- Train multiple classification algorithms.
- Compare model performance.
- Evaluate accuracy, precision, recall, and F1-score.
- Analyze confusion matrices and feature importance.
- Develop an interactive Flask dashboard.
- Validate uploaded network traffic datasets.
- Generate BENIGN and ATTACK predictions.
- Export prediction results to CSV.
- Implement automated application testing.
- Publish documented project code on GitHub.

---

## 3. Technologies Used

### Programming and Data Processing

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- Joblib

### Data Visualization

- Matplotlib

### Web Application

- Flask
- HTML
- CSS

### Testing and Development

- Pytest 9.2.0
- Flask Test Client
- Visual Studio Code
- Git
- GitHub

---

## 4. Dataset

**Dataset:** CICIDS2017

CICIDS2017 contains network traffic records representing
benign network activity and several attack categories.

The dataset provides network flow characteristics that
can be used for intrusion detection research.

### Dataset Sampling

A 5% sample was collected from each source CSV file
in processing chunks to reduce computational requirements.

### Data Preprocessing

The preprocessing workflow included:

1. Loading source CSV files.
2. Sampling network traffic records.
3. Standardizing feature names.
4. Converting features to numeric values.
5. Replacing infinite values with missing values.
6. Removing records containing missing values.
7. Converting traffic labels into binary classes.
8. Removing duplicate feature records.
9. Splitting the cleaned dataset for model evaluation.

### Dataset Split

| Property | Value |
|---|---|
| Training records | 108,547 |
| Testing records | 27,137 |
| Total records | 135,684 |
| Training ratio | 80% |
| Testing ratio | 20% |
| Random state | 42 |
| Split method | Stratified random row split |

The split preserves the approximate class distribution
between the training and testing datasets.

However, random row splitting does not guarantee
independence between network capture sessions.

---

## 5. Machine Learning Models

### Decision Tree

A supervised learning algorithm that creates decision
rules based on network traffic features.

### Random Forest

An ensemble learning algorithm that combines multiple
decision trees to generate classifications.

### XGBoost

A gradient-boosted tree algorithm used to classify
network traffic based on learned feature patterns.

XGBoost was selected as the best-performing baseline
in the project's saved model comparison.

---

## 6. XGBoost Evaluation Results

The trained XGBoost classifier was evaluated using
27,137 saved test records.

| Evaluation Metric | Result |
|---|---|
| Accuracy | 99.87% |
| Precision | 99.62% |
| Recall | 99.68% |
| F1-score | 99.65% |
| True Negatives | 22,182 |
| False Positives | 19 |
| False Negatives | 16 |
| True Positives | 4,920 |

### Confusion Matrix Interpretation

- **True Negative:** BENIGN traffic correctly classified as BENIGN.
- **False Positive:** BENIGN traffic incorrectly classified as ATTACK.
- **False Negative:** ATTACK traffic incorrectly classified as BENIGN.
- **True Positive:** ATTACK traffic correctly classified as ATTACK.

False negatives are particularly important because
they represent attack traffic that was not detected.

### Evaluation Disclaimer

The reported metrics describe performance on the
project's saved test dataset.

They do not establish performance on live traffic,
unseen capture sessions, or independent network environments.

---

## 7. Feature Importance Analysis

XGBoost feature importance analysis was performed
to identify influential network traffic characteristics.

Important features included:

- Bwd Packet Length Std
- Average Packet Size
- Avg Bwd Segment Size
- Max Packet Length
- Bwd Header Length

Feature importance helps explain model behavior but
does not establish that a feature causes an attack.

---

## 8. Flask Network Security Dashboard

An interactive Flask application was developed
to demonstrate network intrusion detection predictions.

### Dashboard Features

- Upload network traffic CSV files.
- Validate uploaded datasets.
- Analyze traffic using the trained XGBoost classifier.
- Display total network flows.
- Display BENIGN prediction counts.
- Display ATTACK prediction counts.
- Calculate traffic classification percentages.
- Visualize network traffic distribution.
- Display potential attack alerts.
- Download prediction results as CSV.

### Dashboard Input Limits

| Configuration | Limit |
|---|---|
| Maximum upload size | 10 MB |
| Maximum rows per upload | 50,000 |
| Required network features | 78 |
| Supported upload format | CSV |

### Dashboard Screenshots

#### Dashboard Homepage

![Dashboard Homepage](screenshots/dashboard_home.png)

#### Intrusion Detection Results

![Prediction Results](screenshots/prediction_results.png)

#### Potential Attack Alert

![Attack Alert](screenshots/attack_alert.png)

### Example Prediction Results

A demonstration CSV containing 100 network traffic
records produced the following predictions:

| Prediction Category | Count | Percentage |
|---|---:|---:|
| BENIGN | 89 | 89.0% |
| ATTACK | 11 | 11.0% |
| Total | 100 | 100.0% |

These values are predictions for the demonstration
input and are not confirmed security incidents.

---

## 9. Input Validation

The Flask application includes checks for:

- Missing uploaded files.
- Unsupported file extensions.
- Empty CSV datasets.
- Missing required network features.
- Duplicate column names.
- Nonnumeric feature values.
- Missing numerical values.
- Infinite numerical values.
- Oversized file uploads.
- Excessive row counts.

Input validation reduces errors and helps prevent
unsupported datasets from reaching the prediction stage.

---

## 10. Automated Flask Application Testing

Automated application tests were implemented using
**Pytest 9.2.0** and the **Flask test client**.

The tests verify selected application behaviors
without requiring a web browser.

### Automated Test Coverage

| Test Case | Result |
|---|---|
| Homepage loading | PASSED |
| Missing uploaded file | PASSED |
| Unsupported file extension | PASSED |
| Empty CSV upload | PASSED |
| Missing required features | PASSED |
| Nonnumeric feature values | PASSED |
| Download without predictions | PASSED |
| Oversized file upload | PASSED |
| Successful prediction and CSV download | PASSED |

### Day 14 Test Results

| Metric | Result |
|---|---|
| Total tests | 9 |
| Passed | 9 |
| Failed | 0 |
| Errors | 0 |
| Execution time | 4.82 seconds |
| Testing framework | Pytest 9.2.0 |

All nine tests passed during the verified Day 14 test run.

### Prediction and Download Test

The successful prediction test uses a controlled
mock classifier to verify:

1. Valid CSV upload.
2. Processing of two network traffic records.
3. Generation of BENIGN and ATTACK predictions.
4. Successful prediction CSV download.
5. Verification of exported prediction labels.
6. Verification of exported prediction classes.

The mock classifier returns known predictions to
verify the application's processing workflow.

This test does not measure the accuracy of the
trained XGBoost model.

### Running Automated Tests

From the project root directory, run:

    python -m pytest tests/test_app.py -v --basetemp=".pytest_temp"

Expected verified test count:

    9 passed

### Testing Limitations

The automated tests cover selected Flask application
behaviors.

They do not constitute a complete security audit,
load test, or independent model evaluation.

Additional testing is recommended before deploying
the application in a production environment.

---

## 11. Data Leakage Investigation

Data leakage checks were performed to investigate
possible overlap between training and testing records.

### Investigation Results

| Check | Result |
|---|---:|
| Duplicate training feature rows | 0 |
| Duplicate testing feature rows | 0 |
| Exact train/test feature overlap | 0 |
| Rounded feature overlap | 129 |

The 129 rounded feature matches do not automatically
indicate data leakage.

However, related network sessions or capture periods
may appear in both training and testing datasets.

Further session-independent evaluation is required
to assess model generalization.

---

## 12. Project Structure

    ML_Network_IDS/
    |
    |-- app.py
    |-- analyze_data.py
    |-- check_data_leakage.py
    |-- check_data_quality.py
    |-- compare_models.py
    |-- count_attacks.py
    |-- explore_data.py
    |-- main.py
    |-- preprocess_data.py
    |-- split_data.py
    |-- train_decision_tree.py
    |-- train_random_forest.py
    |-- train_xgboost.py
    |-- validate_model.py
    |
    |-- data/
    |-- models/
    |-- reports/
    |-- exports/
    |
    |-- screenshots/
    |   |-- dashboard_home.png
    |   |-- prediction_results.png
    |   |-- attack_alert.png
    |
    |-- templates/
    |   |-- index.html
    |
    |-- static/
    |   |-- style.css
    |
    |-- tests/
    |   |-- test_app.py
    |
    |-- .gitignore
    |-- README.md
    |-- requirements.txt

### Important Repository Notes

The following files are excluded from GitHub:

- Raw and processed datasets.
- Trained model files.
- Generated prediction exports.
- Virtual environments.
- Temporary testing files.
- Local backup files.
- Sensitive environment configuration.

The model artifact must be generated locally
or provided separately before running the dashboard.

---

## 13. Installation and Setup

### Step 1: Clone the Repository

    git clone https://github.com/PriyankaSridhar17/ML-Based-Network-Intrusion-Detection-System.git

### Step 2: Open the Project Directory

    cd ML-Based-Network-Intrusion-Detection-System

### Step 3: Create a Virtual Environment

    python -m venv .venv

### Step 4: Activate the Virtual Environment

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

### Step 5: Install Dependencies

    python -m pip install -r requirements.txt

### Step 6: Configure the Flask Secret Key

Set a secure secret key in your PowerShell session:

    $env:FLASK_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"

The key should be provided through the environment
rather than hardcoded into the application.

### Step 7: Provide the Trained Model

The Flask application requires a compatible trained
XGBoost model at:

    models/xgboost_model.joblib

The saved model must be compatible with the
application's required network features.

If the model is not available, it must be generated
using the project's training pipeline.

### Step 8: Start the Flask Application

    python app.py

### Step 9: Open the Dashboard

Open the following address in your browser:

http://127.0.0.1:5000

### Step 10: Upload Network Traffic Data

Upload a compatible CSV containing the required
78 network traffic features.

The dashboard will display classification results
and allow prediction results to be downloaded.

---

## 14. Model Validation

Model evaluation was performed using the saved
test dataset.

Validation activities included:

- Accuracy evaluation.
- Precision evaluation.
- Recall evaluation.
- F1-score evaluation.
- Confusion matrix analysis.
- Feature importance analysis.
- Duplicate record checks.
- Train/test overlap investigation.
- Input feature consistency checks.

These evaluations provide an initial assessment
of model behavior on the sampled CICIDS2017 dataset.

---

## 15. Security Considerations

The application is intended for local development
and academic demonstration.

Current security-related practices include:

- Environment-based Flask secret key configuration.
- Flask debug mode disabled.
- Localhost-only application binding.
- CSV upload validation.
- File upload size limits.
- Required feature validation.
- Temporary testing file exclusions.
- Sensitive file exclusions through .gitignore.

Additional hardening is required before deployment
on a publicly accessible server.

---

## 16. Project Limitations

- The model was evaluated on sampled CICIDS2017 data.
- Random row splitting does not ensure session independence.
- The model has not been independently validated on live traffic.
- The dashboard processes uploaded CSV files rather than live packets.
- Predictions are not confirmed security incidents.
- The trained model is not included in the public repository.
- Automated tests cover selected application behaviors only.
- Public deployment requires additional security hardening.
- Per-user prediction result isolation requires improvement.

---

## 17. Future Improvements

Planned improvements include:

1. Independent capture-session model evaluation.
2. Stronger model generalization testing.
3. Expanded automated test coverage.
4. Improved Flask deployment security.
5. Per-user prediction result isolation.
6. Additional network security visualizations.
7. Improved attack investigation and reporting.
8. Exploration of real-time traffic ingestion.
9. Improved reproducibility and deployment documentation.

---

## 18. Project Progress

### Completed

- Dataset sampling and preprocessing.
- Exploratory data analysis.
- Decision Tree model development.
- Random Forest model development.
- XGBoost model development.
- Model comparison and evaluation.
- Confusion matrix analysis.
- Feature importance analysis.
- Data leakage investigation.
- Flask dashboard development.
- CSV upload validation.
- Prediction result visualization.
- Prediction CSV export.
- GitHub repository publication.
- Dashboard screenshot documentation.
- Nine passing automated Flask application tests.

### Ongoing

- Independent model evaluation.
- Application security improvements.
- Additional automated testing.
- Deployment and reproducibility improvements.

---

## 19. Author

**Priyanka Sridhar**

Master's in Computer Engineering  
California State University, Fullerton

**Areas of Interest:**

- Network Engineering
- Network Security
- Machine Learning
- Network Intrusion Detection
- IT Infrastructure

**GitHub:**
https://github.com/PriyankaSridhar17

---

## 20. Disclaimer

This project is intended for academic research,
learning, and demonstration purposes.

The reported evaluation metrics describe performance
on the project's saved test dataset.

They do not establish performance in live,
independent, or production network environments.

Machine learning predictions should not be treated
as confirmed security incidents without additional
investigation.



---

## Day 15 — Independent Source-File Validation

### Objective

Evaluate whether an XGBoost intrusion detection classifier
can identify malicious traffic from a held-out CICIDS2017
capture containing an attack category absent from training.

### Experimental Design

- Dataset: CICIDS2017
- Training: Seven original source CSV files
- Testing: Friday Afternoon DDoS capture
- Model: Separately trained XGBoost classifier
- Sampling: 5% of each source file
- Features: 78 network traffic characteristics
- Training records: 124,492
- Testing records: 11,287
- Overlapping source files: 0

The original trained model and Flask dashboard were
preserved without modification.

### Experimental Results

| Metric | Result |
|---|---:|
| Accuracy | 78.30% |
| Precision | 99.95% |
| Recall | 61.78% |
| F1-score | 76.36% |
| True Negatives | 4,882 |
| False Positives | 2 |
| False Negatives | 2,447 |
| True Positives | 3,956 |

### Confusion Matrix

![Day 15 Confusion Matrix](experiments/day15/confusion_matrix.png)

### Interpretation

The experimental classifier achieved high precision,
but detected only 61.78% of the DDoS attack records.

Of 6,403 actual DDoS records, 2,447 were incorrectly
classified as BENIGN.

This indicates limited recall for the previously unseen
DDoS attack category in the held-out capture.

### Comparison with Original Baseline

| Metric | Random-Split Baseline | Day 15 Holdout |
|---|---:|---:|
| Accuracy | 99.87% | 78.30% |
| Precision | 99.62% | 99.95% |
| Recall | 99.68% | 61.78% |
| F1-score | 99.65% | 76.36% |

These evaluations are not directly controlled comparisons.
They use different training and testing distributions and
potentially different XGBoost hyperparameters.

The difference does not establish data leakage in the
original experiment.

### Limitations

- The experiment used sampled CICIDS2017 traffic.
- Source-file separation does not guarantee complete
  independence between network sessions.
- DDoS was absent from the training attack categories.
- The results do not establish live-network performance.
- Additional capture-based validation is necessary.

### Research Artifacts

- [Validation script](experiments/day15/independent_validation.py)
- [Evaluation metrics](experiments/day15/validation_metrics.csv)
- [Classification report](experiments/day15/classification_report.txt)
- [Experiment summary](experiments/day15/experiment_summary.json)

### Future Work

- Evaluate additional held-out attack categories.
- Investigate false-negative patterns.
- Perform threshold tuning using separate validation data.
- Compare models under controlled experimental conditions.
- Explore independent capture-session evaluation.


---

## Day 16 — False-Negative Investigation

### Objective

Investigate why the experimental XGBoost classifier
failed to identify 2,447 DDoS attack records during
the Day 15 source-file holdout evaluation.

### Methodology

- Reconstructed the original Day 15 held-out test sample.
- Loaded the previously trained experimental XGBoost model.
- Reproduced the original confusion matrix.
- Analyzed predicted ATTACK scores.
- Compared traffic features between missed and detected attacks.
- Preserved the original classification threshold of 0.50.

No model retraining or threshold optimization was performed.

### Reproduced Evaluation Results

| Metric | Result |
|---|---:|
| Test records | 11,287 |
| True negatives | 4,882 |
| False positives | 2 |
| False negatives | 2,447 |
| True positives | 3,956 |
| Attack recall | 61.78% |

### Prediction Score Analysis

| Outcome | Records | Mean score | Median score |
|---|---:|---:|---:|
| False Negative | 2,447 | 0.01990 | 0.00175 |
| False Positive | 2 | 0.67412 | 0.67412 |
| True Negative | 4,882 | 0.00233 | 0.000099 |
| True Positive | 3,956 | 0.78993 | 0.75937 |

Missed DDoS attacks received substantially lower
ATTACK scores than correctly detected attacks.

This suggests that many missed attacks were classified
as strongly BENIGN-like by the experimental model,
rather than narrowly falling below the 0.50 threshold.

### Feature Comparison

| Feature | False Negative Median | True Positive Median |
|---|---:|---:|
| Flow Bytes/s | 4.56 | 10,806.70 |
| Flow Packets/s | 0.76 | 7.55 |
| Total Backward Packets | 0 | 5 |
| Average Packet Size | 7.5 | 1,292.56 |
| Bwd Packet Length Std | 0 | 3,173.37 |

The missed attacks exhibited lower median flow rates,
packet sizes, and backward packet activity.

These observations suggest possible differences in
traffic patterns, but do not establish causation.

### Attack Score Visualization

![Day 16 Attack Score Distribution](experiments/day16/attack_score_distribution.png)

### Research Findings

1. The Day 15 confusion matrix was successfully reproduced.
2. The experimental classifier missed 2,447 DDoS records.
3. Missed attacks had very low median ATTACK scores.
4. Descriptive feature differences were observed between
   missed and correctly detected attacks.
5. Threshold adjustment alone may not address all
   generalization limitations.

### Limitations

- Analysis was performed on sampled CICIDS2017 traffic.
- The held-out capture contains an attack category absent
  from the experimental training data.
- Prediction scores are not necessarily calibrated probabilities.
- Feature comparisons are descriptive, not causal.
- The held-out test data was not used to select a new threshold.
- Results do not establish performance on live network traffic.

### Day 16 Research Artifacts

- [False-negative analysis script](experiments/day16/false_negative_analysis.py)
- [Analysis summary](experiments/day16/analysis_summary.json)
- [Attack score statistics](experiments/day16/score_summary.csv)
- [Feature comparison](experiments/day16/feature_comparison.csv)
- [Attack score visualization](experiments/day16/attack_score_distribution.png)

### Next Steps

- Investigate training-data distribution differences.
- Design a separate validation set for threshold selection.
- Evaluate class imbalance mitigation strategies.
- Compare experimental models under controlled conditions.
- Test generalization across additional held-out captures.
