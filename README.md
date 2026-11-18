
# ML-Based Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (NIDS) developed using Python, Scikit-learn, XGBoost, and Flask.

The project analyzes network traffic flow features and classifies traffic as **BENIGN** or **ATTACK**. It includes data preprocessing, machine learning model comparison, performance evaluation, and an interactive web dashboard.

## Project Objectives

- Detect potentially malicious network traffic using machine learning.
- Preprocess and analyze the CICIDS2017 dataset.
- Train and compare Decision Tree, Random Forest, and XGBoost classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Develop a Flask dashboard for CSV-based network traffic analysis.
- Display prediction statistics and export individual results.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Flask
- Joblib
- Matplotlib
- HTML and CSS
- Visual Studio Code
- Git and GitHub

## Dataset

**Dataset:** CICIDS2017

CICIDS2017 contains network traffic records representing benign activity and several types of network attacks.

For this project, a 5% sample was collected from each source CSV file in chunks to reduce computational requirements.

### Data Preparation

1. Load the original CICIDS2017 CSV files.
2. Sample 5% of records from each processing chunk.
3. Standardize column names.
4. Convert network features to numeric values.
5. Replace infinite values with missing values.
6. Remove records containing missing values.
7. Convert traffic labels into binary classes.
8. Remove duplicate feature records.
9. Split the dataset into training and testing sets.

### Binary Classification

- **0 — BENIGN:** Network traffic classified as legitimate.
- **1 — ATTACK:** Network traffic classified as potentially malicious.

### Dataset Split

- Training records: 108,547
- Testing records: 27,137
- Split ratio: 80% training / 20% testing
- Stratified random split with `random_state=42`

## Machine Learning Models

Three classification algorithms were implemented:

### Decision Tree

A tree-based classifier that learns decision rules from network traffic features.

### Random Forest

An ensemble classifier that combines predictions from multiple decision trees.

### XGBoost

A gradient-boosted decision tree algorithm selected as the project's best-performing baseline according to the model comparison results.

## XGBoost Evaluation Results

The following results were obtained on the saved test dataset.

| Metric | Result |
|---|---|
| Accuracy | 99.87% |
| Precision | 99.62% |
| Recall | 99.68% |
| F1-score | 99.65% |
| True Negatives | 22,182 |
| False Positives | 19 |
| False Negatives | 16 |
| True Positives | 4,920 |

### Confusion Matrix

The confusion matrix summarizes correct and incorrect classifications.

- **True Negative:** BENIGN traffic correctly classified as BENIGN.
- **False Positive:** BENIGN traffic incorrectly classified as ATTACK.
- **False Negative:** ATTACK traffic incorrectly classified as BENIGN.
- **True Positive:** ATTACK traffic correctly classified as ATTACK.

False negatives are particularly important in intrusion detection because they represent attack traffic that the model failed to identify.

## Feature Importance

XGBoost feature importance analysis was performed to identify influential network traffic characteristics.

Important features included:

- Bwd Packet Length Std
- Average Packet Size
- Avg Bwd Segment Size
- Max Packet Length
- Bwd Header Length

Feature importance describes model behavior and does not establish that a feature causes an attack.

## Flask Web Dashboard

An interactive web application was developed using Flask.

### Features

- Upload network traffic CSV files.
- Validate the uploaded dataset.
- Analyze traffic using the trained XGBoost classifier.
- Display total network flows.
- Show BENIGN and ATTACK prediction counts.
- Calculate prediction percentages.
- Visualize traffic distribution.
- Display potential attack alerts.
- Download individual prediction results as CSV.

### Input Validation

The application includes checks for:

- Missing uploaded files
- Unsupported file extensions
- Empty datasets
- Missing required features
- Duplicate column names
- Nonnumeric feature values
- Missing or infinite numerical values
- Oversized uploads
- Excessive row counts

The dashboard supports up to 50,000 rows per upload and a maximum request size of 10 MB.

## Project Structure

```text
ML_Network_IDS/
|
|-- app.py
|-- preprocess_data.py
|-- split_data.py
|-- train_decision_tree.py
|-- train_random_forest.py
|-- train_xgboost.py
|-- compare_models.py
|-- validate_model.py
|-- check_data_leakage.py
|
|-- data/
|-- models/
|-- reports/
|-- exports/
|
|-- templates/
|   |-- index.html
|
|-- static/
|   |-- style.css
|
|-- README.md
```

## Running the Application

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the trained model

The application requires a compatible trained XGBoost model at:

```text
models/xgboost_model.joblib
```

The saved model must contain the classifier and its required feature names.

If the model is not distributed with the repository, it must be generated using the project's training pipeline.

### 5. Start the application

```bash
python app.py
```

### 6. Open the dashboard

Visit:

http://127.0.0.1:5000

Upload a CSV file containing the required 78 network traffic features.

## Validation and Testing

Model evaluation was performed on 27,137 saved test records.

Additional checks included:

- Exact duplicate records
- Exact train/test feature overlap
- Rounded feature overlap
- Missing values
- Required feature consistency
- Missing-feature CSV uploads
- Empty CSV uploads
- Nonnumeric CSV uploads

### Data Leakage Investigation

The investigation found:

- 0 duplicate training feature rows
- 0 duplicate testing feature rows
- 0 exact overlapping train/test feature rows
- 129 test rows matching training rows after numeric rounding

Rounded matches do not automatically indicate leakage.

The current evaluation uses a stratified random row split. Related network sessions or capture periods may therefore appear in both training and testing datasets.

## Limitations

- The model was evaluated on a sampled, cleaned CICIDS2017 dataset.
- Random row splitting does not guarantee independent capture-session evaluation.
- The reported accuracy should not be interpreted as verified performance on live network traffic.
- The application analyzes uploaded CSV records rather than monitoring live packets.
- Predictions are not confirmed security incidents.
- The current Flask application is intended for local development and demonstration.
- Additional security hardening is required before public deployment.

## Future Improvements

- Evaluate the model using independent capture sessions.
- Implement stronger cross-session generalization tests.
- Improve Flask deployment security.
- Add automated application tests.
- Implement per-user prediction result isolation.
- Explore real-time traffic ingestion and monitoring.

## Project Status

Machine learning model development, initial evaluation, Flask dashboard development, and initial input-validation tests are complete.

Documentation, reproducibility improvements, security hardening, and final GitHub preparation are ongoing.

## Author

Priyanka Sridhar

Machine Learning | Network Security | Network Intrusion Detection
