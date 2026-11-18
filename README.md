
# ML-Based Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (NIDS) developed using Python, Scikit-learn, XGBoost, and Flask.

The project analyzes network traffic flow features and classifies traffic as **BENIGN** or **ATTACK**.

It includes data preprocessing, machine learning model comparison, performance evaluation, data leakage investigation, and an interactive web dashboard.

## Project Objectives

- Detect potentially malicious network traffic using machine learning.
- Preprocess and analyze the CICIDS2017 dataset.
- Train and compare Decision Tree, Random Forest, and XGBoost classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Investigate potential data leakage between training and testing datasets.
- Develop a Flask dashboard for CSV-based network traffic analysis.
- Display prediction statistics and export individual prediction results.

## Technologies Used

**Programming Language**
- Python

**Data Processing**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- XGBoost
- Joblib

**Data Visualization**
- Matplotlib

**Web Development**
- Flask
- HTML
- CSS

**Development Tools**
- Visual Studio Code
- Git
- GitHub

## Dataset

**Dataset:** CICIDS2017

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick

**Official Dataset:**
https://www.unb.ca/cic/datasets/ids-2017.html

CICIDS2017 contains network traffic records representing benign activity and multiple categories of network attacks.

For this project, a 5% sample was collected from each source CSV file in chunks to reduce computational requirements.

### Data Preparation

The preprocessing pipeline performs the following operations:

1. Load the original CICIDS2017 CSV files.
2. Sample 5% of records from each processing chunk.
3. Standardize column names.
4. Convert network traffic features into numeric values.
5. Replace infinite values with missing values.
6. Remove records containing missing values.
7. Convert original traffic labels into binary classes.
8. Remove duplicate feature records.
9. Split the cleaned dataset into training and testing sets.

### Binary Classification

- **0 - BENIGN:** Network traffic classified as legitimate.
- **1 - ATTACK:** Network traffic classified as potentially malicious.

All non-BENIGN labels are mapped to the ATTACK class.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 108,547 |
| Testing | 27,137 |
| Total | 135,684 |

**Split Configuration**
- Training: 80%
- Testing: 20%
- Random state: 42
- Stratified random split

Stratification helps preserve the relative distribution of BENIGN and ATTACK records across the training and testing datasets.

However, random row splitting does not guarantee that related network sessions are separated.

## Machine Learning Models

Three classification algorithms were implemented and compared.

### 1. Decision Tree

A tree-based classification algorithm that learns decision rules from network traffic features.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to produce predictions.

### 3. XGBoost

An optimized gradient-boosted decision tree algorithm used for binary network traffic classification.

XGBoost was selected as the project's best-performing baseline according to the recorded model comparison results.

The trained model uses 78 network traffic features.

## XGBoost Evaluation Results

The following results were calculated from the recorded confusion-matrix counts on the saved test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 99.87% |
| Precision | 99.62% |
| Recall | 99.68% |
| F1-score | 99.65% |
| True Negatives | 22,182 |
| False Positives | 19 |
| False Negatives | 16 |
| True Positives | 4,920 |
| Test Records | 27,137 |

Precision, recall, and F1-score refer to the ATTACK class.

These results represent performance on the sampled CICIDS2017 test split and should not be interpreted as verified performance on independent or live network traffic.

### Confusion Matrix

| Actual / Predicted | BENIGN | ATTACK |
|---|---:|---:|
| BENIGN | 22,182 | 19 |
| ATTACK | 16 | 4,920 |

**Confusion Matrix Explanation**

- **True Negative:** BENIGN traffic correctly classified as BENIGN.
- **False Positive:** BENIGN traffic incorrectly classified as ATTACK.
- **False Negative:** ATTACK traffic incorrectly classified as BENIGN.
- **True Positive:** ATTACK traffic correctly classified as ATTACK.

False negatives are particularly important in intrusion detection because they represent potentially malicious traffic that the model failed to identify.

### Performance Metrics

**Accuracy**

Measures the percentage of all network traffic records classified correctly.

**Precision**

Measures how many records predicted as ATTACK were actually labeled ATTACK.

**Recall**

Measures how many actual ATTACK records were successfully identified.

**F1-score**

Combines precision and recall into a single metric.

## Feature Importance

XGBoost feature importance analysis was performed to identify influential network traffic characteristics.

Important features included:

- Bwd Packet Length Std
- Average Packet Size
- Avg Bwd Segment Size
- Max Packet Length
- Bwd Header Length

Feature importance helps explain which network characteristics influence model predictions.

It does not establish that any individual feature causes a network attack.

## Flask Web Dashboard

An interactive web application was developed using Flask.

The dashboard allows users to upload network traffic CSV files and analyze them using the trained XGBoost classifier.

### Dashboard Features

- Upload network traffic CSV files.
- Validate uploaded datasets.
- Analyze traffic using the trained XGBoost model.
- Display total network flows.
- Show BENIGN prediction counts.
- Show ATTACK prediction counts.
- Calculate prediction percentages.
- Visualize network traffic distribution.
- Display potential attack alerts.
- Download individual prediction results as CSV.

### Input Validation

The application includes checks for:

- Missing uploaded files
- Unsupported file extensions
- Empty datasets
- Missing required model features
- Duplicate column names
- Nonnumeric feature values
- Missing numerical values
- Infinite numerical values
- Oversized uploads
- Excessive row counts

**Application Limits**

- Maximum request size: 10 MB
- Maximum prediction rows: 50,000
- Required network features: 78

Uploaded CSV files must contain the required model features.

The dashboard analyzes network flow records rather than raw network packets.

### Prediction Results

Each network flow is classified as:

- BENIGN
- ATTACK

The dashboard displays summary statistics and a traffic distribution chart.

If potentially malicious traffic is predicted, the application displays an alert.

These alerts are machine learning predictions and do not confirm that a security incident occurred.

### CSV Export

The application supports downloading prediction results.

The exported CSV includes:

- Original network traffic features
- Predicted_Label
- Predicted_Class

**Predicted_Class Values**

- 0 = BENIGN
- 1 = ATTACK

## Project Structure

```text
ML_Network_IDS/
|
|-- app.py
|-- main.py
|-- explore_data.py
|-- analyze_data.py
|-- check_data_quality.py
|-- count_attacks.py
|
|-- preprocess_data.py
|-- split_data.py
|
|-- train_decision_tree.py
|-- train_random_forest.py
|-- train_xgboost.py
|
|-- compare_models.py
|-- validate_model.py
|-- check_data_leakage.py
|
|-- data/                     # Local datasets (excluded)
|
|-- models/                   # Trained models excluded
|   |-- *.joblib
|
|-- reports/
|   |-- day11_validation_metrics.csv
|   |-- day11_data_leakage_report.csv
|   |-- model_comparison.csv
|   |-- model_comparison.png
|   |-- xgboost_feature_importance.csv
|   |-- xgboost_feature_importance.png
|
|-- exports/                  # Local prediction exports
|
|-- templates/
|   |-- index.html
|
|-- static/
|   |-- style.css
|
|-- README.md
|-- requirements.txt
|-- .gitignore
```

Some directories and generated files are excluded from GitHub using `.gitignore`.

## Installation and Setup

The following instructions describe how to configure and run the project locally.

### Prerequisites

Install:

- Python
- Git
- A Python-compatible terminal
- Visual Studio Code (optional)

The project was developed using Python 3.13.

Package compatibility may depend on the installed Python version and operating system.

### Step 1 - Clone the Repository

```bash
git clone https://github.com/PriyankaSridhar17/ML-Based-Network-Intrusion-Detection-System.git
```

Navigate into the project directory:

```bash
cd ML-Based-Network-Intrusion-Detection-System
```

### Step 2 - Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3 - Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 4 - Install Dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the Python packages required by the project.

## Dataset Preparation and Model Training

The original CICIDS2017 dataset and trained model files are not included in this GitHub repository.

Users must obtain the dataset and generate the model locally.

### Step 1 - Download CICIDS2017

Download the CICIDS2017 dataset from:

https://www.unb.ca/cic/datasets/ids-2017.html

Obtain the MachineLearningCVE CSV files used for machine learning analysis.

### Step 2 - Configure the Dataset Location

Open:

```text
preprocess_data.py
```

Locate the dataset directory configuration.

Update the path to point to the local folder containing the original CICIDS2017 CSV files.

For example:

```python
dataset_folder = Path(
    r"C:\IDS_Datasets\MachineLearningCVE"
)
```

This is an example path. Replace it with the actual dataset location.

### Step 3 - Preprocess the Dataset

Run:

```bash
python preprocess_data.py
```

The script:

- Reads the source CSV files.
- Samples network traffic records.
- Cleans feature names.
- Converts numerical features.
- Removes invalid records.
- Saves the cleaned sample.

Expected output file:

```text
data/cleaned_sample.csv
```

### Step 4 - Create Training and Testing Datasets

Run:

```bash
python split_data.py
```

The script creates:

```text
data/X_train.csv
data/X_test.csv
data/y_train.csv
data/y_test.csv
```

### Step 5 - Train the XGBoost Model

Run:

```bash
python train_xgboost.py
```

The training script generates the model artifact required by the dashboard.

Expected model location:

```text
models/xgboost_model.joblib
```

The saved artifact must contain the trained classifier and its required feature names.

### Step 6 - Optional Model Comparison

To reproduce the model comparison workflow, run:

```bash
python train_decision_tree.py
python train_random_forest.py
python compare_models.py
```

### Step 7 - Model Validation

Run:

```bash
python validate_model.py
```

To perform the project's data leakage checks, run:

```bash
python check_data_leakage.py
```

These scripts help evaluate the model and investigate potential overlap between the training and testing datasets.

## Running the Flask Application

### Step 1 - Configure the Flask Secret Key

The Flask application requires the following environment variable:

```text
FLASK_SECRET_KEY
```

In Windows PowerShell, generate a random secret key:

```powershell
$env:FLASK_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
```

This sets the key for the current PowerShell session.

Do not publish or commit secret keys to GitHub.

On macOS or Linux, use:

```bash
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

### Step 2 - Start Flask

Run:

```bash
python app.py
```

The application starts a local development server.

### Step 3 - Open the Dashboard

Open a browser and visit:

http://127.0.0.1:5000

### Step 4 - Upload Network Traffic Data

Upload a CSV file containing the required 78 network traffic features.

The application validates the uploaded file before making predictions.

### Step 5 - View Results

The dashboard displays:

- Total network flows
- BENIGN traffic count
- ATTACK traffic count
- Prediction percentages
- Traffic distribution chart
- Potential attack alerts

### Step 6 - Download Predictions

Click the dashboard's download button to export the individual prediction results as a CSV file.

## Validation and Testing

Model evaluation was performed on 27,137 saved test records.

Additional checks included:

- Duplicate training records
- Duplicate testing records
- Exact train/test feature overlap
- Rounded feature overlap
- Missing values
- Required model features
- Missing-feature CSV uploads
- Empty CSV uploads
- Nonnumeric CSV uploads

## Data Leakage Investigation

The investigation produced the following results:

| Check | Result |
|---|---:|
| Training rows | 108,547 |
| Testing rows | 27,137 |
| Duplicate training feature rows | 0 |
| Duplicate testing feature rows | 0 |
| Exact train/test overlap | 0 |
| Rounded feature overlap | 129 |
| Missing training values | 0 |
| Missing testing values | 0 |
| Model feature count | 78 |
| Missing model features | 0 |

### Interpretation

No exact duplicate feature rows were detected across the saved training and testing datasets.

However, 129 test records matched training records after numeric rounding.

Rounded matches do not automatically establish data leakage.

The current evaluation uses a stratified random row split.

Related network sessions or capture periods may therefore appear in both training and testing datasets.

These checks do not rule out session-level, temporal, preprocessing, or model-selection leakage.

Independent capture-session evaluation is recommended before making claims about real-world detection performance.

## Security Considerations

The application includes basic security measures for local development.

These include:

- Environment-based Flask secret key
- Disabled Flask debug mode
- Restricted upload size
- CSV file extension checks
- Required feature validation
- Numerical input validation
- Maximum prediction row limit
- Exclusion of local datasets and generated files from Git

The application is intended for local demonstration and educational use.

It is not currently designed for production security monitoring.

Additional security hardening would be required before deployment in a public or multi-user environment.

## Limitations

- The model was trained and evaluated using a sampled CICIDS2017 dataset.
- Random row splitting does not guarantee independent network-session evaluation.
- The reported accuracy may overestimate performance on unseen network environments.
- The system does not currently analyze live network packets.
- The dashboard requires precomputed network flow features.
- The trained XGBoost model is not included in the GitHub repository.
- Users must reproduce the training pipeline to generate the model.
- Predictions are not confirmed security incidents.
- The Flask application is intended for local demonstration.
- The current application does not provide per-user prediction result isolation.
- Additional testing and security hardening are required before production deployment.

## Future Improvements

- Evaluate the model on independent network capture sessions.
- Perform stronger cross-session generalization testing.
- Add automated unit and integration tests.
- Improve model reproducibility and experiment tracking.
- Implement per-user prediction result isolation.
- Improve Flask deployment security.
- Explore real-time network traffic ingestion.
- Add additional network security visualizations.
- Improve attack investigation and reporting capabilities.

## Project Status

The following development activities have been completed:

- Dataset preprocessing
- Binary label generation
- Training and testing dataset creation
- Decision Tree model training
- Random Forest model training
- XGBoost model training
- Model comparison
- Initial model evaluation
- Data leakage investigation
- Flask dashboard development
- CSV input validation
- Prediction visualization
- CSV prediction export
- Git repository initialization
- Initial Git commit

Independent network-session validation, automated testing, and production security hardening remain future work.

## Author

**Priyanka Sridhar**

Machine Learning | Network Security | Network Intrusion Detection

GitHub:
https://github.com/PriyankaSridhar17

## Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes.

Machine learning predictions should not be treated as confirmed security incidents without additional investigation.

The reported evaluation metrics describe performance on the project's saved test dataset and do not establish performance in live or independent network environments.
