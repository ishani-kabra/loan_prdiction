#  LoanGuard — Loan Approval Prediction System

An end-to-end machine learning application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on an applicant's financial, credit, employment, and asset information.

 **Live Demo:** [https://loanprdiction-af9kgvbkvmoq9ams4lruse.streamlit.app]

---

##  Project Overview

LoanGuard is a machine learning classification project built to predict loan approval status from applicant information.

The project covers the complete machine learning lifecycle:

**Data preprocessing → Feature Engineering → Model Training → Model Comparison → Hyperparameter Tuning → Pipeline → Deployment**

The final model is integrated into an interactive **Streamlit web application**, allowing users to enter applicant details and receive a prediction along with the estimated approval probability.

---

##  Objective

The objective of this project is to build a reliable loan approval prediction system that can:

* Predict whether a loan will be **Approved** or **Rejected**
* Estimate the probability of loan approval
* Process both categorical and numerical applicant information
* Provide predictions through an easy-to-use web interface

---

##  Dataset

The dataset contains information about loan applicants, including their income, credit score, loan details, education, employment status, and assets.

### Features

| Feature                    | Description                            |
| -------------------------- | -------------------------------------- |
| `no_of_dependents`         | Number of dependents                   |
| `education`                | Applicant's education level            |
| `self_employed`            | Whether the applicant is self-employed |
| `income_annum`             | Annual income                          |
| `loan_amount`              | Requested loan amount                  |
| `loan_term`                | Loan repayment term                    |
| `cibil_score`              | Applicant's CIBIL/credit score         |
| `residential_assets_value` | Value of residential assets            |
| `commercial_assets_value`  | Value of commercial assets             |
| `luxury_assets_value`      | Value of luxury assets                 |
| `bank_asset_value`         | Value of bank assets                   |
| `total_asset`              | Total value of all listed assets       |

### Target

`loan_status`

Possible classes:

* `Approved`
* `Rejected`

---

##  Feature Engineering

A new feature called `total_asset` was created by combining the four asset-related features:

```python
total_asset = (
    residential_assets_value
    + commercial_assets_value
    + luxury_assets_value
    + bank_asset_value
)
```

This provides the model with an overall representation of the applicant's asset value.

---

##  Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Gradient Boosting Model
   ↓
Feature Importance
   ↓
Preprocessing + Model Pipeline
   ↓
Model Serialization
   ↓
Streamlit Application
   ↓
Deployment
```

---

##  Models

Different machine learning approaches were evaluated during the project.

The final model selected was:

### Gradient Boosting Classifier

Hyperparameter optimization was performed using:

```python
GridSearchCV
```

with **5-fold cross-validation**.

The final tuned model was then integrated into the preprocessing pipeline.

---

##  Preprocessing Pipeline

The project uses Scikit-learn's `ColumnTransformer` and `Pipeline`.

Categorical features are processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing and final model are combined into a single pipeline.

```text
Raw Input
    ↓
ColumnTransformer
    ↓
Categorical Encoding
    ↓
Numerical Features
    ↓
Gradient Boosting Classifier
    ↓
Prediction
```

The complete trained pipeline was saved using Joblib:

```text
loan_model_pipeline.pkl
```

This allows the Streamlit application to use the exact same preprocessing and trained model used during development.

---

##  Model Evaluation

The final model was evaluated using classification metrics including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Final Model Performance

> Add your actual results here.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | ***98.36065573770492 %** |
| Precision | **0.9840764331210191 %** |
| Recall    | ** 0.9716981132075472%** |
| F1 Score  | **0.9778481012658228** |

---

##  Streamlit Application

The trained pipeline is used in a Streamlit application.

Users can enter:

* Number of dependents
* Education
* Self-employment status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential assets
* Commercial assets
* Luxury assets
* Bank assets

The application then returns:

```text
Loan Approved / Loan Rejected
```

along with the estimated:

```text
Approval Probability
```

---

##  Deployment

The application was deployed using **Streamlit Community Cloud**.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/loan_prediction.git
```

Move into the project directory:

```bash
cd loan_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Project Structure

```text
loan_prediction/
│
├── app.py
├── loan_model_pipeline.pkl
├── loan_approval_dataset.csv
├── 1.ipynb
├── requirements.txt
└── README.md
```

---

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**
* **Git & GitHub**

---

##  Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Classification
* Gradient Boosting
* Model comparison
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Feature importance
* Scikit-learn Pipeline
* Model serialization
* Streamlit development
* Git/GitHub
* Cloud deployment

---

##  Future Improvements

Potential improvements include:

* Add SHAP-based explainability
* Provide detailed reasons behind approval/rejection
* Improve the Streamlit UI
* Add interactive visualizations
* Add model monitoring
* Experiment with additional ensemble models
* Add automated model retraining

---

##  Author

Ishani Kabra



