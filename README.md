# 💳 Credit Scoring Model

### CodeAlpha Machine Learning Internship — Task 1

A Machine Learning classification project that predicts whether a credit applicant has **Good Credit** or **Bad Credit** using the German Credit dataset.

The project demonstrates an end-to-end Machine Learning workflow including data loading, exploratory analysis, preprocessing, model training, evaluation, model persistence, and prediction.

---

## 📌 Project Overview

Credit scoring is an important application of Machine Learning in the financial sector. The goal of this project is to analyze applicant information and classify applicants into two categories:

* 🟢 **Good Credit**
* 🔴 **Bad Credit**

The project uses **Logistic Regression** as the final classification model after comparing it with a Decision Tree classifier.

---

## 🎯 Objectives

The main objectives of this project are:

* Load and analyze the German Credit dataset
* Explore the dataset and identify data types
* Check for missing values
* Encode categorical features
* Scale numerical features
* Split data into training and testing sets
* Train Machine Learning classification models
* Evaluate model performance
* Save the trained model
* Predict the credit status of a new applicant

---

## 📊 Dataset

The project uses the **German Credit Dataset** from the UCI Machine Learning Repository.

### Dataset Statistics

| Property         | Value |
| ---------------- | ----: |
| Total Records    | 1,000 |
| Features         |    20 |
| Good Credit      |   700 |
| Bad Credit       |   300 |
| Missing Values   |     0 |
| Training Samples |   800 |
| Testing Samples  |   200 |

The dataset contains both categorical and numerical features.

### Feature Types

**Categorical Features — 13**

```text
Attribute1
Attribute3
Attribute4
Attribute6
Attribute7
Attribute9
Attribute10
Attribute12
Attribute14
Attribute15
Attribute17
Attribute19
Attribute20
```

**Numerical Features — 7**

```text
Attribute2
Attribute5
Attribute8
Attribute11
Attribute13
Attribute16
Attribute18
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* UCI ML Repository
* Logistic Regression
* Decision Tree
* One-Hot Encoding
* StandardScaler

---

## 🔄 Machine Learning Workflow

```text
                    German Credit Dataset
                             │
                             ▼
                     Data Collection
                             │
                             ▼
                       Data Analysis
                             │
                             ▼
                    Missing Value Check
                             │
                             ▼
                   Feature Identification
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
            Categorical              Numerical
              Features                Features
                  │                     │
                  ▼                     ▼
          One-Hot Encoding        StandardScaler
                  │                     │
                  └──────────┬──────────┘
                             ▼
                       Train/Test Split
                             │
                             ▼
                    Logistic Regression
                             │
                             ▼
                       Model Evaluation
                             │
                             ▼
                       Model Saving
                             │
                             ▼
                      Credit Prediction
```

---

## 📁 Project Structure

```text
CodeAlpha_CreditScoring/
│
├── models/
│   └── credit_model.pkl
│
├── src/
│   ├── load_dataset.py
│   ├── data_analysis.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### File Description

| File               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `load_dataset.py`  | Loads and verifies the German Credit dataset         |
| `data_analysis.py` | Performs dataset analysis and checks missing values  |
| `preprocess.py`    | Handles encoding, scaling and train/test preparation |
| `train_model.py`   | Trains and evaluates the Logistic Regression model   |
| `predict.py`       | Uses the saved model to make credit predictions      |
| `credit_model.pkl` | Saved trained Machine Learning pipeline              |
| `requirements.txt` | Python dependencies                                  |
| `README.md`        | Project documentation                                |

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project:

```bash
cd CodeAlpha_CreditScoring
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Step 1 — Load Dataset

Run:

```bash
python src/load_dataset.py
```

This verifies that the dataset contains:

```text
Features: (1000, 20)
Target:   (1000, 1)
```

---

## Step 2 — Analyze Dataset

Run:

```bash
python src/data_analysis.py
```

This performs:

* Dataset inspection
* Data type analysis
* Missing value detection
* Statistical summary
* Unique value analysis
* Target distribution analysis

The dataset contains **no missing values**.

---

## Step 3 — Preprocess Data

Run:

```bash
python src/preprocess.py
```

The preprocessing pipeline performs:

### Categorical Encoding

Categorical features are converted into numerical features using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Numerical Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

### Train/Test Split

The dataset is divided using:

```text
80% Training
20% Testing
```

Result:

```text
Training Data: (800, 20)
Testing Data:  (200, 20)
```

After One-Hot Encoding and scaling, the number of features becomes:

```text
20 → 61
```

This increase is expected because categorical variables are expanded into multiple binary features.

---

# 🤖 Model Training

The project evaluates classification models during development.

## Logistic Regression

The final model uses:

```python
LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)
```

### Why Logistic Regression?

Logistic Regression is a suitable baseline classification algorithm for binary credit-risk prediction. It is computationally efficient and provides probability estimates that can be useful for credit-risk analysis.

---

# 📈 Model Evaluation

## Final Logistic Regression Results

The improved Logistic Regression model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **75.00%** |
| Precision | **55.81%** |
| Recall    | **80.00%** |
| F1 Score  | **65.75%** |

### Classification Report

```text
              precision    recall  f1-score   support

Good Credit       0.89      0.73      0.80       140
Bad Credit        0.56      0.80      0.66        60

accuracy                              0.75       200
macro avg          0.73      0.76      0.73       200
weighted avg       0.79      0.75      0.76       200
```

---

# 🔲 Confusion Matrix

The final model produced:

```text
[[102  38]
 [ 12  48]]
```

### Interpretation

|                 | Predicted Good | Predicted Bad |
| --------------- | -------------: | ------------: |
| **Actual Good** |            102 |            38 |
| **Actual Bad**  |             12 |            48 |

Therefore:

* **102** Good Credit applicants were correctly classified.
* **38** Good Credit applicants were classified as Bad Credit.
* **12** Bad Credit applicants were classified as Good Credit.
* **48** Bad Credit applicants were correctly classified.

The model achieved **80% recall for Bad Credit**, meaning it correctly identified 80% of the Bad Credit cases in the test set.

---

# ⚖️ Model Comparison

A Decision Tree classifier was also evaluated during development.

| Model                          |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------------------ | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression — Initial  | **78.00%** | **67.39%** |     51.67% |     58.49% |
| Decision Tree                  |     64.50% |     41.79% |     46.67% |     44.09% |
| Logistic Regression — Improved |     75.00% |     55.81% | **80.00%** | **65.75%** |

The initial Logistic Regression model achieved higher accuracy, but the improved version significantly increased **Bad Credit recall from 51.67% to 80.00%**.

For a credit-risk application, improving the detection of potentially risky applicants is important, so the improved Logistic Regression model was selected as the final model.

---

# 💾 Model Saving

The trained Machine Learning pipeline is saved using Joblib:

```text
models/credit_model.pkl
```

The saved pipeline contains:

```text
Raw Input
    ↓
One-Hot Encoding
    ↓
Standard Scaling
    ↓
Logistic Regression
    ↓
Prediction
```

This allows the same preprocessing steps to be automatically applied when making predictions.

---

# 🔮 Making Predictions

Run:

```bash
python src/predict.py
```

The prediction script loads:

```text
models/credit_model.pkl
```

and generates a credit prediction.

### Example Output

```text
========================================
       CREDIT SCORING RESULT
========================================

Prediction: GOOD CREDIT

Good Credit Probability: 91.87%
Bad Credit Probability:  8.13%

========================================
```

### Example Prediction

For the test applicant used during development:

```text
Prediction: GOOD CREDIT
```

with:

```text
Good Credit Probability: 91.87%
Bad Credit Probability:   8.13%
```

---

# 🧪 Sample Applicant

The prediction system was tested using a valid sample from the dataset:

```text
Attribute1  = A11
Attribute2  = 6
Attribute3  = A34
Attribute4  = A43
Attribute5  = 1169
Attribute6  = A65
Attribute7  = A75
Attribute8  = 4
Attribute9  = A93
Attribute10 = A101
Attribute11 = 4
Attribute12 = A121
Attribute13 = 67
Attribute14 = A143
Attribute15 = A152
Attribute16 = 2
Attribute17 = A173
Attribute18 = 1
Attribute19 = A192
Attribute20 = A201
```

The trained model predicted:

```text
GOOD CREDIT
```

---

# 📦 Dependencies

The main Python libraries used in this project are:

```text
pandas
scikit-learn
ucimlrepo
joblib
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

# 🔐 Git & Security

The repository includes a `.gitignore` file to prevent unnecessary files from being uploaded to GitHub.

Examples of ignored files/folders:

```text
venv/
__pycache__/
*.pyc
.env
.vscode/
.idea/
```

The virtual environment should **not** be uploaded to GitHub.

---

# 🚀 Future Improvements

The project can be further improved by adding:

* Hyperparameter tuning
* Cross-validation
* ROC-AUC evaluation
* Precision-Recall curve
* Feature importance analysis
* Random Forest comparison
* XGBoost comparison
* Streamlit web interface
* REST API for predictions
* Better input validation
* Interactive credit-risk dashboard
* Model explainability using SHAP

---

# ⚠️ Disclaimer

This project is developed for **educational and internship purposes** as part of the CodeAlpha Machine Learning Internship.

The predictions generated by this model should **not be used as the sole basis for real-world financial or lending decisions**.

---

# 👩‍💻 Author

**Anza Gulzar**

Software Engineering | Machine Learning | Software Development

### CodeAlpha Machine Learning Internship

**Task:** Credit Scoring Model
**Technology:** Python & Scikit-learn
**Dataset:** German Credit Dataset

A173
