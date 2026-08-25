import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load German Credit Dataset
# ==========================================

dataset = fetch_ucirepo(id=144)

X = dataset.data.features
y = dataset.data.targets["class"]

print("Dataset Shape:", X.shape)


# ==========================================
# 2. Convert Target
# ==========================================

# 0 = Good Credit
# 1 = Bad Credit

y = y.map({1: 0, 2: 1})

print("\nTarget Distribution:")
print(y.value_counts())


# ==========================================
# 3. Identify Columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumeric Columns:")
print(numeric_columns)


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        )
    ]
)


# ==========================================
# 5. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 6. Logistic Regression
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ==========================================
# 7. Train Model
# ==========================================

print("\n========================================")
print("TRAINING IMPROVED LOGISTIC REGRESSION")
print("========================================")

model.fit(X_train, y_train)


# ==========================================
# 8. Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 9. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ==========================================
# 10. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Good Credit",
            "Bad Credit"
        ]
    )
)


# ==========================================
# 11. Confusion Matrix
# ==========================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ==========================================
# 12. Final Message
# ==========================================

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")
 # ==========================================
# 13. Save Trained Model
# ==========================================

import os
import joblib

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained pipeline
model_path = "models/credit_model.pkl"

joblib.dump(model, model_path)

print("\n========================================")
print("MODEL SAVED")
print("========================================")
print(f"Model saved successfully at: {model_path}")