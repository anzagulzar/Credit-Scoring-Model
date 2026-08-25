
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load German Credit Dataset
# ==========================================

dataset = fetch_ucirepo(id=144)

X = dataset.data.features
y = dataset.data.targets["class"]

print("Original Features Shape:", X.shape)
print("Original Target Shape:", y.shape)


# ==========================================
# 2. Convert Target
# ==========================================

# Convert class 1/2 into 0/1
# 1 = Good Credit
# 2 = Bad Credit

y = y.map({1: 0, 2: 1})

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 3. Identify Categorical Columns
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
# 4. Create Preprocessor
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
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
# 6. Fit Preprocessor on Training Data
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)

print("\nPreprocessing completed successfully!")

