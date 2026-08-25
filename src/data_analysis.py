
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch German Credit dataset from UCI
statlog_german_credit = fetch_ucirepo(id=144)

# Features and target
X = statlog_german_credit.data.features
y = statlog_german_credit.data.targets

print("\n===== DATASET INFORMATION =====")

print("\nFEATURES:")
print(X.head())

print("\nFEATURE SHAPE:")
print(X.shape)

print("\nTARGET:")
print(y.head())

print("\nTARGET SHAPE:")
print(y.shape)

print("\nTARGET VALUES:")
print(y.value_counts())

print("\n===== MISSING VALUES =====")
print(X.isnull().sum())

print("\n===== DATA TYPES =====")
print(X.dtypes)

print("\n===== STATISTICAL SUMMARY =====")
print(X.describe(include="all"))

print("\n===== UNIQUE VALUES =====")
for column in X.columns:
    print(f"{column}: {X[column].nunique()} unique values")

