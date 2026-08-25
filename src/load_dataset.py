from ucimlrepo import fetch_ucirepo

# Fetch German Credit dataset from UCI
german_credit = fetch_ucirepo(id=144)

# Features
X = german_credit.data.features

# Target
y = german_credit.data.targets

print("FEATURES")
print(X.head())

print("\nFEATURE SHAPE")
print(X.shape)

print("\nTARGET")
print(y.head())

print("\nTARGET SHAPE")
print(y.shape)

print("\nTARGET VALUES")
print(y.value_counts())