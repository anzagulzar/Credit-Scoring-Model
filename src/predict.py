import joblib
import pandas as pd


# ==========================================
# 1. Load Trained Model
# ==========================================

model = joblib.load("models/credit_model.pkl")


# ==========================================
# 2. Sample Applicant
# ==========================================

applicant = pd.DataFrame([{
    "Attribute1": "A11",
    "Attribute2": 6,
    "Attribute3": "A34",
    "Attribute4": "A43",
    "Attribute5": 1169,
    "Attribute6": "A65",
    "Attribute7": "A75",
    "Attribute8": 4,
    "Attribute9": "A93",
    "Attribute10": "A101",
    "Attribute11": 4,
    "Attribute12": "A121",
    "Attribute13": 67,
    "Attribute14": "A143",
    "Attribute15": "A152",
    "Attribute16": 2,
    "Attribute17": "A173",
    "Attribute18": 1,
    "Attribute19": "A192",
    "Attribute20": "A201"
}])


# ==========================================
# 3. Make Prediction
# ==========================================

prediction = model.predict(applicant)[0]

probability = model.predict_proba(applicant)[0]


# ==========================================
# 4. Display Result
# ==========================================

print("\n========================================")
print("       CREDIT SCORING RESULT")
print("========================================")

if prediction == 0:
    print("Prediction: GOOD CREDIT")
else:
    print("Prediction: BAD CREDIT")

print(f"\nGood Credit Probability: {probability[0] * 100:.2f}%")
print(f"Bad Credit Probability:  {probability[1] * 100:.2f}%")

print("========================================")