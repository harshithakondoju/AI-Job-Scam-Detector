import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "text": [
        "Earn 50,000 per week without interview",
        "Limited seats apply now pay registration fee",
        "Urgent hiring no experience required high salary",
        "Software engineer position at reputed company",
        "Looking for data analyst with 2 years experience",
        "Government job notification official recruitment"
    ],
    "label": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

joblib.dump(model, "job_scam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully!")
input("Press Enter to exit...")
