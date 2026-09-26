"""
Trains the salary-prediction model on experience + education based
features, with salary in INR, and saves everything the Streamlit
dashboard needs at inference time.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATA_PATH = "ds_salaries_inr.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

categorical_cols = ["education_level", "employment_type", "job_title", "company_size"]
data_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in data_encoded.columns if c != "salary_in_inr"]
X = data_encoded[feature_cols]
y = data_encoded["salary_in_inr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42)
model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)
print(f"Test R^2: {r2:.3f}")

options = {
    "education_level": ["High School", "Bachelor's", "Master's", "PhD"],
    "employment_type": sorted(df["employment_type"].unique().tolist()),
    "job_title": sorted(df["job_title"].unique().tolist()),
    "company_size": ["S", "M", "L"],
    "remote_ratio": sorted(df["remote_ratio"].unique().tolist()),
    "experience_min": int(df["experience_years"].min()),
    "experience_max": int(df["experience_years"].max()),
}

avg_by_education = df.groupby("education_level")["salary_in_inr"].mean().to_dict()
avg_by_job_title = df.groupby("job_title")["salary_in_inr"].mean().to_dict()
overall_avg = df["salary_in_inr"].mean()
overall_salaries = df["salary_in_inr"].tolist()

artifact = {
    "model": model,
    "feature_cols": feature_cols,
    "options": options,
    "avg_by_education": avg_by_education,
    "avg_by_job_title": avg_by_job_title,
    "overall_avg": overall_avg,
    "overall_salaries": overall_salaries,
    "test_r2": r2,
}

with open(f"{MODEL_DIR}/salary_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print(f"Saved model + metadata -> {MODEL_DIR}/salary_model.pkl")
