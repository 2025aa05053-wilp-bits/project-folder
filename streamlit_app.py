import numpy as np
import pandas as pd
import os
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Bank Marketing Classification – Model Comparison")
st.write(
    "This application compares multiple machine learning classification models "
    "on the Bank Marketing dataset using standard evaluation metrics."
)

# -------------------------------
# Load dataset (Streamlit-safe)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bank-additional-full.csv")

if not os.path.exists(DATA_PATH):
    st.error(
        "Dataset file `bank-additional-full.csv` not found. "
        "Please ensure it is committed to the GitHub repository root."
    )
    st.stop()

df = pd.read_csv(DATA_PATH, sep=";")

# -------------------------------
# Target and features
# -------------------------------
y = df["y"].map({"yes": 1, "no": 0})
X = df.drop(columns=["y"])

# -------------------------------
# Encode categorical features
# -------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

# -------------------------------
# Train & Evaluate
# -------------------------------
results = []

with st.spinner("Training models and computing evaluation metrics..."):
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

results_df = pd.DataFrame(results)

# -------------------------------
# Display Results
# -------------------------------
st.subheader("Final Model Comparison")
st.dataframe(results_df, use_container_width=True)

st.success("Model training and evaluation completed successfully.")
