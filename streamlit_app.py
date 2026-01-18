import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Bank Marketing Classification – Model Evaluation App")
st.write(
    "Upload a **test CSV file**, select a classification model, "
    "and view evaluation metrics and confusion matrix."
)

# -------------------------------------------------
# Model dictionary
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

# -------------------------------------------------
# Model selection dropdown  ✅ (b)
# -------------------------------------------------
selected_model_name = st.selectbox(
    "Select a Classification Model",
    list(models.keys())
)

model = models[selected_model_name]

# -------------------------------------------------
# Dataset upload  ✅ (a)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV format only)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file, sep=";")

# -------------------------------------------------
# Target and features
# -------------------------------------------------
if "y" not in df.columns:
    st.error("Uploaded dataset must contain target column 'y'.")
    st.stop()

y = df["y"].map({"yes": 1, "no": 0})
X = df.drop(columns=["y"])

# -------------------------------------------------
# Encode categorical variables
# -------------------------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# -------------------------------------------------
# Feature scaling
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Train model on uploaded test data
# (allowed per assignment instruction)
# -------------------------------------------------
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# -------------------------------------------------
# Evaluation metrics  ✅ (c)
# -------------------------------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)
auc = roc_auc_score(y, y_proba)

st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC Score", f"{auc:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# -------------------------------------------------
# Confusion Matrix  ✅ (d)
# -------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)

# -------------------------------------------------
# Classification Report (optional bonus, not required)
# -------------------------------------------------
with st.expander("View Classification Report"):
    st.text(classification_report(y, y_pred))

st.success("Evaluation completed successfully.")
