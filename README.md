# Bank Marketing Classification using Machine Learning

## Dataset Description
This project uses the **Bank Marketing Dataset** from the **UCI Machine Learning Repository**.
The dataset contains information related to direct marketing campaigns of a Portuguese banking institution.

- Dataset file: `bank-additional-full.csv`
- Total instances: 45,211
- Number of features: 16
- Target variable: `y`
  - `yes` → Client subscribed to a term deposit
  - `no` → Client did not subscribe

The problem is formulated as a **binary classification task**.

---

## Objective
The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on demographic, social, and campaign-related features.

---

## Machine Learning Models Implemented
The following six classification models were implemented on the same dataset and evaluated using identical train-test splits:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

Each model is implemented as a separate `.py` file inside the `model/` directory, as per assignment requirements.

---

## Evaluation Metrics
Each model is evaluated using the following performance metrics:

1. Accuracy  
2. AUC Score  
3. Precision  
4. Recall  
5. F1 Score  
6. Matthews Correlation Coefficient (MCC)

These metrics provide a comprehensive evaluation, especially considering class imbalance in the dataset.

---

## Experimental Results

The following table summarizes the performance of all models on the test dataset:

| Model               | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC   |
|--------------------|----------|-----------|-----------|--------|----------|-------|
| Logistic Regression | 0.9139   | 0.9370    | 0.7002    | 0.4127 | 0.5193   | 0.4956 |
| Decision Tree       | 0.8956   | 0.7535    | 0.5343    | 0.5700 | 0.5516   | 0.4929 |
| KNN                 | 0.9053   | 0.8617    | 0.6267    | 0.3944 | 0.4841   | 0.4491 |
| Naive Bayes         | 0.8536   | 0.8606    | 0.4024    | 0.6175 | 0.4872   | 0.4189 |
| Random Forest       | 0.9205   | 0.9507    | 0.6888    | 0.5366 | 0.6033   | 0.5652 |
| XGBoost             | 0.9167   | 0.9495    | 0.6505    | 0.5636 | 0.6039   | 0.5595 |

---

## Observations
- Ensemble models (**Random Forest** and **XGBoost**) achieved the best overall performance across most metrics.
- Random Forest obtained the highest MCC score, indicating better balanced classification performance.
- Logistic Regression achieved high accuracy and AUC but lower recall, reflecting the class imbalance.
- Naive Bayes showed higher recall but lower precision compared to other models.
- MCC proved to be a useful metric for comparing models beyond accuracy.

---

## Project Structure
