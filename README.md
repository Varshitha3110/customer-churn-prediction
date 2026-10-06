# 📊 Customer Churn Prediction

An end-to-end machine learning project for predicting
whether a telecom customer is likely to churn.

## 🎯 Project Overview

Customer churn is a major business problem for subscription-based
companies. Identifying customers who are likely to leave allows
businesses to take proactive retention actions.

This project develops a machine learning pipeline that analyzes
customer demographics, service usage, contract information and
billing behavior to predict customer churn.

---

## 🚀 Key Features

- Exploratory Data Analysis
- Data cleaning and validation
- Missing-value handling
- Categorical feature encoding
- Feature scaling
- Multiple classification algorithms
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Classification metrics
- Confusion matrix
- Churn probability prediction
- Reusable ML pipeline

---

## 🧠 Machine Learning Models

The project evaluates:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Naive Bayes

---

## 📈 Model Performance

Initial notebook experiments produced the following
test accuracy results:

| Model | Accuracy |
|---|---:|
| Gradient Boosting | 80.74% |
| AdaBoost | 80.67% |
| Random Forest | 79.89% |
| Decision Tree | 77.04% |
| KNN | 76.62% |
| Logistic Regression | 76.40% |
| SVM | 75.91% |
| Naive Bayes | 70.86% |

> These results come from the original notebook experiment.
> The production-style pipeline may produce different results
> after improved preprocessing, stratification and evaluation.

---

## 🔬 Project Workflow

```text
Customer Data
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing Pipeline
     │
     ├── Numerical Features
     │       ↓
     │     Scaling
     │
     └── Categorical Features
             ↓
        One-Hot Encoding
     │
     ▼
Model Training
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ▼
Churn Prediction
