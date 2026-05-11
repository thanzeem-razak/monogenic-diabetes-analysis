# Monogenic Diabetes Clinical Parameter Analysis

**Indiana University — Luddy School of Informatics, Computing, and Engineering**  
Team: Aditi Elizabath George, Zhen Hou, Thanzeem Noorul Amin Razak, Swikriti Subedi

---

## Overview

This project investigates whether specific clinical parameters are statistically associated 
with monogenic diabetes (MODY and Neonatal Diabetes Mellitus), and builds machine learning 
models to predict diabetes type — with the goal of reducing reliance on expensive genetic testing.

---

## Research Questions

- Do clinical parameters (insulin levels, BMI, blood pressure, cholesterol, blood glucose, 
  birth weight, and others) show statistically significant associations with MODY or NDM?
- Can machine learning models predict monogenic diabetes type from clinical parameters alone?

---

## Dataset

Source: [Kaggle — Diabetes Dataset](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data)  
Size: 70,000 rows × 35 columns  
Stored and queried via MySQL; analyzed in Python.


---

## Methods

### Preprocessing
- No null values found
- Outliers detected via IQR/boxplots and corrected using **Winsorization**

### Exploratory Data Analysis
- Histograms for numerical variables by diabetes type
- Pie charts for categorical variables
- Correlation heatmap (StandardScaler applied before correlation)

### Statistical Analysis
- Normality: Shapiro-Wilk flagged (n > 5000); confirmed non-normal via **Q-Q plots**
- Numerical variables: **Kruskal-Wallis H test**
- Categorical variables: **Chi-square test**

### Machine Learning
Two classification tasks:
- **Task 1**: Binary — monogenic vs. non-monogenic diabetes
- **Task 2**: Multiclass — MODY vs. NDM vs. non-monogenic

Models: Logistic Regression, Random Forest, SVM, KNN  
Evaluation: Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix, K-Fold Cross-Validation

---

## Results Summary

| Model               | Task 1 Accuracy | Task 2 Accuracy | ROC-AUC |
|---------------------|----------------|----------------|---------|
| Logistic Regression | 91.6%          | 93.2%          | 0.9534  |
| Random Forest       | **95.95%**     | **95.68%**     | **0.9916** |
| SVM                 | 94.3%          | 94.2%          | 0.9750  |
| KNN                 | 93.3%          | 93.1%          | 0.9504  |

**Random Forest** was the best-performing model across both tasks, with K-fold cross-validation 
confirming stability (mean AUC 0.957, std ±0.023).

### Key Statistical Findings
- **Numerical variables** (insulin, BMI, blood pressure, glucose, cholesterol, birth weight): 
  significant association with both MODY and NDM (p < 0.05) → **null hypothesis rejected**
- **Categorical variables** (glucose tolerance test, liver function, urine test, early onset 
  symptoms): no significant association (p > 0.05) → **null hypothesis not rejected**

---

## Setup

```bash
pip install -r requirements.txt
```

Run scripts in order:
```bash
python src/01_data_loading.py
python src/02_preprocessing.py
python src/03_eda.py
python src/04_statistical_analysis.py
python src/05_ml_models.py
```

> **Note:** Scripts 01 uses a MySQL connection. If you don't have the database set up, 
> load the CSV directly using `pd.read_csv('data/diabetes_012.csv')` as a substitute 
> at the top of each file.

---

## Limitations

- No healthy control group in the dataset
- MODY + NDM cases represent only ~14% of total samples (class imbalance)
- Unknown population/geography limits generalizability

---

## Technologies

Python · Pandas · NumPy · Scikit-learn · SciPy · Matplotlib · Seaborn · MySQL

---

## Report

Full write-up available in `/data/Project report.pdf`