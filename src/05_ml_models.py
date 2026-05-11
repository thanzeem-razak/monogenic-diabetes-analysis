import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/diabetes_012.csv")

numeric_columns = ['Insulin Levels', 'BMI', 'Blood Pressure',
                   'Blood Glucose Levels', 'Cholesterol Levels', 'Birth Weight']

categorical_vars = ['Glucose Tolerance Test', 'Liver Function Tests',
                    'Urine Test', 'Early Onset Symptoms']

# ── Preprocessor (shared between both tasks) ──────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_columns),
        ('cat', OneHotEncoder(), categorical_vars)
    ]
)

# ── Task 1: Binary classification — monogenic vs non-monogenic ────────────────
X_task1 = df[numeric_columns + categorical_vars]
y_task1 = df['Target'].apply(
    lambda x: 1 if x in ['MODY', 'Neonatal Diabetes Mellitus (NDM)'] else 0
)

X_train_task1, X_test_task1, y_train_task1, y_test_task1 = train_test_split(
    X_task1, y_task1, test_size=0.3, random_state=42, stratify=y_task1
)

models_task1 = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier()
}

plt.figure(figsize=(10, 8))
for name, model in models_task1.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train_task1, y_train_task1)
    y_pred = pipeline.predict(X_test_task1)
    y_prob = pipeline.predict_proba(X_test_task1)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{name} — Task 1 (Monogenic Diabetes Prediction):")
    print(f"  Accuracy: {accuracy_score(y_test_task1, y_pred):.4f}")
    print(classification_report(y_test_task1, y_pred,
                                target_names=['Non-Monogenic', 'Monogenic']))
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test_task1, y_prob)
        print(f"  ROC-AUC: {roc_auc:.4f}")
        fpr, tpr, _ = roc_curve(y_test_task1, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve — Task 1: Monogenic Diabetes Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ── Task 2: Multiclass — MODY vs NDM vs non-monogenic ────────────────────────
X_task2 = df[numeric_columns + categorical_vars]
y_task2 = df['Target'].apply(
    lambda x: 0 if x == 'MODY' else (1 if x == 'Neonatal Diabetes Mellitus (NDM)' else 2)
)
label_mapping_task2 = {0: 'MODY', 1: 'NDM', 2: 'Non-Monogenic'}

X_train_task2, X_test_task2, y_train_task2, y_test_task2 = train_test_split(
    X_task2, y_task2, test_size=0.3, random_state=42, stratify=y_task2
)

models_task2 = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier()
}

plt.figure(figsize=(10, 8))
for name, model in models_task2.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train_task2, y_train_task2)
    y_pred = pipeline.predict(X_test_task2)
    y_prob = pipeline.predict_proba(X_test_task2) if hasattr(model, "predict_proba") else None

    print(f"\n{name} — Task 2 (MODY / NDM / Non-Monogenic):")
    print(f"  Accuracy: {accuracy_score(y_test_task2, y_pred):.4f}")
    print(classification_report(y_test_task2, y_pred,
                                target_names=['MODY', 'NDM', 'Non-Monogenic']))
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test_task2, y_prob, multi_class="ovr")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        for i, label in label_mapping_task2.items():
            if i == 2:
                continue
            fpr, tpr, _ = roc_curve(y_test_task2 == i, y_prob[:, i])
            roc_auc_label = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} - {label} (AUC = {roc_auc_label:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve — Task 2: MODY and NDM Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 4, figsize=(20, 12))

for i, (name, model) in enumerate(models_task1.items()):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train_task1, y_train_task1)
    cm = confusion_matrix(y_test_task1, pipeline.predict(X_test_task1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0, i], cbar=False)
    ax[0, i].set_title(f'{name}\nTask 1 Confusion Matrix')
    ax[0, i].set_xlabel('Predicted')
    ax[0, i].set_ylabel('True')

for i, (name, model) in enumerate(models_task2.items()):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train_task2, y_train_task2)
    cm = confusion_matrix(y_test_task2, pipeline.predict(X_test_task2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1, i], cbar=False)
    ax[1, i].set_title(f'{name}\nTask 2 Confusion Matrix')
    ax[1, i].set_xlabel('Predicted')
    ax[1, i].set_ylabel('True')

plt.tight_layout()
plt.show()

# ── Accuracy comparison bar charts ────────────────────────────────────────────
models_list      = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
task1_accuracies = [0.9161, 0.9595, 0.9432, 0.9334]
task2_accuracies = [0.9315, 0.9568, 0.9415, 0.9306]

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(models_list))

rects1 = ax.bar(x - bar_width / 2, task1_accuracies, bar_width,
                label='Task 1 Accuracy', color='skyblue')
rects2 = ax.bar(x + bar_width / 2, task2_accuracies, bar_width,
                label='Task 2 Accuracy', color='lightgreen')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison — Task 1 vs Task 2')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_ylim(0, 1.05)

for rect in list(rects1) + list(rects2):
    height = rect.get_height()
    ax.annotate(f'{height * 100:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

# ── K-Fold cross-validation on Random Forest ─────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])

task1_fold_scores = []
task2_fold_scores = []

plt.figure(figsize=(15, 6))

# Task 1 folds
plt.subplot(1, 2, 1)
for fold, (train_idx, val_idx) in enumerate(cv.split(X_task1, y_task1), start=1):
    pipeline.fit(X_task1.iloc[train_idx], y_task1.iloc[train_idx])
    y_prob = pipeline.predict_proba(X_task1.iloc[val_idx])[:, 1]
    score  = roc_auc_score(y_task1.iloc[val_idx], y_prob)
    task1_fold_scores.append(score)
    plt.plot(fold, score, 'bo-', markersize=8, label='Fold scores' if fold == 1 else "")

mean1 = np.mean(task1_fold_scores)
std1  = np.std(task1_fold_scores)
plt.plot(range(1, 6), task1_fold_scores, 'b-')
plt.axhline(y=mean1, color='r', linestyle='--', label=f'Mean ROC-AUC = {mean1:.4f}')
plt.fill_between(range(1, 6), mean1 - std1, mean1 + std1,
                 color='b', alpha=0.2, label=f'Std: ±{std1:.4f}')
plt.ylim(min(task1_fold_scores) - 0.001, max(task1_fold_scores) + 0.001)
plt.title('Task 1: Monogenic Diabetes\nK-Fold Cross-Validation')
plt.xlabel('Fold')
plt.ylabel('ROC-AUC Score')
plt.legend()
plt.grid(True)

# Task 2 folds
plt.subplot(1, 2, 2)
for fold, (train_idx, val_idx) in enumerate(cv.split(X_task2, y_task2), start=1):
    pipeline.fit(X_task2.iloc[train_idx], y_task2.iloc[train_idx])
    y_prob = pipeline.predict_proba(X_task2.iloc[val_idx])
    score  = roc_auc_score(y_task2.iloc[val_idx], y_prob,
                           multi_class='ovr', average='macro')
    task2_fold_scores.append(score)
    plt.plot(fold, score, 'go-', markersize=8, label='Fold scores' if fold == 1 else "")

mean2 = np.mean(task2_fold_scores)
std2  = np.std(task2_fold_scores)
plt.plot(range(1, 6), task2_fold_scores, 'g-')
plt.axhline(y=mean2, color='orange', linestyle='--', label=f'Mean ROC-AUC = {mean2:.4f}')
plt.fill_between(range(1, 6), mean2 - std2, mean2 + std2,
                 color='g', alpha=0.2, label=f'Std: ±{std2:.4f}')
plt.ylim(min(task2_fold_scores) - 0.001, max(task2_fold_scores) + 0.001)
plt.title('Task 2: MODY and NDM Classification\nK-Fold Cross-Validation')
plt.xlabel('Fold')
plt.ylabel('ROC-AUC Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()