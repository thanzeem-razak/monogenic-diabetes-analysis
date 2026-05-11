import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ── Load and re-apply winsorization ───────────────────────────────────────────
df = pd.read_csv("data/diabetes_012.csv")

numeric_columns = ['Insulin Levels', 'BMI', 'Blood Pressure',
                   'Blood Glucose Levels', 'Cholesterol Levels', 'Birth Weight']

categorical_vars = ['Glucose Tolerance Test', 'Liver Function Tests',
                    'Urine Test', 'Early Onset Symptoms']

# Filter to monogenic diabetes cases only
df_filtered = df[df['Target'].isin(['MODY', 'Neonatal Diabetes Mellitus (NDM)'])]

# ── Histograms for numerical variables ────────────────────────────────────────
for var in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_filtered, x=var, hue='Target', kde=True)
    plt.title(f"Distribution of {var} by Target")
    plt.tight_layout()
    plt.show()

# ── Pie charts for categorical variables ─────────────────────────────────────
for var in categorical_vars:
    for target in df_filtered['Target'].unique():
        data = df_filtered[df_filtered['Target'] == target][var].value_counts()
        if not data.empty:
            plt.figure(figsize=(6, 6))
            data.plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f"{var} Distribution for {target}")
            plt.ylabel('')
            plt.tight_layout()
            plt.show()

# ── Correlation heatmap ───────────────────────────────────────────────────────
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_filtered[numeric_columns]),
    columns=numeric_columns
)

correlation_matrix = df_scaled.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()