import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/diabetes_012.csv")

# ── Outlier detection via boxplots ────────────────────────────────────────────
def detect_outliers_boxplot(df, numeric_columns):
    outliers = {}
    num_columns = len(numeric_columns)
    num_rows = (num_columns + 2) // 3

    plt.figure(figsize=(15, num_rows * 5))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, 3, i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        column_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers[col] = {
            'count': len(column_outliers),
            'percentage': (len(column_outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    plt.tight_layout()
    plt.show()
    return outliers


numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
outliers = detect_outliers_boxplot(df, numeric_columns)

print("\nOutlier Detection Summary:")
for col, info in outliers.items():
    print(f"{col}:")
    print(f"  Outlier Count: {info['count']}")
    print(f"  Outlier Percentage: {info['percentage']:.2f}%")
    print(f"  Lower Bound: {info['lower_bound']}")
    print(f"  Upper Bound: {info['upper_bound']}")

# ── Winsorization ─────────────────────────────────────────────────────────────
def winsorize_data(df, numeric_columns):
    df_winsorized = df.copy()
    print("\nOutlier Winsorization Summary:")

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_winsorized[col] = np.clip(df[col], lower_bound, upper_bound)

        original_min = df[col].min()
        original_max = df[col].max()
        winsorized_min = df_winsorized[col].min()
        winsorized_max = df_winsorized[col].max()

        print(f"\n{col}:")
        print(f"  Original range:   [{original_min}, {original_max}]")
        print(f"  Winsorized range: [{winsorized_min}, {winsorized_max}]")

    return df_winsorized


df_winsorized = winsorize_data(df, numeric_columns)

# ── Distribution: original vs winsorized ──────────────────────────────────────
num_rows = (len(numeric_columns) + 1) // 2
plt.figure(figsize=(15, num_rows * 5))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, 2, i)
    sns.histplot(df[col], color='blue', alpha=0.5, label='Original', bins=30, kde=True)
    sns.histplot(df_winsorized[col], color='orange', alpha=0.5, label='Winsorized', bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.legend()

plt.tight_layout()
plt.show()

# ── Before/after boxplots for columns that had outliers ───────────────────────
features_with_outliers = ['Waist Circumference', 'Pulmonary Function']
num_features = len(features_with_outliers)

plt.figure(figsize=(10, num_features * 6))
for i, col in enumerate(features_with_outliers):
    plt.subplot(num_features, 2, 2 * i + 1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col} (Before Winsorization)')

    plt.subplot(num_features, 2, 2 * i + 2)
    sns.boxplot(x=df_winsorized[col])
    plt.title(f'Boxplot of {col} (After Winsorization)')

plt.tight_layout()
plt.show()