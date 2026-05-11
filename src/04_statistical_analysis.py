import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, kruskal, chi2_contingency

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/diabetes_012.csv")

numeric_columns = ['Insulin Levels', 'BMI', 'Blood Pressure',
                   'Blood Glucose Levels', 'Cholesterol Levels', 'Birth Weight']

categorical_vars = ['Glucose Tolerance Test', 'Liver Function Tests',
                    'Urine Test', 'Early Onset Symptoms']

df_filtered = df[df['Target'].isin(['MODY', 'Neonatal Diabetes Mellitus (NDM)'])]

# ── Normality: Shapiro-Wilk (note: unreliable for n > 5000) ──────────────────
print("Shapiro-Wilk Test for Normality (informational — dataset exceeds 5000 rows):\n")
for col in numeric_columns:
    try:
        stat, p = shapiro(df_filtered[col])
        print(f"{col}: W-statistic = {stat:.4f}, p-value = {p:.4f}")
    except ValueError as e:
        print(f"{col}: ERROR - {e}")

# ── Normality: Q-Q plots (primary method used) ────────────────────────────────
print("\nGenerating Q-Q plots for visual normality assessment...")
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    stats.probplot(df_filtered[column], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {column}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ── Hypothesis testing: Kruskal-Wallis (numerical variables) ─────────────────
print("\nMODY Hypothesis Testing Results (Kruskal-Wallis H Test):")
for col in numeric_columns:
    group_data = df[df['Target'] == 'MODY'][col]
    rest_data  = df[df['Target'] != 'MODY'][col]

    if len(group_data) == 0 or len(rest_data) == 0:
        print(f"Skipping MODY - {col}: One of the groups is empty.")
        continue

    test_stat, p_val = kruskal(group_data, rest_data)
    print(f"\nTesting for: MODY - {col}")
    print(f"  Kruskal-Wallis H Statistic: {test_stat:.2f},  p-value: {p_val:.5f}")

print("\nNeonatal Diabetes Mellitus (NDM) Hypothesis Testing Results (Kruskal-Wallis H Test):")
for col in numeric_columns:
    group_data = df[df['Target'] == 'Neonatal Diabetes Mellitus (NDM)'][col]
    rest_data  = df[df['Target'] != 'Neonatal Diabetes Mellitus (NDM)'][col]

    if len(group_data) == 0 or len(rest_data) == 0:
        print(f"Skipping NDM - {col}: One of the groups is empty.")
        continue

    test_stat, p_val = kruskal(group_data, rest_data)
    print(f"\nTesting for: NDM - {col}")
    print(f"  Kruskal-Wallis H Statistic: {test_stat:.2f},  p-value: {p_val:.5f}")

# ── Hypothesis testing: Chi-square (categorical variables) ───────────────────
print("\nMODY Hypothesis Testing Results (Chi-squared test):")
for col in categorical_vars:
    group_data = df[df['Target'] == 'MODY'][col].value_counts()
    rest_data  = df[df['Target'] != 'MODY'][col].value_counts()

    if group_data.empty or rest_data.empty:
        print(f"Skipping MODY - {col}: One of the groups is empty.")
        continue

    contingency_table = pd.DataFrame({'Group': group_data, 'Rest': rest_data}).fillna(0)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    print(f"\nTesting for: MODY - {col}")
    print(f"  Chi-Squared Statistic: {chi2:.2f},  p-value: {p_val:.5f}")

print("\nNeonatal Diabetes Mellitus (NDM) Hypothesis Testing Results (Chi-squared test):")
for col in categorical_vars:
    group_data = df[df['Target'] == 'Neonatal Diabetes Mellitus (NDM)'][col].value_counts()
    rest_data  = df[df['Target'] != 'Neonatal Diabetes Mellitus (NDM)'][col].value_counts()

    if group_data.empty or rest_data.empty:
        print(f"Skipping NDM - {col}: One of the groups is empty.")
        continue

    contingency_table = pd.DataFrame({'Group': group_data, 'Rest': rest_data}).fillna(0)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    print(f"\nTesting for: NDM - {col}")
    print(f"  Chi-Squared Statistic: {chi2:.2f},  p-value: {p_val:.5f}")