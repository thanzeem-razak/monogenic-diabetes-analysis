import pandas as pd

# ── Option A: Load from MySQL (original setup) ────────────────────────────────
# Uncomment this block if you have access to the MySQL database.
#
# myvars = {}
# with open("_____-mysql-password") as myfile: #replaceusername
#     for line in myfile:
#         name, var = line.partition(":")[::2]
#         myvars[name.strip()] = var.strip()
#
# import MySQLdb
# conn = MySQLdb.connect(
#     host="localhost",
#     user=myvars['DB username'],
#     passwd=myvars['DB password'],
#     db=myvars['DB databasename']
# )
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM diabetes_012")
# column_names = [column[0] for column in cursor.description]
# rows = cursor.fetchall()
# df = pd.DataFrame(rows, columns=column_names)

# ── Option B: Load from CSV (recommended for local/GitHub use) ────────────────
# Download the dataset from:
# https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset/data
# Place it in the data/ folder as diabetes_012.csv

df = pd.read_csv("data/diabetes_012.csv")

# ── Basic overview ─────────────────────────────────────────────────────────────
print(df.info())
print(f"\nShape: {df.shape}  (Rows, Columns)")
print(f"Total Elements: {df.size}")

# ── Null value check ──────────────────────────────────────────────────────────
def check_null_values(df):
    null_count = df.isnull().sum()
    print("\nNull Value Count:")
    print(null_count)
    return null_count

check_null_values(df)