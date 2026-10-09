from pathlib import Path
import pandas as pd

# Locate the Excel file inside the project's data folder.
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "laramee26openBankTransactionData.xlsx"

# Open the workbook.
workbook = pd.ExcelFile(file_path)
print("Sheet names:", workbook.sheet_names)

# Read the transaction data.
df = pd.read_excel(
    workbook,
    sheet_name="Anonymized Original with Catego"
)

print("\n--- Dataset size ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- First 5 transactions ---")
print(df.head().to_string(index=False))

print("\n--- Column data types ---")
print(df.dtypes)

print("\n--- Missing values per column ---")
print(df.isna().sum())

print("\n--- Transactions per category ---")
print(df["Category"].value_counts(dropna=False).to_string())

# Check whether each transaction has a debit or credit amount.
has_debit = df["Debit Amount"].notna()
has_credit = df["Credit Amount"].notna()

print("\n--- Debit and credit checks ---")
print("Debit only:", (has_debit & ~has_credit).sum())
print("Credit only:", (~has_debit & has_credit).sum())
print("Both amounts present:", (has_debit & has_credit).sum())
print("Neither amount present:", (~has_debit & ~has_credit).sum())

print("\n--- Unique transaction descriptions ---")
print(df["Transaction Description"].nunique())

# This sheet starts with a citation, rather than column headings.
# header=None preserves every row as data.
definitions = pd.read_excel(
    workbook,
    sheet_name="Description of Categories",
    header=None
)

print("\n--- Category definitions ---")
print(
    definitions.dropna(how="all").to_string(
        index=False,
        header=False
    )
)

workbook.close()