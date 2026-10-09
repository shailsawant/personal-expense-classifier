from pathlib import Path
import pandas as pd

# 1. Locate the input file and output folder.
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

file_path = data_folder / "laramee26openBankTransactionData.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Anonymized Original with Catego"
)

# 2. Create cleaned columns without changing the original columns.
df["category_clean"] = df["Category"].astype("string").str.strip()

df["description_clean"] = (
    df["Transaction Description"]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.replace(r"\s+", " ", regex=True)
)

df["date_clean"] = pd.to_datetime(
    df["Transaction Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

# 3. Identify debit and credit transactions.
has_debit = df["Debit Amount"].notna()
has_credit = df["Credit Amount"].notna()

df["direction"] = "Unknown"
df.loc[has_debit & ~has_credit, "direction"] = "Debit"
df.loc[has_credit & ~has_debit, "direction"] = "Credit"

# Use one positive amount column for either direction.
df["amount"] = df["Debit Amount"].fillna(df["Credit Amount"])

# 4. Find descriptions with multiple labels in the same direction.
label_counts = (
    df.dropna(subset=["category_clean"])
    .groupby(["description_clean", "direction"])["category_clean"]
    .nunique()
    .rename("label_count")
    .reset_index()
)

df = df.merge(
    label_counts,
    on=["description_clean", "direction"],
    how="left",
    validate="many_to_one"
)

df["conflicting_labels"] = df["label_count"].fillna(0) > 1

# 5. Find categories with fewer than 10 labelled debit records.
debit_category_counts = df.loc[
    df["direction"] == "Debit",
    "category_clean"
].value_counts()

rare_categories = debit_category_counts[
    debit_category_counts < 10
].index

# These categories are outside the first expense model's scope.
excluded_categories = [
    "Savings",
    "Investment",
    "Cash",
    "Account transfer",
    "Paycheck",
    "Interest",
    "Supplementary Income",
    "Travel Reimbursement",
    "Safety Deposit Return"
]

# 6. Record why a transaction is excluded from training.
# A row may have more than one reason.
df["exclusion_reason"] = ""


def add_reason(mask, reason):
    df.loc[mask, "exclusion_reason"] = (
        df.loc[mask, "exclusion_reason"] + reason + "; "
    )


add_reason(
    df["direction"] != "Debit",
    "Not a debit expense candidate"
)

add_reason(
    df["category_clean"].isna() | df["category_clean"].eq("").fillna(False),
    "Missing category"
)

add_reason(
    df["category_clean"].isin(excluded_categories),
    "Category outside expense model scope"
)

add_reason(
    df["category_clean"].isin(rare_categories),
    "Fewer than 10 debit examples in category"
)

add_reason(
    df["conflicting_labels"],
    "Same description and direction has multiple labels"
)

add_reason(
    df["description_clean"].isna()
    | df["description_clean"].eq("").fillna(False),
    "Missing description"
)

add_reason(
    df["date_clean"].isna(),
    "Invalid date"
)

add_reason(
    df["amount"].isna() | (df["amount"] <= 0),
    "Invalid amount"
)

df["exclusion_reason"] = df["exclusion_reason"].str.rstrip("; ")
df["training_eligible"] = df["exclusion_reason"].eq("")

# 7. Separate eligible examples from excluded records.
training_data = df.loc[df["training_eligible"]].copy()
excluded_data = df.loc[~df["training_eligible"]].copy()

# 8. Save all transactions and the two subsets.
# The original Excel file remains unchanged.
df.to_csv(
    data_folder / "transactions_prepared.csv",
    index=False
)

training_data.to_csv(
    data_folder / "expense_training_data.csv",
    index=False
)

excluded_data.to_csv(
    data_folder / "transactions_excluded.csv",
    index=False
)

# 9. Print the preparation results.
print("\n--- Preparation summary ---")
print("Total transactions:", len(df))
print("Eligible for expense training:", len(training_data))
print("Excluded from training:", len(excluded_data))

print("\n--- Training examples per category ---")
print(training_data["category_clean"].value_counts().to_string())

print("\n--- Distinct descriptions per training category ---")
print(
    training_data.groupby("category_clean")["description_clean"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)

print("\n--- Exclusion reason combinations ---")
print(excluded_data["exclusion_reason"].value_counts().to_string())

print("\nSaved:")
print("data/transactions_prepared.csv")
print("data/expense_training_data.csv")
print("data/transactions_excluded.csv")