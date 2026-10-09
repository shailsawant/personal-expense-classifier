from pathlib import Path
import pandas as pd

# 1. Locate and read the Excel file.
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "laramee26openBankTransactionData.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Anonymized Original with Catego"
)

# 2. Remove surrounding spaces from categories.
# Keep the original Category column unchanged.
df["category_clean"] = df["Category"].astype("string").str.strip()

print("\n--- Original category labels ---")
# repr() makes surrounding spaces visible.
for category in sorted(df["Category"].dropna().unique()):
    print(repr(category))

print("\n--- Category counts after trimming spaces ---")
print(df["category_clean"].value_counts(dropna=False).to_string())

# 3. Identify transaction direction.
# Our previous inspection confirmed exactly one amount per row.
df["direction"] = "Debit"
df.loc[df["Credit Amount"].notna(), "direction"] = "Credit"

print("\n--- Debit and credit counts by category ---")
print(
    pd.crosstab(
        df["category_clean"].fillna("UNLABELLED"),
        df["direction"]
    ).to_string()
)

# 4. Check for zero or negative amounts.
print("\n--- Zero or negative amounts ---")
for column in ["Debit Amount", "Credit Amount"]:
    invalid = df[column].notna() & (df[column] <= 0)
    print(f"{column}: {invalid.sum()}")

# 5. Check transaction dates.
dates = pd.to_datetime(
    df["Transaction Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

print("\n--- Invalid dates ---")
print("Invalid or missing dates:", dates.isna().sum())
print("Earliest date:", dates.min())
print("Latest date:", dates.max())

# Columns to display when inspecting individual transactions.
columns = [
    "Transaction Description",
    "Debit Amount",
    "Credit Amount",
    "category_clean"
]

# 6. Inspect categories with fewer than 10 transactions.
category_counts = df["category_clean"].value_counts()
rare_categories = category_counts[category_counts < 10].index

print("\n--- Transactions with rare categories ---")
print(
    df.loc[
        df["category_clean"].isin(rare_categories),
        columns
    ].to_string(index=False)
)

# 7. Inspect transactions without category labels.
print("\n--- Transactions without categories ---")
print(
    df.loc[
        df["category_clean"].isna(),
        columns
    ].to_string(index=False)
)

# 8. Inspect outgoing transactions labelled as income.
print("\n--- Debits labelled Supplementary Income ---")
print(
    df.loc[
        (df["category_clean"] == "Supplementary Income")
        & (df["direction"] == "Debit"),
        columns
    ].to_string(index=False)
)

# 9. Inspect credits outside the dataset's income categories.
# These are records to examine, not automatically errors.
income_categories = [
    "Paycheck",
    "Interest",
    "Supplementary Income",
    "Travel Reimbursement"
]

print("\n--- Credits outside clear income categories ---")
print(
    df.loc[
        (df["direction"] == "Credit")
        & ~df["category_clean"].isin(income_categories),
        columns
    ].to_string(index=False)
)

# 10. Standardise descriptions for comparison.
# Strip surrounding spaces, use uppercase, and collapse repeated spaces.
# Keep the original Transaction Description column unchanged.
df["description_clean"] = (
    df["Transaction Description"]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.replace(r"\s+", " ", regex=True)
)

# 11. Count distinct category labels for each description and direction.
# Exclude unlabelled rows from the category count.
label_counts = (
    df.dropna(subset=["category_clean"])
    .groupby(["description_clean", "direction"])["category_clean"]
    .nunique()
)

# More than one label means the description/direction pair is ambiguous.
conflicting_groups = label_counts[label_counts > 1]

print("\n--- Description/direction groups with multiple labels ---")
print("Number of groups:", len(conflicting_groups))

for description, direction in conflicting_groups.index:
    matching_rows = df.loc[
        (df["description_clean"] == description)
        & (df["direction"] == direction)
    ]

    print(f"\nDescription: {description} | Direction: {direction}")
    print(
        matching_rows["category_clean"]
        .value_counts(dropna=False)
        .to_string()
    )