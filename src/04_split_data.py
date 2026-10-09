from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Read the prepared training candidates.
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

df = pd.read_csv(
    data_folder / "expense_training_data.csv"
)

# 2. Count distinct descriptions in each category.
description_counts = (
    df.groupby("category_clean")["description_clean"].nunique()
)

# A category needs at least two descriptions:
# one for training and one for testing.
supported_categories = description_counts[
    description_counts >= 2
].index

supported_data = df.loc[
    df["category_clean"].isin(supported_categories)
].copy()

unsupported_data = df.loc[
    ~df["category_clean"].isin(supported_categories)
].copy()

# 3. Split descriptions separately within each category.
train_parts = []
test_parts = []

for category, category_rows in supported_data.groupby("category_clean"):
    descriptions = sorted(
        category_rows["description_clean"].unique()
    )

    train_descriptions, test_descriptions = train_test_split(
        descriptions,
        test_size=0.20,
        random_state=42
    )

    train_parts.append(
        category_rows.loc[
            category_rows["description_clean"].isin(train_descriptions)
        ]
    )

    test_parts.append(
        category_rows.loc[
            category_rows["description_clean"].isin(test_descriptions)
        ]
    )

# 4. Combine the categories into two datasets.
train_data = pd.concat(train_parts, ignore_index=True)
test_data = pd.concat(test_parts, ignore_index=True)

# 5. Verify that no exact cleaned description appears on both sides.
overlap = (
    set(train_data["description_clean"])
    & set(test_data["description_clean"])
)

if overlap:
    raise ValueError("Some descriptions appear in both datasets.")

# 6. Save the datasets.
train_data.to_csv(data_folder / "train.csv", index=False)
test_data.to_csv(data_folder / "test.csv", index=False)

unsupported_data.to_csv(
    data_folder / "unsupported_training_categories.csv",
    index=False
)

# 7. Print the results.
print("\n--- Split summary ---")
print("Training transactions:", len(train_data))
print("Testing transactions:", len(test_data))
print("Unsupported transactions:", len(unsupported_data))
print("Descriptions appearing on both sides:", len(overlap))

summary = pd.DataFrame({
    "train_rows": train_data.groupby("category_clean").size(),
    "test_rows": test_data.groupby("category_clean").size(),
    "train_descriptions": (
        train_data.groupby("category_clean")["description_clean"].nunique()
    ),
    "test_descriptions": (
        test_data.groupby("category_clean")["description_clean"].nunique()
    )
}).fillna(0).astype(int)

print("\n--- Split by category ---")
print(summary.to_string())

print("\n--- Unsupported categories ---")
print(unsupported_data["category_clean"].value_counts().to_string())