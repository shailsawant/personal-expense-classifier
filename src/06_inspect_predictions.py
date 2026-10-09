from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent

results = pd.read_csv(
    project_folder / "data" / "test_predictions.csv"
)

# Show each description once, with its transaction count.
summary = (
    results.groupby(
        ["description_clean", "category_clean", "predicted_category"]
    )
    .size()
    .reset_index(name="transaction_count")
)

mistakes = summary.loc[
    summary["category_clean"] != summary["predicted_category"]
].copy()

print("\n--- Bills predictions ---")
print(
    summary.loc[summary["category_clean"] == "Bills"]
    .sort_values("transaction_count", ascending=False)
    .to_string(index=False)
)

print("\n--- Most frequent incorrect predictions ---")
print(
    mistakes.sort_values("transaction_count", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\n--- Category confusion: transaction counts ---")
print(
    pd.crosstab(
        results["category_clean"],
        results["predicted_category"]
    ).to_string()
)