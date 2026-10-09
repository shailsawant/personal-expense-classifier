from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

df = pd.read_excel(
    data_folder / "laramee26openBankTransactionData.xlsx",
    sheet_name="Anonymized Original with Catego"
)

columns = [
    "Transaction Description",
    "Transaction Date",
    "Debit Amount",
    "Credit Amount",
    "Category"
]

sample = df[columns].head(20)

output_path = data_folder / "moneydata_upload_sample.csv"
sample.to_csv(output_path, index=False)

print("Exported transactions:", len(sample))
print("Saved:", output_path)