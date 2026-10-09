# Personal Expense Classifier

A machine-learning prototype that suggests expense categories from bank
transaction descriptions. Users review predictions, correct categories,
and explore spending from confirmed transactions.

Built with Python, pandas, scikit-learn and Streamlit.

## Live demo

https://personal-expense-classifier.streamlit.app/

The public demo stores corrections only within the current browser session.
Refresh clears them. Download reviewed results before leaving.

## Features

- Single-transaction category suggestions with the top three model scores.
- CSV uploads with description, date, debit and credit column mapping.
- Validation of dates, missing descriptions and invalid amounts.
- Separation of incoming money from outgoing transactions.
- Editable categories and explicit review checkboxes.
- Local storage of confirmed categories for matching debit descriptions.
- Support for marking transactions outside the expense model's scope.
- Spending totals and charts for reviewed, valid debit expenses.
- CSV export containing predictions, corrections and review status.

## Dataset

This project uses MoneyData, a real anonymised retail bank transaction
dataset published by Robert Laramee.

Source: https://data.mendeley.com/datasets/dnxtg6n4rv/1

Dataset licence: CC BY 4.0.

The downloaded workbook contains 6,567 transactions. Our inspection found
908 distinct original transaction descriptions and 24 missing category
labels.

Required attribution:

Elif E Firat, Dharmateja Vytia, Navya Vasudeva, Zhuoqun Jiang,
Robert S Laramee. *MoneyVis: Open Bank Transaction Data for Visualization
and Beyond*. EuroVis Short Papers, 2023.
https://doi.org/10.2312/evs.20231052

The dataset is downloaded separately and is not included in this repository.
The repository's MIT licence applies to the project code, not the dataset.

## Data preparation

The preparation script preserves every original transaction and adds
cleaned descriptions, trimmed categories and training eligibility flags.

The first expense model excludes:

- Incoming transactions.
- Savings, investments, cash withdrawals and other categories outside scope.
- Missing category labels.
- Description/direction groups containing conflicting category labels.
- Categories with fewer than 10 labelled debit examples.
- Invalid dates, descriptions or amounts.

This produces 3,589 training candidates. Another 34 transactions in
`Services/Home Improvement` are excluded from the split because they share
only one distinct description.

The remaining 3,555 transactions cover 15 categories.

Exclusion means omission from model training, not deletion from the
prepared dataset.

## Model

The model is a scikit-learn pipeline containing:

1. Character TF-IDF features using sequences of 3–5 characters within words.
2. Logistic regression with balanced class weights.

The first model uses transaction descriptions only. Amounts and dates are
used for validation and reporting, not category prediction.

Training and testing are split by distinct cleaned description within each
category. No identical cleaned description appears on both sides.

Different descriptions can still refer to the same merchant, so this is
an unseen-description evaluation, not an unseen-merchant evaluation.

## Initial evaluation

| Measure | Result |
|---|---:|
| Training transactions | 2,484 |
| Test transactions | 1,071 |
| Most-frequent-training-category baseline accuracy | 6.07% |
| Model transaction accuracy | 71.43% |
| Accuracy on distinct test descriptions | 63.82% |
| Macro F1, rounded | 0.56 |
| Bills recall | 0% |

These are initial results on a curated subset of the dataset.

Amazon accounts for 450 test transactions and was classified correctly
in every case. This strongly influences transaction accuracy.

Insurance and Mortgage each have only one distinct test description.
Their results provide limited evidence about generalisation.

The test errors have been inspected during development. Future model
selection requires validation data and a fresh final evaluation.

## Project structure

| Path | Purpose |
|---|---|
| `dashboard/app.py` | Streamlit interface |
| `src/01_inspect_data.py` | Inspect workbook sheets and columns |
| `src/02_audit_data.py` | Audit labels, amounts and ambiguous descriptions |
| `src/03_prepare_data.py` | Prepare transactions and training candidates |
| `src/04_split_data.py` | Split by distinct description |
| `src/05_train_model.py` | Train, evaluate and save the model |
| `src/06_inspect_predictions.py` | Inspect classification errors |
| `src/07_predict_expense.py` | Interactive terminal predictions |
| `src/08_export_upload_sample.py` | Export 20 real transactions for upload testing |
| `src/transaction_utils.py` | Validate uploaded transaction columns |
| `tests/test_transaction_utils.py` | Transaction validation regression test |
| `requirements.txt` | Python dependencies |
| `data/` | Local datasets, exports and saved corrections; ignored by Git |
| `models/` | Saved model pipeline; ignored by Git |

## Local setup on Windows Command Prompt

Clone the repository:

```cmd
git clone https://github.com/shailsawant/personal-expense-classifier.git
cd personal-expense-classifier
```

Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Install dependencies and create local folders:

```cmd
python -m pip install -r requirements.txt
mkdir data
mkdir models
```

Download the MoneyData workbook and place it at:

```text
data/laramee26openBankTransactionData.xlsx
```

Prepare the data and train the model:

```cmd
python src/01_inspect_data.py
python src/02_audit_data.py
python src/03_prepare_data.py
python src/04_split_data.py
python src/05_train_model.py
```

Launch the dashboard:

```cmd
python -m streamlit run dashboard/app.py
```

## CSV upload format

The dashboard accepts UTF-8 CSV files with separate description, date,
debit and credit columns. Column names can be mapped in the interface.

Example synthetic input:

```csv
description,date,debit,credit
GROCERY PURCHASE,01/10/2026,25.50,
PHONE BILL,02/10/2026,30,
SALARY PAYMENT,03/10/2026,,2000
```

Use plain numeric amounts without currency symbols or thousands separators.
Blank or zero opposite amounts are accepted.

Supported date formats:

- `%d/%m/%Y` — day/month/year.
- `%Y-%m-%d` — year-month-day.
- `%m/%d/%Y` — month/day/year.

Rows with positive debit and credit amounts together, negative or unreadable
amounts, invalid dates or missing descriptions are flagged for review.

## Review workflow

1. Upload a CSV and map its columns.
2. Click **Validate and suggest categories**.
3. Check each transaction and correct its confirmed category.
4. Mark savings, investments, transfers and cash withdrawals as
   **Needs review / Outside model scope** where appropriate.
5. Tick **Reviewed** after checking a row.
6. Click **Save reviewed categories for future uploads** to remember
   supported category choices and outside-scope choices for valid debits.
7. Download the categorised CSV.

Saved matches use the cleaned description. A matching description can
represent different purchases, so reused categories still need review.

Corrections are stored locally in:

```text
data/category_corrections.json
```

Saving corrections does not retrain the model. Review flags reset on a
new upload. Downloaded exports preserve the review status of that export.

## Spending calculations

Spending includes only rows that are:

- Marked Reviewed.
- Valid according to transaction validation.
- Debits.
- Assigned one of the model's expense categories.

Incoming money and outside-scope transactions are excluded.

Totals represent reviewed expenses only, not necessarily all spending.
Amounts are rounded to two decimal places for reporting. Upload one
currency at a time; no currency conversion is performed.

Refunds are not deducted from spending.

## Tests

Run the validation regression test:

```cmd
python -m unittest discover -s tests -v
```

The test covers blank opposite amounts, valid debits and credits, both
amounts being positive, unreadable amounts, negative amounts and invalid dates.

## Limitations

- This is a local, single-user prototype.
- Historical UK data does not establish accuracy on Indian merchants or UPI.
- Unfamiliar names and vague descriptions can produce wrong predictions.
- Model scores are not calibrated probabilities of correctness.
- The model cannot automatically recognise every transaction outside scope.
- The Amazon category identifies a merchant, not the purchased product.
- Incoming money may be income, a refund or a transfer.
- Saved corrections are shared by users of the same local installation;
  separate user storage is required before multi-user deployment.
- Local JSON storage is not designed for concurrent writers.
- Personal bank statements and saved corrections must not be committed.

## Author

Shailendra Sawant  
GitHub: https://github.com/shailsawant