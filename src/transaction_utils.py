import numpy as np
import pandas as pd


def prepare_transactions(
    df,
    description_column,
    date_column,
    debit_column,
    credit_column,
    date_format="%d/%m/%Y"
):
    result = df.copy()

    result["description_clean"] = (
        df[description_column]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

    result["date_clean"] = pd.to_datetime(
        df[date_column],
        format=date_format,
        errors="coerce"
    )

    debit_text = (
        df[debit_column].astype("string").fillna("").str.strip()
    )
    credit_text = (
        df[credit_column].astype("string").fillna("").str.strip()
    )

    # Use ordinary float arrays: missing numbers become np.nan.
    debit = pd.Series(
        pd.to_numeric(debit_text, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan
        ),
        index=df.index
    )

    credit = pd.Series(
        pd.to_numeric(credit_text, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan
        ),
        index=df.index
    )

    # Convert text checks to ordinary True/False values too.
    debit_entered = debit_text.ne("").to_numpy(dtype=bool)
    credit_entered = credit_text.ne("").to_numpy(dtype=bool)

    invalid_amount = (
        (debit_entered & debit.isna())
        | (credit_entered & credit.isna())
        | (debit < 0)
        | (credit < 0)
        | (debit.notna() & ~np.isfinite(debit))
        | (credit.notna() & ~np.isfinite(credit))
    )

    has_debit = debit.fillna(0) > 0
    has_credit = credit.fillna(0) > 0

    result["direction"] = "Needs review"

    valid_debit = has_debit & ~has_credit & ~invalid_amount
    valid_credit = has_credit & ~has_debit & ~invalid_amount

    result.loc[valid_debit, "direction"] = "Debit"
    result.loc[valid_credit, "direction"] = "Credit"

    result["amount"] = np.nan
    result.loc[valid_debit, "amount"] = debit.loc[valid_debit]
    result.loc[valid_credit, "amount"] = credit.loc[valid_credit]

    result["validation_issue"] = ""

    def flag(mask, message):
        # Every selection mask must contain only True or False.
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)

        result.loc[mask, "validation_issue"] = (
            result.loc[mask, "validation_issue"] + message + "; "
        )

    flag(invalid_amount, "Invalid, negative or non-finite amount")
    flag(has_debit & has_credit, "Both debit and credit are positive")
    flag(~has_debit & ~has_credit, "No positive amount")
    flag(result["date_clean"].isna(), "Invalid or missing date")
    flag(result["description_clean"].eq(""), "Missing description")

    result["validation_issue"] = (
        result["validation_issue"].str.rstrip("; ")
    )

    result["valid_transaction"] = result["validation_issue"].eq("")

    return result