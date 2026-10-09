from pathlib import Path
from io import BytesIO
import hashlib
import json
import sys

import joblib
import pandas as pd
import streamlit as st


# 1. Configure the page and locate project files.
st.set_page_config(
    page_title="Personal Expense Classifier",
    page_icon="💳"
)

project_folder = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(project_folder))
from src.transaction_utils import prepare_transactions

model_path = project_folder / "models" / "expense_classifier.joblib"
corrections_path = project_folder / "data" / "category_corrections.json"

corrections_path.parent.mkdir(exist_ok=True)

REVIEW_CATEGORY = "Needs review / Outside model scope"
INCOMING_CATEGORY = "Incoming money"


# 2. Helper functions.
def clean_description(description):
    return " ".join(str(description).upper().split())


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def load_corrections():
    if not corrections_path.exists():
        return {}

    corrections = json.loads(
        corrections_path.read_text(encoding="utf-8")
    )

    if not isinstance(corrections, dict):
        raise ValueError("The corrections file must contain a dictionary.")

    return corrections


def save_corrections(corrections):
    temporary_path = corrections_path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(corrections, indent=2),
        encoding="utf-8"
    )

    temporary_path.replace(corrections_path)


# 3. Load the model and saved corrections.
if not model_path.exists():
    st.error("Model file is missing. Run src/05_train_model.py first.")
    st.stop()

model = load_model(model_path)

try:
    corrections = load_corrections()
except (OSError, ValueError) as error:
    st.error(f"Could not read saved corrections: {error}")
    st.stop()

model_categories = sorted(model.classes_.tolist())
category_options = model_categories + [REVIEW_CATEGORY]


# 4. Initialise session storage.
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confirmed_transactions" not in st.session_state:
    st.session_state.confirmed_transactions = []

if "csv_generation" not in st.session_state:
    st.session_state.csv_generation = 0


# 5. Introduction.
st.title("Personal Expense Classifier")

st.write(
    "Review suggested expense categories and save your corrections."
)

st.caption(
    "Trained on historical UK transactions. Indian merchant and UPI "
    "accuracy has not been evaluated. Savings, investments and cash "
    "withdrawals are outside this model's scope."
)

st.caption(
    "Saved categories are reused for matching debit descriptions. "
    "They do not retrain the model. This version is for local, single-user use."
)


# 6. Predict one outgoing expense.
st.subheader("Enter one expense")

with st.form("prediction_form"):
    description = st.text_input(
        "Transaction description",
        placeholder="For example: TALKTALK LIMITED"
    )

    submitted = st.form_submit_button("Suggest category")

if submitted:
    if not description.strip():
        st.session_state.prediction = None
        st.warning("Enter a transaction description.")
    else:
        description_key = clean_description(description)

        probabilities = model.predict_proba([description_key])[0]

        suggestions = pd.DataFrame({
            "Category": model.classes_,
            "Model score": probabilities
        }).sort_values("Model score", ascending=False)

        saved_category = corrections.get(description_key)

        if saved_category not in model_categories:
            saved_category = None

        default_category = (
            saved_category
            if saved_category is not None
            else suggestions.iloc[0]["Category"]
        )

        st.session_state.prediction = {
            "description": description.strip(),
            "description_key": description_key,
            "suggestions": suggestions.head(3).copy(),
            "saved_category": saved_category
        }

        st.session_state.selected_category = default_category

prediction = st.session_state.prediction

if prediction is not None:
    st.write("Description:", prediction["description"])

    if prediction["saved_category"] is not None:
        st.info(
            f"Previously confirmed category: "
            f"{prediction['saved_category']}"
        )

    display_table = prediction["suggestions"].copy()

    display_table["Model score"] = display_table["Model score"].map(
        lambda score: f"{score:.1%}"
    )

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Model scores are not verified probabilities of correctness. "
        "Confirm or change the category."
    )

    selected_category = st.selectbox(
        "Choose the category",
        options=category_options,
        key="selected_category"
    )

    if st.button("Save confirmation", key="save_single"):
        save_succeeded = True

        if selected_category in model_categories:
            try:
                updated_corrections = load_corrections()

                updated_corrections[
                    prediction["description_key"]
                ] = selected_category

                save_corrections(updated_corrections)
            except (OSError, ValueError) as error:
                st.error(f"Could not save the category: {error}")
                save_succeeded = False

        if save_succeeded:
            st.session_state.confirmed_transactions.append({
                "description": prediction["description"],
                "suggested_category": (
                    prediction["suggestions"].iloc[0]["Category"]
                ),
                "confirmed_category": selected_category
            })

            st.session_state.prediction = None
            st.rerun()


# 7. Display and download session confirmations.
st.subheader("Session confirmations")

if st.session_state.confirmed_transactions:
    confirmed_df = pd.DataFrame(
        st.session_state.confirmed_transactions
    )

    st.dataframe(
        confirmed_df,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download confirmations",
        data=confirmed_df.to_csv(index=False).encode("utf-8"),
        file_name="expense_confirmations.csv",
        mime="text/csv",
        key="download_single"
    )
else:
    st.write("No confirmations saved in this session.")

st.caption(
    "Supported category choices are remembered on this computer. "
    "The session table resets on refresh; download it to keep its history."
)


# 8. Upload transactions with dates, debits and credits.
st.divider()
st.subheader("Upload transactions")

st.caption(
    "Use separate debit and credit columns with plain numeric amounts. "
    "Incoming money is kept separate from expense predictions."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    try:
        uploaded_df = pd.read_csv(BytesIO(file_bytes))
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError
    ):
        st.error("Could not read this CSV. Use a valid UTF-8 CSV file.")
        st.stop()

    if uploaded_df.empty:
        st.warning("The CSV contains no transactions.")
        st.stop()

    if len(uploaded_df.columns) < 4:
        st.warning(
            "This upload needs description, date, debit and credit columns."
        )
        st.stop()

    st.write("Preview")

    st.dataframe(
        uploaded_df.head(),
        hide_index=True,
        use_container_width=True
    )

    column_names = uploaded_df.columns.tolist()

    description_column = st.selectbox(
        "Description column",
        column_names,
        index=0
    )

    date_column = st.selectbox(
        "Date column",
        column_names,
        index=1
    )

    debit_column = st.selectbox(
        "Debit column",
        column_names,
        index=2
    )

    credit_column = st.selectbox(
        "Credit column",
        column_names,
        index=3
    )

    date_format = st.selectbox(
        "Date format",
        ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"],
        help="Day/month/year, year-month-day, or month/day/year."
    )

    selected_columns = [
        description_column,
        date_column,
        debit_column,
        credit_column
    ]

    upload_key = json.dumps([
        file_id,
        selected_columns,
        date_format
    ])

    if st.button("Validate and suggest categories"):
        if len(set(selected_columns)) != 4:
            st.error("Select four different columns.")
        else:
            prepared = prepare_transactions(
                uploaded_df,
                description_column,
                date_column,
                debit_column,
                credit_column,
                date_format
            )

            review_df = pd.DataFrame({
                "row_number": range(1, len(prepared) + 1),
                "date": prepared["date_clean"].dt.strftime("%Y-%m-%d"),
                "description": (
                    uploaded_df[description_column]
                    .astype("string")
                    .fillna("")
                ),
                "direction": prepared["direction"],
                "amount": prepared["amount"],
                "validation_issue": prepared["validation_issue"],
                "suggested_category": REVIEW_CATEGORY,
                "model_score": 0.0,
                "confirmed_category": REVIEW_CATEGORY,
                "category_source": "Needs review",
                "reviewed": False
            })

            # Credits are incoming money, not expense predictions.
            valid_credit = (
                prepared["valid_transaction"]
                & prepared["direction"].eq("Credit")
            )

            review_df.loc[
                valid_credit, "suggested_category"
            ] = INCOMING_CATEGORY

            review_df.loc[
                valid_credit, "confirmed_category"
            ] = INCOMING_CATEGORY

            review_df.loc[
                valid_credit, "category_source"
            ] = "Credit direction"

            # Predict only valid debit transactions.
            valid_debit = (
                prepared["valid_transaction"]
                & prepared["direction"].eq("Debit")
            )

            if valid_debit.any():
                probabilities = model.predict_proba(
                    prepared.loc[valid_debit, "description_clean"]
                )

                review_df.loc[
                    valid_debit, "suggested_category"
                ] = model.classes_[probabilities.argmax(axis=1)]

                review_df.loc[
                    valid_debit, "model_score"
                ] = probabilities.max(axis=1)

                review_df.loc[
                    valid_debit, "confirmed_category"
                ] = review_df.loc[valid_debit, "suggested_category"]

                review_df.loc[
                    valid_debit, "category_source"
                ] = "Model suggestion"

                # Reuse saved corrections for valid debits.
                for row_index in prepared.index[valid_debit]:
                    description_key = prepared.at[
                        row_index, "description_clean"
                    ]

                    saved_category = corrections.get(description_key)

                    if saved_category in category_options:
                        review_df.at[
                            row_index, "confirmed_category"
                        ] = saved_category

                        review_df.at[
                            row_index, "category_source"
                        ] = "Saved confirmation"

            st.session_state.csv_review = review_df
            st.session_state.csv_upload_key = upload_key
            st.session_state.csv_generation += 1

    # 9. Review, save corrections and export.
    if st.session_state.get("csv_upload_key") == upload_key:
        review_df = st.session_state.csv_review

        invalid_count = int(
            review_df["validation_issue"].ne("").sum()
        )

        st.write(f"Transactions needing data fixes: {invalid_count}")

        st.caption(
            "Incoming money may be salary, a refund or a transfer. "
            "Debits may also be transfers or savings; choose "
            "Needs review / Outside model scope where appropriate. "
            "Fix invalid source rows and upload again."
        )

        editor_key = (
            f"csv_editor_{upload_key}_"
            f"{st.session_state.csv_generation}"
        )

        edited_df = st.data_editor(
            review_df,
            hide_index=True,
            use_container_width=True,
            disabled=[
                "row_number",
                "date",
                "description",
                "direction",
                "amount",
                "validation_issue",
                "suggested_category",
                "model_score",
                "category_source"
            ],
            column_config={
                "confirmed_category": st.column_config.SelectboxColumn(
                    "Confirmed category",
                    options=category_options + [INCOMING_CATEGORY],
                    required=True
                ),
                "model_score": st.column_config.NumberColumn(
                    "Model score",
                    format="%.3f"
                ),
                "reviewed": st.column_config.CheckboxColumn(
                    "Reviewed"
                )
            },
            key=editor_key
        )

        st.caption(
            f"Reviewed: {int(edited_df['reviewed'].sum())} "
            f"of {len(edited_df)} transactions. "
            "Check saved categories too: matching descriptions "
            "can represent different purchases."
        )

        if st.button("Save reviewed categories for future uploads"):
            updates = {}
            conflicts = set()

            for _, row in edited_df.iterrows():
                description_key = clean_description(row["description"])
                category = row["confirmed_category"]

                eligible = (
                    bool(row["reviewed"])
                    and row["direction"] == "Debit"
                    and row["validation_issue"] == ""
                    and bool(description_key)
                    and category in category_options
                )

                if eligible:
                    if (
                        description_key in updates
                        and updates[description_key] != category
                    ):
                        conflicts.add(description_key)

                    updates[description_key] = category

            # Do not save contradictory matches from this upload.
            for description_key in conflicts:
                updates.pop(description_key, None)

            try:
                updated_corrections = load_corrections()
                updated_corrections.update(updates)
                save_corrections(updated_corrections)

                st.success(
                    f"Saved {len(updates)} description/category matches."
                )

                if conflicts:
                    st.warning(
                        "Conflicting categories for the same description "
                        "were not saved. Existing matches were unchanged."
                    )
            except (OSError, ValueError) as error:
                st.error(f"Could not save categories: {error}")

        st.download_button(
            "Download categorised CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="categorised_transactions.csv",
            mime="text/csv",
            key="download_csv"
        )

        st.caption(
            "The download includes all rows and their review status. "
            "Only reviewed, valid debit categories are remembered "
            "when you click Save reviewed categories."
        )

        # 10. Show spending for reviewed, valid debit expenses.
if uploaded_file is not None:
    if st.session_state.get("csv_upload_key") == upload_key:
        st.divider()
        st.subheader("Reviewed spending")

        expense_mask = (
            edited_df["reviewed"].fillna(False).astype(bool)
            & edited_df["direction"].eq("Debit")
            & edited_df["validation_issue"].eq("")
            & edited_df["confirmed_category"].isin(model_categories)
        )

        expenses = edited_df.loc[expense_mask].copy()

        st.caption(
            "Includes only reviewed, valid debit rows assigned an expense "
            "category. Amounts use the uploaded file's currency; upload "
            "one currency at a time. Refunds are not deducted."
        )

        if expenses.empty:
            st.info(
                "Review at least one valid debit expense to see spending."
            )
        else:
            # Round each transaction to two decimal places and sum
            # integer minor units to avoid floating-point total errors.
            expenses["amount_minor"] = (
                expenses["amount"].round(2).mul(100).round().astype("int64")
            )

            total_spending = expenses["amount_minor"].sum() / 100

            st.metric(
                "Reviewed expense total",
                f"{total_spending:,.2f}"
            )

            st.write(f"Included transactions: {len(expenses)}")

            category_totals = (
                expenses.groupby("confirmed_category")["amount_minor"]
                .sum()
                .div(100)
                .sort_values(ascending=False)
                .rename("Amount")
            )

            st.write("Spending by category")
            st.dataframe(category_totals.to_frame())
            st.bar_chart(category_totals)

            expenses["month"] = pd.to_datetime(
                expenses["date"],
                format="%Y-%m-%d"
            ).dt.strftime("%Y-%m")

            monthly_totals = (
                expenses.groupby("month")["amount_minor"]
                .sum()
                .div(100)
                .sort_index()
                .rename("Amount")
            )

            st.write("Spending by month")
            st.dataframe(monthly_totals.to_frame())
            st.bar_chart(monthly_totals)