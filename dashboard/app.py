from pathlib import Path
from io import BytesIO
import hashlib
import json

import joblib
import pandas as pd
import streamlit as st


# 1. Page settings and file paths.
st.set_page_config(
    page_title="Personal Expense Classifier",
    page_icon="💳"
)

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "expense_classifier.joblib"
corrections_path = project_folder / "data" / "category_corrections.json"

corrections_path.parent.mkdir(exist_ok=True)

REVIEW_CATEGORY = "Needs review / Outside model scope"


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
    "Review suggested categories for outgoing expenses "
    "and save your corrections."
)

st.caption(
    "Trained on historical UK transactions. Indian merchant and UPI "
    "accuracy has not been evaluated. Savings, investments and cash "
    "withdrawals are outside this model's scope."
)

st.caption(
    "Saved categories are reused for matching descriptions. "
    "They do not retrain the model. This version is for local, single-user use."
)


# 6. Single-description prediction.
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
        cleaned_description = clean_description(description)

        probabilities = model.predict_proba([cleaned_description])[0]

        suggestions = pd.DataFrame({
            "Category": model.classes_,
            "Model score": probabilities
        }).sort_values("Model score", ascending=False)

        saved_category = corrections.get(cleaned_description)

        if saved_category not in model_categories:
            saved_category = None

        default_category = (
            saved_category
            if saved_category is not None
            else suggestions.iloc[0]["Category"]
        )

        st.session_state.prediction = {
            "description": description.strip(),
            "description_key": cleaned_description,
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
        "These are model suggestions. Scores are not verified "
        "probabilities of correctness."
    )

    selected_category = st.selectbox(
        "Choose the category",
        options=category_options,
        key="selected_category"
    )

    if st.button("Save confirmation", key="save_single"):
        save_succeeded = True

        # Remember supported categories for future matching descriptions.
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


# 7. Display confirmations from this session.
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


# 8. CSV upload.
st.divider()
st.subheader("Upload expense descriptions")

st.caption(
    "For this version, upload outgoing expenses only. "
    "Date, amount and debit/credit mapping will be added separately."
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

    st.write("Preview")

    st.dataframe(
        uploaded_df.head(),
        hide_index=True,
        use_container_width=True
    )

    description_column = st.selectbox(
        "Which column contains transaction descriptions?",
        uploaded_df.columns.tolist()
    )

    upload_key = f"{file_id}:{description_column}"

    if st.button("Suggest categories for CSV"):
        descriptions = (
            uploaded_df[description_column]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

        valid_description = descriptions.ne("")

        review_df = pd.DataFrame({
            "row_number": range(1, len(uploaded_df) + 1),
            "description": (
                uploaded_df[description_column]
                .astype("string")
                .fillna("")
            ),
            "suggested_category": REVIEW_CATEGORY,
            "model_score": 0.0,
            "confirmed_category": REVIEW_CATEGORY,
            "category_source": "Missing description",
            "reviewed": False
        })

        if valid_description.any():
            probabilities = model.predict_proba(
                descriptions.loc[valid_description]
            )

            predictions = model.classes_[probabilities.argmax(axis=1)]
            scores = probabilities.max(axis=1)

            review_df.loc[
                valid_description, "suggested_category"
            ] = predictions

            review_df.loc[
                valid_description, "model_score"
            ] = scores

            review_df.loc[
                valid_description, "confirmed_category"
            ] = predictions

            review_df.loc[
                valid_description, "category_source"
            ] = "Model suggestion"

        # Reuse saved categories without changing the model suggestion.
        for row_index in review_df.index:
            description_key = descriptions.iloc[row_index]
            saved_category = corrections.get(description_key)

            if description_key and saved_category in model_categories:
                review_df.at[
                    row_index, "confirmed_category"
                ] = saved_category

                review_df.at[
                    row_index, "category_source"
                ] = "Saved confirmation"

        st.session_state.csv_review = review_df
        st.session_state.csv_upload_key = upload_key

        # Reset the editor when predictions are generated again.
        st.session_state.csv_generation += 1

    if st.session_state.get("csv_upload_key") == upload_key:
        st.write(
            "Correct categories, then tick Reviewed for each row "
            "you have checked."
        )

        editor_key = (
            f"csv_editor_{upload_key}_"
            f"{st.session_state.csv_generation}"
        )

        edited_df = st.data_editor(
            st.session_state.csv_review,
            hide_index=True,
            use_container_width=True,
            disabled=[
                "row_number",
                "description",
                "suggested_category",
                "model_score",
                "category_source"
            ],
            column_config={
                "confirmed_category": st.column_config.SelectboxColumn(
                    "Confirmed category",
                    options=category_options,
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
            "Check saved categories too: the same description "
            "can represent different purchases."
        )

        if st.button("Save reviewed categories for future uploads"):
            updates = {}
            conflicting_descriptions = set()

            for _, row in edited_df.iterrows():
                description_key = clean_description(row["description"])
                category = row["confirmed_category"]

                if (
                    bool(row["reviewed"])
                    and description_key
                    and category in model_categories
                ):
                    if (
                        description_key in updates
                        and updates[description_key] != category
                    ):
                        conflicting_descriptions.add(description_key)

                    updates[description_key] = category

            # Avoid remembering ambiguous descriptions.
            for description_key in conflicting_descriptions:
                updates.pop(description_key, None)

            try:
                updated_corrections = load_corrections()
                updated_corrections.update(updates)
                save_corrections(updated_corrections)

                st.success(
                    f"Saved {len(updates)} description/category matches."
                )

                if conflicting_descriptions:
                    st.warning(
                        "Some descriptions had different reviewed categories "
                        "in this upload. Their saved matches were not updated."
                    )
            except (OSError, ValueError) as error:
                st.error(f"Could not save reviewed categories: {error}")

        st.download_button(
            "Download categorised CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="categorised_expenses.csv",
            mime="text/csv",
            key="download_csv"
        )

        st.caption(
            "Download keeps all rows, including unreviewed suggestions. "
            "Only reviewed, supported categories are remembered when "
            "you click Save reviewed categories."
        )