from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# 1. Configure the page.
st.set_page_config(
    page_title="Personal Expense Classifier",
    page_icon="💳"
)

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "expense_classifier.joblib"

# 2. Load the model once and reuse it.
@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not model_path.exists():
    st.error("Model file is missing. Run src/05_train_model.py first.")
    st.stop()

model = load_model(model_path)

# 3. Keep predictions and confirmations across screen updates.
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confirmed_transactions" not in st.session_state:
    st.session_state.confirmed_transactions = []

st.title("Personal Expense Classifier")
st.write("Enter an outgoing expense description and review its category.")
st.caption(
    "Trained on historical UK transactions. "
    "Indian merchant and UPI accuracy has not been evaluated. "
    "Savings, investments and cash withdrawals are outside this model's scope."
)

# 4. Collect a description.
with st.form("prediction_form"):
    description = st.text_input(
        "Transaction description",
        placeholder="For example: LIDL GB NOTTINGHA"
    )

    submitted = st.form_submit_button("Suggest category")

if submitted:
    if not description.strip():
        st.session_state.prediction = None
        st.warning("Enter a transaction description.")
    else:
        cleaned_description = " ".join(description.upper().split())

        probabilities = model.predict_proba([cleaned_description])[0]

        suggestions = pd.DataFrame({
            "Category": model.classes_,
            "Model score": probabilities
        }).sort_values("Model score", ascending=False)

        st.session_state.prediction = {
            "description": description.strip(),
            "suggestions": suggestions.head(3).copy()
        }

        # Default the selection to the newest prediction.
        st.session_state.selected_category = suggestions.iloc[0]["Category"]

# 5. Display suggestions and let the user correct them.
prediction = st.session_state.prediction

if prediction is not None:
    st.subheader("Review category")
    st.write("Description:", prediction["description"])

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

    category_options = sorted(model.classes_.tolist()) + [
        "Needs review / Outside model scope"
    ]

    selected_category = st.selectbox(
        "Choose the category",
        options=category_options,
        key="selected_category"
    )

    if st.button("Save confirmation"):
        st.session_state.confirmed_transactions.append({
            "description": prediction["description"],
            "suggested_category": prediction["suggestions"].iloc[0]["Category"],
            "confirmed_category": selected_category
        })

        st.session_state.prediction = None
        st.rerun()

# 6. Display and export the session's confirmations.
st.subheader("Saved confirmations")

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
        mime="text/csv"
    )
else:
    st.write("No confirmations saved yet.")

st.caption(
    "Confirmations remain in this browser session only. "
    "Download them before closing or refreshing the session."
)