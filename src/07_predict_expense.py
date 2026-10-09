from pathlib import Path
import joblib
import pandas as pd

# Locate and load the saved pipeline.
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "expense_classifier.joblib"

model = joblib.load(model_path)

print("Expense category suggestions")
print("Enter a debit expense description, or type 'exit' to stop.")
print("Every suggestion needs confirmation.")

while True:
    description = input("\nTransaction description: ").strip()

    if description.lower() == "exit":
        break

    if not description:
        print("Please enter a description.")
        continue

    # Apply the same description cleaning used during preparation.
    cleaned_description = " ".join(description.upper().split())

    probabilities = model.predict_proba([cleaned_description])[0]

    suggestions = pd.DataFrame({
        "category": model.classes_,
        "model_score": probabilities
    }).sort_values("model_score", ascending=False)

    print("\nTop 3 suggestions:")
    print(
        suggestions.head(3).to_string(
            index=False,
            formatters={"model_score": "{:.1%}".format}
        )
    )

    print("\nReview required: confirm the category yourself.")