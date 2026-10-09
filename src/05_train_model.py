from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Locate the project folders.
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"
models_folder = project_folder / "models"

models_folder.mkdir(exist_ok=True)

# 2. Read the training and testing datasets.
train_df = pd.read_csv(data_folder / "train.csv")
test_df = pd.read_csv(data_folder / "test.csv")

X_train = train_df["description_clean"]
y_train = train_df["category_clean"]

X_test = test_df["description_clean"]
y_test = test_df["category_clean"]

# 3. Create a simple baseline.
# It always predicts the most common training category.
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(train_df[["amount"]], y_train)
baseline_predictions = baseline.predict(test_df[["amount"]])

print("\n--- Baseline ---")
print(
    "Accuracy:",
    f"{accuracy_score(y_test, baseline_predictions):.2%}"
)

# 4. Create the text-classification pipeline.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# 5. Learn from training descriptions and their categories.
model.fit(X_train, y_train)

# 6. Predict categories for held-out descriptions.
predictions = model.predict(X_test)

print("\n--- Model accuracy ---")
print(f"{accuracy_score(y_test, predictions):.2%}")

print("\n--- Classification report ---")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# 7. Also evaluate each distinct test description once.
# This prevents frequently repeated descriptions dominating the result.
unique_test = test_df.drop_duplicates(subset=["description_clean"])

unique_predictions = model.predict(
    unique_test["description_clean"]
)

print("\n--- Accuracy on distinct test descriptions ---")
print(
    f"{accuracy_score(
        unique_test['category_clean'],
        unique_predictions
    ):.2%}"
)

# 8. Save predictions for examining mistakes.
results = test_df.copy()
results["predicted_category"] = predictions
results["correct"] = (
    results["category_clean"] == results["predicted_category"]
)

results.to_csv(
    data_folder / "test_predictions.csv",
    index=False
)

# 9. Save the fitted text converter and classifier together.
model_path = models_folder / "expense_classifier.joblib"
joblib.dump(model, model_path)

print("\nModel saved:", model_path)
print("Predictions saved:", data_folder / "test_predictions.csv")