
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

MODELS = [
    "full_finetuning",
    "head_training",
    "feature_extraction_svm"
]


def analyze_model(model_name):
    path = PREDICTIONS_DIR / f"{model_name}_predictions.csv"
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(model_name.upper())
    print("=" * 60)

    # Most frequently confused directed class pairs
    errors = df[~df["correct"]]

    confusion_pairs = (
        errors.groupby(["true_class", "predicted_class"])
        .size()
        .sort_values(ascending=False)
    )

    print("\nTOP 10 CONFUSIONS")
    print("-----------------")

    print(confusion_pairs.head(10).to_string())

    # Representative correct predictions
    print("\nREPRESENTATIVE CORRECT PREDICTIONS")
    print("----------------------------------")

    correct_examples = (
        df[df["correct"]]
        .groupby("true_class", group_keys=False)
        .sample(n=1, random_state=42)
        .head(5)
    )

    print(
        correct_examples[
            ["image", "true_class", "predicted_class"]
        ].to_string(index=False)
    )

    # Representative incorrect predictions:
    # one example for each of the five most frequent confusion pairs
    print("\nREPRESENTATIVE INCORRECT PREDICTIONS")
    print("------------------------------------")

    top_pairs = confusion_pairs.head(5).index

    incorrect_examples = pd.concat([
        errors[
            (errors["true_class"] == true_class) &
            (errors["predicted_class"] == predicted_class)
        ].head(1)
        for true_class, predicted_class in top_pairs
    ])

    print(
        incorrect_examples[
            ["image", "true_class", "predicted_class"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    for model_name in MODELS:
        analyze_model(model_name)
