
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

    print(
        df[df["correct"]]
        .head(5)[["image", "true_class", "predicted_class"]]
        .to_string(index=False)
    )

    # Representative incorrect predictions
    print("\nREPRESENTATIVE INCORRECT PREDICTIONS")
    print("------------------------------------")

    print(
        errors.head(5)[["image", "true_class", "predicted_class"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    for model_name in MODELS:
        analyze_model(model_name)
