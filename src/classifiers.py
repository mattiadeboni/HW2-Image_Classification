
import time
import joblib
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = PROJECT_ROOT / "results" / "features" / "pet_features.npz"


def train_linear_svm():
    # Load precomputed training and validation features.
    data = np.load(FEATURE_PATH)

    X_train = data["train_features"]
    y_train = data["train_labels"]

    X_val = data["val_features"]
    y_val = data["val_labels"]

    # Try several values of C to control the SVM regularization strength.
    c_values = [0.01, 0.1, 1, 10, 100]

    # Track the best model found during the hyperparameter search.
    best_accuracy = -1
    best_c = None
    best_model = None

    for c in c_values:
        # Standardize features before fitting a linear SVM.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=c))
        ])

        start_time = time.perf_counter()

        # Fit the pipeline using the training data.
        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        # Predict validation labels and measure classification performance.
        val_predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, val_predictions)
        macro_f1 = f1_score(y_val, val_predictions, average="macro")

        print(
            f"C={c:<6} | "
            f"Val Acc: {accuracy:.4f} | "
            f"Val Macro F1: {macro_f1:.4f} | "
            f"Training time: {training_time:.2f}s"
        )

        # Keep the model with the highest validation accuracy.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = c
            best_model = model

    # Create the checkpoint directory if it does not exist.
    model_dir = PROJECT_ROOT / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the best trained pipeline for later reuse.
    joblib.dump(
        best_model,
        model_dir / "linear_svm_best.joblib"
    )

    print(f"Saved model: {model_dir / 'linear_svm_best.joblib'}")

    # Report the result of the hyperparameter search.
    print("\nBEST LINEAR SVM")
    print("----------------")
    print(f"Best C: {best_c}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    return best_model, best_c


if __name__ == "__main__":
    train_linear_svm()
