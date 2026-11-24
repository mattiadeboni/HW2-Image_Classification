
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.dataset import get_dataloaders
from src.model import build_model, extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CM_DIR = RESULTS_DIR / "confusion_matrices"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def evaluate_torch_model(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    evaluation_time = time.perf_counter() - start_time

    return (
        np.array(all_labels),
        np.array(all_predictions),
        evaluation_time
    )


def extract_test_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = extract_features(model, images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    extraction_time = time.perf_counter() - start_time

    return features, labels, extraction_time


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        )
    }


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(f"{model_name} - Test Confusion Matrix")
    fig.tight_layout()

    CM_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        CM_DIR / f"{model_name}_confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)


def save_predictions(y_true, y_pred, image_paths, class_names, model_name):
    correct = y_true == y_pred

    records = []

    for i in range(len(y_true)):
        records.append({
            "image": Path(image_paths[i]).relative_to(PROJECT_ROOT).as_posix(),
            "true_label": int(y_true[i]),
            "true_class": class_names[y_true[i]],
            "predicted_label": int(y_pred[i]),
            "predicted_class": class_names[y_pred[i]],
            "correct": bool(correct[i])
        })

    df = pd.DataFrame(records)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PREDICTIONS_DIR / f"{model_name}_predictions.csv",
        index=False
    )

    print(f"Correct predictions: {correct.sum()}/{len(correct)}")
    print(f"Incorrect predictions: {(~correct).sum()}/{len(correct)}")


def evaluate_all():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders()

    test_dataset = test_loader.dataset
    class_names = test_dataset.classes
    image_paths = test_dataset._images

    all_results = {}

    # ---------------------------------------------------------
    # 1. Full fine-tuning
    # ---------------------------------------------------------

    print("\nEvaluating full fine-tuning...")

    model = build_model("full_finetuning").to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "full_finetuning_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, eval_time = evaluate_torch_model(
        model, test_loader, device
    )

    all_results["full_finetuning"] = calculate_metrics(
        y_true, y_pred
    )

    print(all_results["full_finetuning"])

    save_confusion_matrix(
        y_true, y_pred, class_names, "full_finetuning"
    )

    save_predictions(
        y_true, y_pred, image_paths, class_names,
        "full_finetuning"
    )

    del model

    # ---------------------------------------------------------
    # 2. Classifier-head training
    # ---------------------------------------------------------

    print("\nEvaluating classifier-head training...")

    model = build_model("head_training").to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "head_training_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, eval_time = evaluate_torch_model(
        model, test_loader, device
    )

    all_results["head_training"] = calculate_metrics(
        y_true, y_pred
    )

    print(all_results["head_training"])

    save_confusion_matrix(
        y_true, y_pred, class_names, "head_training"
    )

    save_predictions(
        y_true, y_pred, image_paths, class_names,
        "head_training"
    )

    del model

    # ---------------------------------------------------------
    # 3. Feature extraction + Linear SVM
    # ---------------------------------------------------------

    print("\nEvaluating feature extraction + Linear SVM...")

    model = build_model("feature_extraction").to(device)

    X_test, y_true, extraction_time = extract_test_features(
        model, test_loader, device
    )

    svm = joblib.load(
        CHECKPOINT_DIR / "linear_svm_best.joblib"
    )

    y_pred = svm.predict(X_test)

    all_results["feature_extraction_svm"] = calculate_metrics(
        y_true, y_pred
    )

    print(all_results["feature_extraction_svm"])
    print(f"Test feature extraction time: {extraction_time:.2f}s")

    save_confusion_matrix(
        y_true, y_pred, class_names, "feature_extraction_svm"
    )

    save_predictions(
        y_true, y_pred, image_paths, class_names,
        "feature_extraction_svm"
    )

    # ---------------------------------------------------------
    # Save final metrics
    # ---------------------------------------------------------

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nFINAL TEST RESULTS")
    print("------------------")

    for name, metrics in all_results.items():
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro F1={metrics['macro_f1']:.4f}"
        )

    print(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate_all()
