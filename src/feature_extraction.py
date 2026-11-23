
import time
from pathlib import Path

import numpy as np
import torch

from src.dataset import get_feature_dataloaders
from src.model import build_model, extract_features
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "results" / "features"

def extract_dataset_features(model, loader, device):
    all_features = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = extract_features(model, images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels

def run_feature_extraction():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    train_loader, val_loader = get_feature_dataloaders()

    model = build_model("feature_extraction").to(device)
    model.eval()

    start_time = time.perf_counter()

    print("Extracting training features...")
    train_features, train_labels = extract_dataset_features(
        model, train_loader, device
    )

    print("Extracting validation features...")
    val_features, val_labels = extract_dataset_features(
        model, val_loader, device
    )

    extraction_time = time.perf_counter() - start_time

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURE_DIR / "pet_features.npz",
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels
    )

    print("\nFEATURE EXTRACTION COMPLETED")
    print("----------------------------")
    print(f"Training features: {train_features.shape}")
    print(f"Validation features: {val_features.shape}")
    print(f"Feature dimensions: {train_features.shape[1]}")
    print(f"Extraction time: {extraction_time:.2f} seconds")
    print(f"Saved to: {FEATURE_DIR / 'pet_features.npz'}")

if __name__ == "__main__":
    run_feature_extraction()
