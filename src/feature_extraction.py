
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
    # Collect features and labels separately for every loader batch.
    all_features = []
    all_labels = []

    # Disable training behavior such as dropout.
    model.eval()

    # Prevent PyTorch from building a computation graph during extraction.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # Convert each image batch into fixed-length feature vectors.
            features = extract_features(model, images)

            # Move tensors to the CPU before converting them to NumPy arrays.
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    # Combine all batches into one array for the complete dataset split.
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels

def run_feature_extraction():
    # Use a fixed seed for reproducible data loading and initialization.
    set_seed(42)

    # Run the pretrained network on a GPU when CUDA is available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Load the training and validation batches without test data.
    train_loader, val_loader = get_feature_dataloaders()

    # Build MobileNetV3 frozen for use as a feature extractor.
    model = build_model("feature_extraction").to(device)
    model.eval()

    start_time = time.perf_counter()

    # Extract features and labels for the training split.
    print("Extracting training features...")
    train_features, train_labels = extract_dataset_features(
        model, train_loader, device
    )

    # Extract features and labels for the validation split.
    print("Extracting validation features...")
    val_features, val_labels = extract_dataset_features(
        model, val_loader, device
    )

    # Measure the total time needed for both extraction passes.
    extraction_time = time.perf_counter() - start_time

    # Create the feature output directory if it does not exist.
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Save training and validation arrays together in a compressed NPZ file.
    np.savez_compressed(
        FEATURE_DIR / "pet_features.npz",
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels
    )

    # Report the resulting array sizes and save location.
    print("\nFEATURE EXTRACTION COMPLETED")
    print("----------------------------")
    print(f"Training features: {train_features.shape}")
    print(f"Validation features: {val_features.shape}")
    print(f"Feature dimensions: {train_features.shape[1]}")
    print(f"Extraction time: {extraction_time:.2f} seconds")
    print(f"Saved to: {FEATURE_DIR / 'pet_features.npz'}")

if __name__ == "__main__":
    run_feature_extraction()
