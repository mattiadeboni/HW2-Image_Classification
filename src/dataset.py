
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights


# ---------- CONFIGURATION ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = PROJECT_ROOT / "splits" / "pet_splits.npz"

BATCH_SIZE = 32
NUM_WORKERS = 0

# ---------- TRANSFORMATIONS ----------

weights = MobileNet_V3_Small_Weights.DEFAULT
eval_transform = weights.transforms()

# Applied only to training images
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(
        eval_transform.resize_size[0],
        interpolation=eval_transform.interpolation
    ),
    transforms.RandomCrop(eval_transform.crop_size[0]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=eval_transform.mean,
        std=eval_transform.std
    )
])

# Applied to validation and test images
validation_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    eval_transform
])


# ---------- DATASET AND DATALOADERS ----------

def get_dataloaders():

    # Load the saved training and validation indices
    split_data = np.load(SPLIT_PATH)

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    # Training dataset with augmentation
    train_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True
    )

    # Validation dataset without augmentation
    val_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=validation_transform,
        download=True
    )

    # Official test dataset, kept separate
    test_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        transform=validation_transform,
        download=True
    )

    # Apply the saved indices
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


def get_feature_dataloaders():

    # Load the saved training and validation indices
    split_data = np.load(SPLIT_PATH)

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    # Use deterministic preprocessing for both splits
    train_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=validation_transform,
        download=True
    )

    val_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=validation_transform,
        download=True
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


# ---------- QUICK TEST ----------

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print("DATA LOADERS")
    print("------------")
    print(f"Training images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader)) # Get one batch of training data

    print("\nTRAINING BATCH")
    print("--------------")
    print(f"Images shape: {images.shape}") # Print the shape of the images tensor: (batch_size, channels, height, width)
    print(f"Labels shape: {labels.shape}") # Print the shape of the labels tensor: (batch_size)
    print(f"Labels: {labels[:10].tolist()}") # Print the first 10 labels in the batch: they go from 0 to 36, representing the 37 classes of pets in the dataset
