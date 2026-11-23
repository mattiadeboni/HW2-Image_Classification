
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision import datasets

# Reproducibility
SEED = 42

# Load the official trainval partition
dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True
)

# Extract labels and corresponding dataset indices
labels = np.array(dataset._labels) # 3680 labels
indices = np.arange(len(dataset)) # 3680 indices

# Stratified 80/20 split 
# (train_indices = 0.8 * 3680 = 2944 indices for training, 
# val_indices = 0.2 * 3680 = 736 indices for validation) 
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

# Save the split indices for reuse across all approaches
os.makedirs("./splits", exist_ok=True)

np.savez(
    "./splits/pet_splits.npz",
    train_indices=train_indices,
    val_indices=val_indices
)

# Verify the resulting partitions
# Count the number of images belonging to each breed in the training and validation sets
train_counts = Counter(labels[train_indices])
val_counts = Counter(labels[val_indices])

print("DATA SPLIT")
print("----------")
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Total:", len(train_indices) + len(val_indices))

print("\nCLASS DISTRIBUTION")
print("------------------")

for class_id in sorted(train_counts):
    print(
        f"Class {class_id:2d}: "
        f"train = {train_counts[class_id]:3d}, "
        f"validation = {val_counts[class_id]:3d}"
    )

print("\nSplit saved successfully.")
