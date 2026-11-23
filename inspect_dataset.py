
from torchvision import datasets
from collections import Counter

# Dataset download directory
DATA_DIR = "./data"

# Load the official training + validation partition
trainval_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="category",
    download=True
)

# Load the official test partition
test_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    target_types="category",
    download=True
)

# Basic dataset information
print("DATASET INFORMATION")
print("-------------------")
print("Training + validation images:", len(trainval_dataset))
print("Test images:", len(test_dataset))

# Inspect one sample
image, label = trainval_dataset[0]

print("\nSAMPLE INFORMATION")
print("------------------")
print("Image type:", type(image))
print("Image dimensions:", image.size)
print("Label:", label)

# Count images belonging to each breed
trainval_counts = Counter(trainval_dataset._labels)
test_counts = Counter(test_dataset._labels)

print("\nNUMBER OF BREEDS")
print("----------------")
print("Trainval:", len(trainval_counts))
print("Test:", len(test_counts))

print("\nCLASS DISTRIBUTION")
print("------------------")
for class_id in sorted(trainval_counts):
    print(
        f"Class {class_id:2d}: "
        f"trainval = {trainval_counts[class_id]:3d}, "
        f"test = {test_counts[class_id]:3d}"
    )
