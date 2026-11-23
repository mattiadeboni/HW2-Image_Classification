
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import set_seed


# ---------- CONFIGURATION ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BATCH_SIZE = 32
NUM_EPOCHS = 10

LEARNING_RATES = {
    "full_finetuning": 1e-4,
    "head_training": 1e-3
}


# ---------- TRAINING ----------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    # Keep the frozen feature extractor in evaluation mode
    # so that its BatchNorm statistics do not change.
    if not any(p.requires_grad for p in model.features.parameters()):
        model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------- VALIDATION ----------

def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------- COMPLETE TRAINING PROCESS ----------

def train_model(strategy):
    set_seed(42)

    if strategy not in LEARNING_RATES:
        raise ValueError(f"Unsupported training strategy: {strategy}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Strategy: {strategy}")

    # Load data
    train_loader, val_loader, _ = get_dataloaders()

    # Build model
    model = build_model(strategy)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer: update only trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATES[strategy]
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"{strategy}_best.pth"

    best_val_accuracy = -1.0
    best_epoch = 0

    start_time = time.perf_counter()

    # Training loop
    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # Save the best model according to validation accuracy
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "strategy": strategy,
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_accuracy
                },
                checkpoint_path
            )

    training_time = time.perf_counter() - start_time

    print("\nTRAINING COMPLETED")
    print("------------------")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")

    return model, best_val_accuracy, training_time


# ---------- MAIN ----------

if __name__ == "__main__":

    train_model("head_training")
