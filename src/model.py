
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


NUM_CLASSES = 37

def build_model(strategy="full_finetuning"):

    # Load MobileNetV3-Small with pretrained ImageNet weights
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Replace the final classifier layer
    in_features = model.classifier[3].in_features # information about the class at: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    # Configure trainable parameters
    if strategy == "full_finetuning":

        # All model parameters are trainable
        model.requires_grad_(True)

    elif strategy == "head_training":

        # Freeze the convolutional feature extractor
        model.features.requires_grad_(False)

        # Train the fully connected classifier
        model.classifier.requires_grad_(True)

    elif strategy == "feature_extraction":

        # Freeze the entire pretrained network
        model.requires_grad_(False)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model


def count_trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def extract_features(model, images):

    # Extract convolutional features
    features = model.features(images)

    # Apply average pooling
    features = model.avgpool(features)

    # Flatten into feature vectors
    features = torch.flatten(features, start_dim=1)

    return features


# ---------- QUICK TEST ----------

if __name__ == "__main__":

    strategies = [
        "full_finetuning",
        "head_training",
        "feature_extraction"
    ]

    for strategy in strategies:

        model = build_model(strategy)

        print(f"\nStrategy: {strategy}")
        print(f"Trainable parameters: {count_trainable_parameters(model):,}")

        # Test the output dimensions
        dummy_input = torch.randn(2, 3, 224, 224)

        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"Output shape: {output.shape}")

        # Test feature extraction
        if strategy == "feature_extraction":

            with torch.no_grad():
                features = extract_features(model, dummy_input)

            print(f"Feature shape: {features.shape}")
