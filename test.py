import torch
import torchvision
from torchvision import models

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)

print("\nHardware")
print("CUDA available:", torch.cuda.is_available())
print("CPU threads:", torch.get_num_threads())

print("\nMobileNetV3-Small")
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

print("Model successfully loaded")
print("Number of output classes:", model.classifier[-1].out_features)
print("Preprocessing:", weights.transforms())
