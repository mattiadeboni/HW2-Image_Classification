# HW2 – Image Classification

Author: Mattia De Boni (VR555453)

This project implements and compares three image-classification approaches on the **Oxford-IIIT Pet Dataset**, which contains 7,349 images belonging to 37 cat and dog breeds.

The experiments use a pretrained **MobileNetV3-Small**, initialized with ImageNet-1K weights, and compare:

1. **Full fine-tuning** – all network parameters are trainable.
2. **Classifier-head training** – the convolutional feature extractor is frozen, and only the classifier is trained.
3. **Feature extraction + Linear SVM** – the pretrained network is used as a fixed feature extractor, and a scikit-learn Linear SVM performs classification.

## Repository structure

```text
Image-Classification/
├── checkpoints/
│   ├── full_finetuning_best.pth
│   ├── head_training_best.pth
│   └── linear_svm_best.joblib
├── results/
│   ├── confusion_matrices/
│   │   ├── feature_extraction_svm_confusion_matrix.png
│   │   ├── full_finetuning_confusion_matrix.png
│   │   └── head_training_confusion_matrix.png
│   ├── features/
│   │   └── pet_features.npz
│   ├── metrics/
│   │   └── test_metrics.json
│   └── predictions/
│       ├── feature_extraction_svm_predictions.csv
│       ├── full_finetuning_predictions.csv
│       └── head_training_predictions.csv
├── splits/
│   └── pet_splits.npz
└── src/
    ├── __init__.py
    ├── analyze_results.py
    ├── classifiers.py
    ├── create_splits.py
    ├── dataset.py
    ├── evaluation.py
    ├── feature_extraction.py
    ├── model.py
    ├── train.py
    └── utils.py
```

The `data/` directory is intentionally excluded from the repository because it contains the dataset images and annotations.

## Requirements

The project was developed and tested with:

- Python 3.11.9
- PyTorch 2.15.0+cpu
- TorchVision 0.29.2+cpu
- NumPy 2.4.6
- scikit-learn 1.9.1
- Matplotlib 3.11.2

Additional dependencies include `pandas` and `joblib`.

The experiments were run on a CPU. A CUDA-compatible environment can also be used; the scripts automatically select CUDA when available.

### Environment setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux:

```bash
source venv/bin/activate
```

Install the dependencies. For a CPU-only PyTorch installation, use:

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 scikit-learn==1.9.1 matplotlib==3.11.2 pandas joblib
```

If using a GPU, install the appropriate PyTorch and TorchVision builds for the installed CUDA version instead.

## Dataset

The project uses the **Oxford-IIIT Pet Dataset**, consisting of 37 pet categories.

The dataset is not included in this repository. It is downloaded automatically by TorchVision when the data-loading scripts are first executed, using `download=True`.

No manual download or extraction is required. The dataset is stored locally under:

```text
data/
└── oxford-iiit-pet/
    ├── annotations/
    └── images/
```

The official dataset provides two partitions:

- **Trainval:** 3,680 images.
- **Test:** 3,669 images.

The official test partition is kept separate from training and validation.

The trainval partition is further divided into:

- **Training:** 2,944 images.
- **Validation:** 736 images.

The split is stratified by breed, using a fixed random seed of 42. The resulting indices are saved in `splits/pet_splits.npz` and reused by all three approaches to ensure consistent data partitions.

The existing split file is included in the repository. Therefore, it is not necessary to recreate the split when using the provided project files. To generate it again, run:

```bash
python -m src.create_splits
```

This command downloads the dataset if necessary and recreates the stratified split.

## Preprocessing

All images are converted to RGB and processed according to the preprocessing associated with the pretrained MobileNetV3-Small weights.

For training the two neural-network approaches, data augmentation consists of:

- Resizing.
- Random cropping.
- Random horizontal flipping with probability 0.5.
- Conversion to tensors.
- Normalization using the ImageNet mean and standard deviation associated with the pretrained weights.

Validation and test images use deterministic preprocessing, without augmentation.

For feature extraction, both training and validation images also use deterministic preprocessing. This ensures that the extracted features are consistent across runs.

The dataset is approximately balanced across the 37 classes, so no class weighting or oversampling is applied.

## Experimental methods

### 1. Full fine-tuning

The pretrained MobileNetV3-Small network is adapted to the 37-class classification task by replacing its final fully connected layer.

All network parameters are trainable.

- Loss: Cross-Entropy.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 32.
- Epochs: 10.
- Model selection: checkpoint with the highest validation accuracy.

### 2. Classifier-head training

The convolutional feature extractor is frozen, while the fully connected classifier is trainable.

The trainable layers are the two Linear layers in the MobileNetV3-Small classifier. The intermediate activation and dropout layers have no trainable parameters.

- Loss: Cross-Entropy.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 32.
- Epochs: 10.
- Model selection: checkpoint with the highest validation accuracy.

### 3. Feature extraction + Linear SVM

The pretrained MobileNetV3-Small network is used as a fixed feature extractor. Features are taken after the convolutional feature extractor and average pooling, resulting in 576-dimensional feature vectors.

A scikit-learn pipeline applies:

1. StandardScaler, fitted exclusively on the training features.
2. Linear SVM classification.

The regularization parameter \(C\) is selected using validation accuracy, testing the following values:

```text
0.01, 0.1, 1, 10, 100
```

The selected value is \(C = 0.01\).

No neural-network parameters are trained in this approach.

## Running the experiments

Run the commands from the project root directory, the directory containing `src/`, `checkpoints/`, and `results/`.

### 1. Train the full fine-tuning model

```bash
python -c "from src.train import train_model; train_model('full_finetuning')"
```

The best checkpoint is saved as:

```text
checkpoints/full_finetuning_best.pth
```

### 2. Train the classifier-head model

```bash
python -m src.train
```

The script's main entry point runs the `head_training` strategy.

The best checkpoint is saved as:

```text
checkpoints/head_training_best.pth
```

### 3. Extract features

```bash
python -m src.feature_extraction
```

This extracts the training and validation features using the frozen pretrained network and saves them to:

```text
results/features/pet_features.npz
```

### 4. Train the Linear SVM

```bash
python -m src.classifiers
```

The script evaluates the candidate values of \(C\) on the validation set and saves the selected pipeline to:

```text
checkpoints/linear_svm_best.joblib
```

### 5. Evaluate all models

After training the two neural networks and the SVM, run:

```bash
python -m src.evaluation
```

This evaluates the three approaches on the official test partition and generates:

- Test accuracy and macro-F1 metrics.
- Confusion matrices.
- Per-image prediction CSV files.
- A JSON file containing the final test metrics.

The generated files are saved under `results/`.

**Important:** The test partition is used only for the final evaluation. Model selection and hyperparameter selection are performed using the validation partition.

### 6. Analyze predictions

To inspect the most frequent classification errors and representative correct and incorrect predictions, run:

```bash
python -m src.analyze_results
```

This script reads the prediction CSV files generated during evaluation.

## Reproducibility

The random seed is set to 42 for Python, NumPy, and PyTorch.

The implementation also enables deterministic cuDNN operations, disables cuDNN benchmarking, and requests deterministic PyTorch algorithms.

The same saved train/validation indices are used across all approaches.

Exact numerical reproducibility across different hardware, software versions, or CUDA configurations is not guaranteed.

## Results

The following table reports the final results obtained on the official test partition.

| Method | Test accuracy | Macro-F1 |
|---|---:|---:|
| Full fine-tuning | 83.07% | 82.71% |
| Classifier-head training | 82.58% | 82.48% |
| Feature extraction + Linear SVM | **84.46%** | **84.33%** |

The complete metrics are available in:

```text
results/metrics/test_metrics.json
```

The confusion matrices and per-image predictions are available in their respective subdirectories under `results/`.

## References

- TorchVision documentation: [MobileNetV3-Small](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
- Oxford-IIIT Pet Dataset: [Official dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/).
