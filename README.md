# 🖼️ SwinIR for Image Super-Resolution

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Torch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Ready](https://img.shields.io/badge/Run%20on-Kaggle-blue)](https://www.kaggle.com/)

This repository implements a custom version of **SwinIR (Swin Transformer for Image Restoration)** for single image super-resolution (SISR). The model is trained and evaluated on low-resolution (LR) and high-resolution (HR) image pairs and then used to predict HR images from unseen LR test inputs.


## 📌 Features

- 🔍 Custom SwinIR architecture in PyTorch
- 📦 Dataset loader for LR/HR pairs
- 🏋️ Mixed-precision training with `torch.amp`
- 📈 Training & evaluation with MSE loss
- 🧪 Test inference & prediction CSV generation


## 🗂️ Dataset Structure

Make sure your dataset is organized as follows:

```
dataset/
├── train/
│   ├── train/        # Low-resolution training images
│   └── gt/           # Ground truth high-resolution images
├── val/
│   ├── val/          # Low-resolution validation images
│   └── gt/           # Ground truth validation images
└── test/             # Low-resolution test images
```


## 🛠️ Setup

Install the required dependencies (assumes PyTorch with CUDA support is installed):

```bash
pip install torch torchvision numpy pandas pillow
```

> Or use [Kaggle Notebooks](https://www.kaggle.com/) where these packages are pre-installed.


## 🧠 Model: SwinIR

A simplified version of the SwinIR model using hierarchical convolutional blocks:

```python
model = SwinIR().cuda()
```

Hyperparameters:
- `img_size=64`, `embed_dim=60`
- `depths=[6,6,6]`, `num_heads=[6,6,6]`
- `upscale=4` (4x super-resolution)


## 🔁 Training Loop

- Loss: **MSE**
- Optimizer: **Adam**
- Epochs: **50**
- Mixed precision: ✅ (via `torch.amp`)

```python
for epoch in range(num_epochs):
    ...
    with autocast():
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)
```


## 🧪 Evaluation

During validation, MSE loss is computed per batch:

```python
model.eval()
with torch.no_grad():
    for lr_images, hr_images, _ in eval_loader:
        ...
        eval_loss += loss.item()
```


## 📤 Test Inference & Submission

1. Predict HR outputs from LR test images.
2. Save predicted images to `/kaggle/working/test_predictions/`.
3. Flatten grayscale-converted outputs into a submission CSV:

```python
images_to_csv(output_dir, "/kaggle/working/submission.csv")
```

Each row in the CSV contains:
- `ID`: Image ID (e.g. `gt_001`)
- `pixel_0`, `pixel_1`, ..., `pixel_N`: Flattened pixel values


## ✅ Example Results

Training and evaluation loss after 50 epochs:

```
Epoch [50/50], Training Loss: 0.001150
Evaluation Loss: 0.024336
```


## 📁 Output Files

- `test_predictions/`: Folder with upsampled test images
- `submission.csv`: CSV file for competition or external evaluation


## 📜 License

This project is licensed under the [MIT License](LICENSE).


## 🙌 Acknowledgements

- Based on the [SwinIR paper](https://arxiv.org/abs/2108.10257)
- Inspired by the [SwinIR GitHub Repo](https://github.com/JingyunLiang/SwinIR)
