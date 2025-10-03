# Assignment 3 - CS6886: Systems Engineering for Deep Learning

**Submitted by:**  
**CS25S025 - R Sai Ashwin**  
Department of Computer Science and Engineering  
IIT Madras  

---

## Overview

This repository contains the submission for **Assignment 3** of *CS6886 - Systems Engineering for Deep Learning*.  
The assignment focuses on **training and compressing MobileNetV2 on CIFAR-10**, exploring techniques such as:

- Training MobileNetV2 with data augmentation and mixed precision  
- Model evaluation and visualization utilities  
- Quantization-aware training (QAT) with STE-based quantization  
- K-Means based weight sharing for model compression  
- Logging experiments with **Weights & Biases (wandb)**  

The goal is to analyze trade-offs between **model size** and **accuracy** when applying quantization and weight sharing.  

---

## Repository Structure

```
.
├── data.py         # CIFAR-10 dataset loader with augmentation
├── main.py         # Main entry point for training, quantization, and evaluation
├── model.py        # MobileNetV2 wrapper and model size utilities
├── quant.py        # Quantization and K-Means weight sharing modules
├── trainer.py      # Training loop with mixed precision + cosine LR schedule
├── utils.py        # Visualization + per-layer stats utilities
└── README.md       # This file
```

---

## Features

### 1. Data Pipeline (`data.py`)
- CIFAR-10 dataset with Resize → Crop → Flip → AutoAugment → Normalize.  
- Train/test loaders with efficient `DataLoader` settings.

### 2. Model (`model.py`)
- MobileNetV2 wrapper for CIFAR-10.  
- Functions to compute:
  - Raw model size (FP32)  
  - Quantized/compressed model size (with custom modules)  

### 3. Training (`trainer.py`)
- Uses SGD with momentum + cosine annealing LR.  
- Mixed-precision training with `torch.amp`.  
- Automatic model saving on best validation accuracy.  
- wandb logging for metrics.

### 4. Quantization & Compression (`quant.py`)
- QAT layers:  
  - `QATConv2d`, `QATLinear` with per-channel/per-tensor quantization.  
  - `QATActFakeQuant` for activations.  
- Weight sharing via K-Means:  
  - `KMeansSharedConv2d` and `KMeansSharedLinear`.  
- Automatic replacement utilities (`replace_all_conv2d`, `replace_all_linear`).  
- Activation calibration for post-training quantization.

### 5. Utilities (`utils.py`)
- `WeightVisualizer`: Histogram of weights.  
- `BatchNormVisualizer`: Analyze gamma values.  
- `LayerStats`: Quantization error per conv/linear layer.  
- `ActivationStats`: Collect activation ranges and int8 loss.  
- CLI tools for analysis (e.g., `--histogram`, `--activation-stats`).  

---

## Usage

### 1. Install Dependencies
```bash
pip install torch torchvision wandb matplotlib torchinfo
```

### 2. Train MobileNetV2 on CIFAR-10
```bash
python main.py --epochs 200 --batch-size 256 --device cuda:0 --run-name baseline_run
```

### 3. Load Pretrained Weights & Evaluate
```bash
python main.py --no-train --weights mobilenetv2_cifar10.pth --device cuda:0
```

### 4. Apply Quantization
```bash
python main.py --weights mobilenetv2_cifar10.pth --quant --quant-bits-act 8
python main.py --weights mobilenetv2_cifar10.pth --quant-linear --quant-linear-bits 8
python main.py --weights mobilenetv2_cifar10.pth --quant-all-conv --quant-conv-bits 8 --quant-conv-perchannel
```

### 5. Analyze a Model
```bash
python utils.py --weights mobilenetv2_cifar10.pth --histogram
python utils.py --weights mobilenetv2_cifar10.pth --bn-gamma --bn-table
python utils.py --weights mobilenetv2_cifar10.pth --activation-stats
```

---

## Logging with Weights & Biases
- Every run automatically logs:  
  - Training/validation accuracy and loss  
  - Learning rate schedule  
  - Model size (raw & quantized)  
  - Post-compression accuracy  

You can customize the project/run name:
```bash
python main.py --wandb-project Assignment3 --run-name quant_experiment
```

---

## Notes
- Default backbone: MobileNetV2 (width_mult=0.5, dropout=0.2).  
- CIFAR-10 images are resized to 224×224 for compatibility with ImageNet pretraining.  
- Code is designed to be modular for extending with pruning or other compression techniques.  

---

## Acknowledgements
This work was completed as part of **CS6886 - Systems Engineering for Deep Learning** coursework,  
Department of CSE, IIT Madras.  
