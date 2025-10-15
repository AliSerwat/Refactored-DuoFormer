# 🚀 Getting Started with Refactored DuoFormer

Welcome! This guide will walk you through the **Refactored DuoFormer** codebase for general medical image classification.

---

## 📋 Quick Navigation

- [What is This?](#what-is-this)
- [5-Minute Quickstart](#5-minute-quickstart)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Common Tasks](#common-tasks)
- [FAQ](#faq)

---

## 🎯 What is This?

**Refactored DuoFormer** is a production-ready implementation of a multi-scale vision transformer for **general medical image classification**.

### Supports Multiple Medical Imaging Modalities:
- 🔬 **Histopathology**: H&E, IHC, whole slide images
- 🏥 **Radiology**: X-ray, CT, MRI, ultrasound
- 🎨 **Dermatology**: Skin lesions, dermoscopy
- 👁️ **Ophthalmology**: Retinal images, OCT, fundus photography
- 🧬 **Pathology**: Cell classification, tissue analysis

### Key Enhancements:
✅ Platform-independent (Windows/Linux/macOS)
✅ Hardware-agnostic (auto-detects CUDA/MPS/CPU)
✅ Modern PyTorch (no deprecated APIs)
✅ Professional MLOps practices
✅ Clean, modular structure
✅ Resource-efficient testing

> **Based on**: [xiaoyatang/duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA)
> **Paper**: [arXiv:2506.12982](https://arxiv.org/abs/2506.12982)

---

## ⚡ 5-Minute Quickstart

```bash
# 1. Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 2. Install dependencies (one command!)
python setup_environment.py

# 3. Check your system
python scripts/check_system.py

# 4. Run quick tests
python tests/run_tests.py --unit

# 5. Try the demo
jupyter notebook demo_duoformer.ipynb
```

**That's it!** You're ready to train on your medical images.

---

## 📁 Project Structure

```
refactored-duoformer/
│
├── 📁 config/                    Configuration Management
│   ├── model_config.py              Type-safe configs (Python dataclasses)
│   ├── default_config.yaml          Balanced settings
│   ├── lightweight_config.yaml      Fast experiments (ResNet-18, 2 scales)
│   └── performance_config.yaml      Best accuracy (ResNet-50, 4 scales)
│
├── 📁 models/                    Model Architectures
│   ├── __init__.py                  Exports: build_model_no_extra_params(), count_parameters()
│   ├── model_wo_extra_params.py     Main DuoFormer (recommended)
│   ├── model.py                     Original implementation
│   ├── multi_vision_transformer.py  Multi-scale transformer
│   ├── multiscale_attn.py           Multi-scale attention
│   ├── projection_head.py           Projection layers
│   ├── resnet50ssl.py               Self-supervised ResNet
│   └── scale_attention.py           Scale attention mechanisms
│
├── 📁 utils/                     Training Utilities
│   ├── trainer.py                   Professional trainer (checkpointing, TensorBoard)
│   ├── dataset.py                   MedicalImageDataset, augmentation
│   ├── device_utils.py              Auto-detect CUDA/MPS/CPU
│   └── platform_utils.py            Platform-specific optimizations
│
├── 📁 tests/                     Testing (Resource-Efficient!)
│   ├── unit/                        Fast tests (<30s, no GPU)
│   ├── integration/                 Full tests (slower, GPU optional)
│   ├── fixtures/                    Mock data generators
│   └── run_tests.py                 Central test runner
│
├── 📁 scripts/                   Utility Scripts
│   ├── check_system.py              System capabilities + recommendations
│   ├── health_check.py              Code health validation
│   └── verify_installation.py      Installation verification
│
├── 📁 examples/                  Usage Examples
│   ├── demo_robust.py               Platform-agnostic demo
│   └── example_usage.py             Feature demonstrations
│
├── 🐍 train.py                   Main Training Script
├── 🔧 setup_environment.py       One-Command Setup
├── 📓 demo_duoformer.ipynb       Interactive Demo
│
├── 📦 requirements.in            Direct Dependencies
└── 📦 requirements.txt           Lockfile (compiled automatically from requirements.in)
```

---

## ✨ Key Features

### 1. **Automatic Hardware Detection**

The codebase automatically detects your hardware and optimizes:

```bash
# Just run with --device auto (default)
python train.py --data_dir ./data

# Automatically detects and uses:
# - CUDA (NVIDIA GPUs)
# - MPS (Apple Silicon M1/M2/M3)
# - CPU (fallback)
```

### 2. **Platform-Specific Optimization**

**Windows**:
- `num_workers=4` (optimal for Windows)
- `pin_memory=False` (prevents crashes)

**Linux**:
- `num_workers=8` (full multi-process)
- `pin_memory=True` (faster GPU transfer)
- `persistent_workers=True`

**macOS**:
- MPS support for Apple Silicon
- Optimized worker count

### 3. **Flexible Dataset Loading**

Two ways to load your data:

**Option A - Directory Structure:**
```
data/
├── class_0/
│   ├── image1.png
│   └── image2.png
├── class_1/
│   └── ...
```

**Option B - CSV File:**
```csv
image_path,label
/path/to/image1.png,0
/path/to/image2.png,1
```

Both work automatically!

### 4. **Professional Training Pipeline**

```python
from utils import Trainer

trainer = Trainer(model, criterion, optimizer, device)
trainer.fit(train_loader, val_loader, epochs=100)

# Automatically handles:
# ✓ Checkpointing (best, latest, periodic)
# ✓ Early stopping
# ✓ TensorBoard logging
# ✓ Mixed precision (AMP)
# ✓ Gradient clipping
```

### 5. **Resource-Efficient Testing**

```bash
# Fast unit tests (30 seconds, no GPU)
python tests/run_tests.py --unit

# Full integration tests (when needed)
python tests/run_tests.py --integration
```

---

## 📚 Common Tasks

### Task 1: Train on Your Medical Images

```bash
# Organize your data (Option A - folders)
# data/
#   ├── benign/
#   └── malignant/

# Train with auto-configuration
python train.py --data_dir ./data --num_classes 2

# Or use a config file
python train.py --config config/default_config.yaml
```

### Task 2: Fine-Tune on Small Dataset

```bash
# Use lightweight config + frozen backbone
python train.py \
    --config config/lightweight_config.yaml \
    --freeze_backbone \
    --lr 1e-5 \
    --epochs 50
```

### Task 3: Train on Specific GPU

```bash
# Single GPU
python train.py --device cuda:0 --data_dir ./data

# Multiple GPUs
python train.py --gpu_ids 0,1,2 --data_dir ./data
```

### Task 4: CPU-Only Training

```bash
python train.py --device cpu --batch_size 8 --data_dir ./data
```

### Task 5: Monitor Training

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir=runs

# Open http://localhost:6006 in browser
```

### Task 6: Resume Training

```bash
python train.py --resume checkpoints/best_checkpoint.pt
```

### Task 7: Inference on New Images

```python
import torch
from models import build_model_no_extra_params

# Load checkpoint
checkpoint = torch.load('checkpoints/best_checkpoint.pt')

# Build model
model = build_model_no_extra_params(
    depth=12, embed_dim=768, num_heads=12,
    num_classes=10, num_layers=2, backbone='r50'
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1)
```

---

## 🎓 Understanding the Architecture

### Model Components

```
Input Image (224×224×3)
    ↓
┌─────────────────────────┐
│ ResNet Backbone         │ Extract features at multiple scales
│ (ResNet-50/18)          │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Multi-Scale Features    │ 2, 3, or 4 resolution scales
│ (from different layers) │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Multi-Scale Transformer │ Attention across scales
│ (DuoFormer core)        │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Classification Head     │ Final predictions
└─────────────────────────┘
    ↓
Output (num_classes)
```

### Configuration Profiles

| Profile | Backbone | Scales | Speed | Accuracy | Use When |
|---------|----------|--------|-------|----------|----------|
| **Lightweight** | ResNet-18 | 2 | Fast | Good | Quick experiments |
| **Default** | ResNet-50 | 2 | Medium | Better | Standard training |
| **Performance** | ResNet-50 | 4 | Slow | Best | Maximum accuracy |

---

## 🔧 Customization

### Modify Configuration

Edit `config/my_experiment.yaml`:

```yaml
exp_name: my_experiment
device: auto  # Auto-detect hardware

backbone:
  name: resnet50
  pretrained: true
  freeze: true  # Faster convergence

transformer:
  depth: 12
  embed_dim: 768
  num_heads: 12

multiscale:
  num_layers: 2  # 2, 3, or 4 scales

training:
  learning_rate: 0.0001
  epochs: 100
  batch_size: 32  # Adjust based on GPU memory

data:
  data_dir: ./my_data
  num_classes: 5  # Your number of classes
  image_size: 224
```

Then train:
```bash
python train.py --config config/my_experiment.yaml
```

---

## ❓ FAQ

### Q: Which configuration should I use?

**A**: Start with `lightweight_config.yaml` for testing, then use `default_config.yaml` for real training.

### Q: My GPU runs out of memory?

**A**: Reduce batch size:
```bash
python train.py --batch_size 16  # or 8
```

Or use gradient accumulation (coming soon).

### Q: Can I use this without a GPU?

**A**: Yes! Use CPU mode:
```bash
python train.py --device cpu --batch_size 8
```

### Q: What image formats are supported?

**A**: PNG, JPG, JPEG, TIF, TIFF

### Q: How do I add data augmentation?

**A**: It's automatic! Controlled in config:
```yaml
data:
  use_augmentation: true
  random_flip: true
  random_rotation: 10
  color_jitter: true
```

### Q: How do I check if my installation works?

**A**:
```bash
python scripts/verify_installation.py
python scripts/check_system.py
python tests/run_tests.py --unit
```

---

## 📖 Next Steps

1. ✅ **Installation**: `python setup_environment.py`
2. ✅ **Verification**: `python scripts/verify_installation.py`
3. ✅ **System Check**: `python scripts/check_system.py`
4. ✅ **Quick Test**: `python tests/run_tests.py --unit`
5. ✅ **Try Demo**: `jupyter notebook demo_duoformer.ipynb`
6. ✅ **Train Model**: `python train.py --data_dir ./your_data`

---

## 📚 Additional Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Platform-specific setup instructions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](docs/CONTRIBUTING.md)** - Development guidelines
- **[Code Review](docs/CODE_REVIEW_REPORT.md)** - Code quality analysis
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Complete documentation map

---

## 📞 Support

- **Check System**: `python scripts/check_system.py`
- **Health Check**: `python scripts/health_check.py`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **Examples**: See `examples/` directory
- **Original Paper**: [arXiv:2506.12982](https://arxiv.org/abs/2506.12982)

---

**Happy Training!** 🎉

