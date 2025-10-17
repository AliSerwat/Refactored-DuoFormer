# 🧠 Refactored DuoFormer
## Multi-Scale Vision Transformer for Medical Imaging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Quality](https://img.shields.io/badge/code%20quality-A+-success.svg)](docs/CODE_REVIEW_REPORT.md)

**Production-ready, refactored implementation of DuoFormer for general medical image classification with enterprise-grade MLOps practices.**

- 💡 **Purpose**:
    - Refactored for general medical imaging use cases including histopathology, radiology, dermatology, and more
- 🔬 **Original Repository**:
    - [xiaoyatang/duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA)
- 📄 **Paper**:
    - [Hierarchical Vision Transformer for Medical Image Segmentation (MIDL 2025)](https://arxiv.org/abs/2506.12982)
- 🚀 **This Repository**:
    - [AliSerwat/Refactored-DuoFormer](https://github.com/AliSerwat/Refactored-DuoFormer)

---

## ✨ What's New in This Enhanced Version

This is a professionally refactored and enhanced version with:

- ✅ **Platform-Independent**: Works on Windows, Linux, macOS
- ✅ **Hardware-Agnostic**: Auto-detects CUDA, MPS (Apple Silicon), or CPU
- ✅ **Modern PyTorch**: No deprecated APIs, latest best practices
- ✅ **MLOps Ready**: Configuration management, auto-checkpointing, TensorBoard
- ✅ **Clean Code**: No wildcard imports, explicit dependencies
- ✅ **Professional Structure**: Modular, testable, maintainable
- ✅ **Comprehensive Testing**: Unit tests and health checks
- ✅ **Auto-Configuration**: Optimal settings for your hardware

---

## 🚀 Quick Start

### 1. Installation (30 seconds)

```bash
# 1. Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# One-command setup (handles everything)
python setup_environment.py
```

### 2. Verify Installation

```bash
# Check system capabilities
python scripts/check_system.py

# Verify installation
python scripts/verify_installation.py
```

### 3. Run Demo

```bash
# Interactive notebook
jupyter notebook demo_duoformer.ipynb

# Or platform-agnostic demo
python examples/demo_robust.py
```

### 4. Train Model

```bash
# Auto-configuration (detects your hardware)
python train.py --data_dir /path/to/data

# Or with specific configuration
python train.py --config config/default_config.yaml
```

---

## 📊 Architecture Overview

**DuoFormer** combines ResNet backbones with multi-scale transformers for medical image classification:

```
Input Image (224×224×3)
    ↓
ResNet Backbone (ResNet-50/18)
    ↓
Multi-Scale Features (2/3/4 scales)
    ↓
Multi-Scale Transformer
    ↓
Classification Head
    ↓
Predictions (num_classes)
```

**Key Features**:
- 🔄 Multi-scale attention at different resolutions
- 🏗️ Hybrid CNN-Transformer architecture
- 🎯 Optimized for medical imaging (histopathology, radiology)
- ⚡ Supports 2, 3, or 4 scale levels

---

## 📁 Project Structure

```
refactored-duoformer/
│
├── 📄 README.md                      Main project documentation (you are here)
├── 📄 GETTING_STARTED.md             Comprehensive beginner's guide
│
├── 📁 docs/                          Documentation directory
│   ├── README.md                     Documentation overview
│   ├── INSTALLATION.md               Platform-specific installation
│   ├── TROUBLESHOOTING.md            Common issues & solutions
│   ├── CONTRIBUTING.md               Development guidelines
│   ├── DOCUMENTATION_INDEX.md        Complete documentation map
│   ├── CODE_REVIEW_REPORT.md         Code quality analysis
│   ├── FIXES_APPLIED.md              What's been fixed
│   └── QUICK_FIX_REFERENCE.md        Developer quick reference
│
├── 📁 config/                        Configuration Management
│   ├── model_config.py               Type-safe configs (Python dataclasses)
│   ├── default_config.yaml           Balanced settings
│   ├── lightweight_config.yaml       Fast experiments (ResNet-18, 2 scales)
│   └── performance_config.yaml       Best accuracy (ResNet-50, 4 scales)
│
├── 📁 models/                        Model Architectures
│   ├── __init__.py                   Exports: build_model_no_extra_params(), count_parameters()
│   ├── model_wo_extra_params.py      Main DuoFormer (recommended)
│   ├── model.py                      Original implementation
│   ├── multi_vision_transformer.py   Multi-scale transformer
│   ├── multiscale_attn.py            Multi-scale attention
│   ├── projection_head.py            Projection layers
│   ├── resnet50ssl.py                Self-supervised ResNet
│   └── scale_attention.py            Scale attention mechanisms
│
├── 📁 utils/                         Training Utilities
│   ├── trainer.py                    Professional trainer (checkpointing, TensorBoard)
│   ├── dataset.py                    MedicalImageDataset, augmentation
│   ├── device_utils.py               Auto-detect CUDA/MPS/CPU
│   └── platform_utils.py             Platform-specific optimizations
│
├── 📁 tests/                         Testing (Resource-Efficient!)
│   ├── unit/                         Fast tests (<30s, no GPU)
│   ├── integration/                  Full tests (slower, GPU optional)
│   ├── fixtures/                     Mock data generators
│   └── run_tests.py                  Central test runner
│
├── 📁 scripts/                       Utility Scripts
│   ├── check_system.py               System capabilities + recommendations
│   ├── health_check.py               Code health validation
│   └── verify_installation.py        Installation verification
│
├── 📁 examples/                      Usage Examples
│   ├── demo_robust.py                Platform-agnostic demo
│   └── example_usage.py              Feature demonstrations
│
├── 🐍 train.py                       Main Training Script
├── 🔧 setup_environment.py           One-Command Setup
├── 📓 demo_duoformer.ipynb           Interactive Demo
│
├── 📦 requirements.in                Direct Dependencies
└── 📦 requirements.txt               Lockfile (compiled automatically from requirements.in)
```
---
---

## 💻 Usage

### Training

```bash
# Basic training (auto-configures everything)
python train.py --data_dir /path/to/data

# With custom configuration
python train.py --config config/performance_config.yaml

# Specific device
python train.py --device cuda --data_dir ./data

# Multi-GPU
python train.py --gpu_ids 0,1,2 --data_dir ./data

# CPU only
python train.py --device cpu --batch_size 8 --data_dir ./data

# With mixed precision
python train.py --amp --data_dir ./data

# Resume training
python train.py --resume checkpoints/best_checkpoint.pt
```

### Configuration

```python
from config import ModelConfig

# Load from YAML
config = ModelConfig.from_yaml('config/my_experiment.yaml')

# Or use presets
from config import DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, PERFORMANCE_CONFIG

# Modify and save
config.training.epochs = 200
config.to_yaml('config/my_custom.yaml')
```

### Inference

```python
import torch
from models import build_model_no_extra_params

# Load checkpoint
checkpoint = torch.load('checkpoints/best_checkpoint.pt')

# Create model
model = build_model_no_extra_params(
    depth=12,
    embed_dim=768,
    num_heads=12,
    num_classes=10,
    num_layers=2,
    backbone='r50'
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

## 🔧 Configuration Options

### Pre-configured Profiles

| Profile | Backbone | Scales | Params | Speed | Use Case |
|---------|----------|--------|--------|-------|----------|
| `lightweight_config.yaml` | ResNet-18 | 2 | ~30M | Fast | Quick experiments |
| `default_config.yaml` | ResNet-50 | 2 | ~50M | Medium | Standard training |
| `performance_config.yaml` | ResNet-50 | 4 | ~70M | Slow | Best accuracy |

### Custom Configuration

Create `config/my_experiment.yaml`:

```yaml
exp_name: my_tcga_experiment
device: auto  # Auto-detects CUDA/MPS/CPU

backbone:
  name: resnet50
  pretrained: true
  freeze: true

transformer:
  depth: 12
  embed_dim: 768
  num_heads: 12

multiscale:
  num_layers: 2
  proj_dim: 768

training:
  learning_rate: 0.0001
  epochs: 100
  batch_size: 32
  optimizer: adamw
  scheduler: cosine

data:
  data_dir: ./data
  num_classes: 10
  image_size: 224
```

Then train:
```bash
python train.py --config config/my_experiment.yaml
```

---

## 🌐 Cross-Platform Support

### Automatic Hardware Detection

The codebase automatically detects and optimizes for your platform:

**Windows**:
- Auto-detects CUDA or falls back to CPU
- Optimized `num_workers=4`
- `pin_memory=False` (prevents crashes)

**Linux**:
- Multi-GPU support
- Optimized `num_workers=8`
- `pin_memory=True` (faster transfer)

**macOS**:
- Apple Silicon (M1/M2/M3) MPS support
- Optimized `num_workers=4`
- Automatic device selection

### Device Options

```bash
# Auto-detect (recommended)
python train.py --device auto

# Specific GPU
python train.py --device cuda:0

# Apple Silicon
python train.py --device mps

# CPU only
python train.py --device cpu

# Multi-GPU
python train.py --gpu_ids 0,1,2
```

---

## 📚 Key Features

### 1. **Professional Training Pipeline**

```python
from utils import Trainer

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # Mixed precision
    gradient_clip_val=1.0
)

trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
    patience=20  # Early stopping
)
```

Features:
- ✅ Automatic checkpointing (best, latest, periodic)
- ✅ Early stopping
- ✅ Mixed precision training (AMP)
- ✅ Gradient clipping
- ✅ TensorBoard logging
- ✅ Learning rate scheduling

### 2. **Flexible Dataset Management**

```python
from utils import create_dataloaders

# From directory structure
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='./data',
    batch_size=32
)

# From CSV file
train_loader, val_loader, test_loader = create_dataloaders(
    csv_file='data.csv',
    batch_size=32
)
```

### 3. **Hardware Optimization**

```python
from utils import setup_device_environment, get_optimal_num_workers

# Auto-detect best device
device = setup_device_environment('auto')

# Get optimal workers for your platform
num_workers = get_optimal_num_workers()
```

---

## 🧪 Testing

```bash
# Fast unit tests (recommended, <30 seconds, no GPU)
python tests/run_tests.py --unit

# Full test suite (all tests)
python tests/run_tests.py

# Integration tests only (slower, may use GPU)
python tests/run_tests.py --integration

# Specific test file
python tests/unit/test_config.py
python tests/integration/test_full_models.py

# With pytest
pytest tests/unit/ -v              # Fast tests only
pytest tests/integration/ -v       # Integration tests
pytest tests/ -v                   # All tests

# Check code health
python scripts/health_check.py
```

---

## 📦 Requirements

### System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **GPU**: Optional (NVIDIA CUDA, or Apple MPS)
- **Storage**: 5GB free space

### Python Dependencies

Managed via `pip-tools` for reproducibility:
- PyTorch 2.0+
- torchvision
- timm (Vision Transformers)
- einops, numpy, matplotlib
- scikit-learn, pandas
- TensorBoard, Jupyter

**Install all**:
```bash
python setup_environment.py
```

---

## 📖 Documentation

- **Quick Start**: This README
- **Getting Started Guide**: See `GETTING_STARTED.md` for detailed walkthrough
- **Installation Guide**: See `docs/INSTALLATION.md` for platform-specific setup
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md` for common issues
- **Contributing**: See `docs/CONTRIBUTING.md` for development guidelines
- **Full Documentation**: See `docs/` directory for complete guides
- **System Check**: `python scripts/check_system.py`
- **Examples**: See `examples/` directory

---

## 🎯 Use Cases

### General Medical Image Classification

This refactored version works with various medical imaging modalities:

```python
# Histopathology (e.g., TCGA, CAMELYON)
python train.py \
    --data_dir ./data/histopathology \
    --num_classes 10 \
    --config config/default_config.yaml

# Radiology (e.g., X-rays, CT, MRI)
python train.py \
    --data_dir ./data/radiology \
    --num_classes 5 \
    --backbone r50 \
    --num_layers 4

# Dermatology (e.g., skin lesions)
python train.py \
    --data_dir ./data/dermatology \
    --num_classes 7 \
    --image_size 224

# Retinal imaging (e.g., diabetic retinopathy)
python train.py \
    --data_dir ./data/retinal \
    --num_classes 5 \
    --config config/performance_config.yaml
```

### Fine-Tuning

```python
# Freeze backbone for faster convergence
python train.py \
    --freeze_backbone \
    --lr 1e-5 \
    --epochs 50 \
    --data_dir ./data
```

---

## 🔬 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{tang2025duoformer,
  title={DuoFormer: Hierarchical Vision Transformer for Medical Image Segmentation},
  author={Tang, Xiaoya and others},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions welcome! This enhanced version adds:
- Modern PyTorch APIs
- Platform independence
- MLOps best practices
- Professional structure

See examples/ for usage patterns.

---

## 📜 License

Same license as original repository.

---

## 🙏 Acknowledgments

- **Original Work**: [xiaoyatang/duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA) - Original TCGA-focused implementation
- **Paper**: Tang, X. et al. "Hierarchical Vision Transformer for Medical Image Segmentation" (MIDL 2025)
- **This Refactoring**: Enhanced for general medical imaging with production-ready MLOps practices

---

## 📞 Support

- **System Issues**: `python scripts/check_system.py`
- **Code Health**: `python scripts/health_check.py`
- **Examples**: See `examples/` directory

**Note**: This is a refactored version for general medical imaging. For the original TCGA-specific implementation, see [xiaoyatang/duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA)

---

## ⭐ Features Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| **Code Quality** | Wildcards, deprecated | Explicit, modern ✅ |
| **Platform** | Linux-only paths | Cross-platform ✅ |
| **Device** | Hardcoded CUDA | Auto-detect ✅ |
| **Configuration** | Hardcoded | YAML/JSON ✅ |
| **Training** | Basic loop | Professional trainer ✅ |
| **Monitoring** | Print | TensorBoard ✅ |
| **Testing** | None | Unit tests ✅ |
| **Dependencies** | Manual | pip-tools lockfile ✅ |

---

<div align="center">

**Made with ❤️ for the medical AI community**

[Original Paper](https://arxiv.org/abs/2506.12982) • [Original Repository](https://github.com/xiaoyatang/duoformer_TCGA)

</div>

