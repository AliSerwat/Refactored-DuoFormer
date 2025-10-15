# 📚 Documentation Index
## Complete Guide to Refactored DuoFormer

Welcome to the **Refactored DuoFormer** documentation! This index helps you find the right resource for your needs.

---

## 🚀 Quick Start

**New to DuoFormer?** Start here:

1. **[README.md](../README.md)** - Overview and quick start guide
2. **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation instructions
3. **[GETTING_STARTED.md](../GETTING_STARTED.md)** - Comprehensive beginner's guide
4. **[demo_duoformer.ipynb](../demo_duoformer.ipynb)** - Interactive tutorial

---

## 📖 User Documentation

### 🎓 Learning Resources

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[README.md](../README.md)** | Project overview, features, quick examples | First time visitor |
| **[GETTING_STARTED.md](../GETTING_STARTED.md)** | Detailed walkthrough, common tasks | Learning the system |
| **[demo_duoformer.ipynb](../demo_duoformer.ipynb)** | Interactive hands-on tutorial | Exploring features |

### 🔧 Setup & Installation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[INSTALLATION.md](INSTALLATION.md)** | Complete installation guide | Installing the system |
| **[setup_environment.py](../setup_environment.py)** | Automated dependency setup | Quick installation |
| **[requirements.in](../requirements.in)** | Direct dependencies | Manual setup |
| **[requirements.txt](../requirements.txt)** | Pinned versions | Reproducible installs |

### 📊 Configuration

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[config/model_config.py](../config/model_config.py)** | Python configuration classes | Understanding configs |
| **[config/default_config.yaml](../config/default_config.yaml)** | Standard settings | Regular training |
| **[config/lightweight_config.yaml](../config/lightweight_config.yaml)** | Fast experiments | Quick testing |
| **[config/performance_config.yaml](../config/performance_config.yaml)** | Best accuracy | Production use |

### 🐛 Troubleshooting

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues & solutions | When things go wrong |
| **[QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)** | Quick fixes for developers | Need fast solution |
| **[CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)** | Known issues & fixes | Understanding codebase |
| **[FIXES_APPLIED.md](FIXES_APPLIED.md)** | What's been fixed | Checking fix status |

---

## 💻 Developer Documentation

### 🏗️ Code Understanding

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)** | Comprehensive code analysis | Understanding architecture |
| **[FIXES_APPLIED.md](FIXES_APPLIED.md)** | Critical fixes documentation | Reviewing changes |
| **[QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)** | Developer quick reference | Daily development |

### 🤝 Contributing

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines | Want to contribute |
| **[tests/README.txt](tests/README.txt)** | Testing guide | Writing tests |

### 🧪 Testing

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[tests/run_tests.py](tests/run_tests.py)** | Test runner | Running tests |
| **[tests/README.txt](tests/README.txt)** | Testing documentation | Understanding tests |
| **Unit Tests** ([tests/unit/](tests/unit/)) | Fast unit tests | Quick validation |
| **Integration Tests** ([tests/integration/](tests/integration/)) | Full system tests | Thorough testing |

---

## 🔬 Research & Academic

### 📄 Academic Resources

| Resource | Purpose | Link |
|----------|---------|------|
| **Original Paper** | DuoFormer methodology | [arXiv:2506.12982](https://arxiv.org/abs/2506.12982) |
| **Original Repository** | TCGA-specific implementation | [duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA) |
| **This Repository** | General medical imaging | [Refactored-DuoFormer](https://github.com/AliSerwat/Refactored-DuoFormer) |

### 📝 Citation

```bibtex
@inproceedings{tang2025duoformer,
  title={DuoFormer: Hierarchical Vision Transformer for Medical Image Segmentation},
  author={Tang, Xiaoya and others},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```

---

## 🛠️ Utility Scripts

### System Diagnostics

| Script | Purpose | Command |
|--------|---------|---------|
| **[check_system.py](scripts/check_system.py)** | System capabilities check | `python scripts/check_system.py` |
| **[verify_installation.py](scripts/verify_installation.py)** | Installation verification | `python scripts/verify_installation.py` |
| **[health_check.py](scripts/health_check.py)** | Code health check | `python scripts/health_check.py` |

### Example Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| **[demo_robust.py](examples/demo_robust.py)** | Platform-agnostic demo | `python examples/demo_robust.py` |
| **[example_usage.py](examples/example_usage.py)** | Feature demonstrations | `python examples/example_usage.py` |

---

## 📂 Code Structure

### Core Modules

```
refactored-duoformer/
│
├── models/                      Model Architectures
│   ├── model_wo_extra_params.py    Main model (recommended)
│   ├── model.py                    Original implementation
│   ├── multi_vision_transformer.py Multi-scale transformer
│   ├── multiscale_attn.py          Multi-scale attention
│   ├── projection_head.py          Projection layers
│   ├── scale_attention.py          Scale attention
│   └── resnet50ssl.py              Self-supervised ResNet
│
├── utils/                       Training Utilities
│   ├── trainer.py                  Professional trainer
│   ├── dataset.py                  Data loading
│   ├── device_utils.py             Hardware detection
│   └── platform_utils.py           Platform optimization
│
├── config/                      Configuration
│   ├── model_config.py             Config classes
│   └── *.yaml                      Config files
│
├── tests/                       Testing
│   ├── unit/                       Fast tests
│   ├── integration/                Full tests
│   └── fixtures/                   Mock data
│
└── scripts/                     Utility Scripts
    ├── check_system.py             System check
    ├── verify_installation.py      Installation check
    └── health_check.py             Code health
```

---

## 🎯 Use Case Navigation

### "I want to..."

#### **...install the system**
1. Read: [INSTALLATION.md](INSTALLATION.md)
2. Run: `python setup_environment.py`
3. Verify: `python scripts/verify_installation.py`

#### **...understand the architecture**
1. Read: [README.md](README.md) - Architecture section
2. Read: [GETTING_STARTED.md](GETTING_STARTED.md) - Understanding section
3. Explore: [demo_duoformer.ipynb](demo_duoformer.ipynb)

#### **...train on my data**
1. Read: [GETTING_STARTED.md](GETTING_STARTED.md) - Common Tasks
2. Organize data (see [README.md](README.md))
3. Run: `python train.py --data_dir ./data`

#### **...fix a problem**
1. Check: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Quick fix: [QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)
3. Deep dive: [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)

#### **...contribute code**
1. Read: [CONTRIBUTING.md](CONTRIBUTING.md)
2. Setup: Development environment
3. Test: `python tests/run_tests.py --unit`
4. Submit: Pull request

#### **...optimize performance**
1. Check system: `python scripts/check_system.py`
2. Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Performance section
3. Try: Different configurations in [config/](config/)

#### **...understand the code**
1. Read: [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)
2. Read: [FIXES_APPLIED.md](FIXES_APPLIED.md)
3. Explore: Source code with inline comments

---

## 📱 Quick Links

### Essential Documents
- 🏠 [Home (README)](../README.md)
- 🚀 [Getting Started](../GETTING_STARTED.md)
- 📦 [Installation](INSTALLATION.md)
- 🔧 [Troubleshooting](TROUBLESHOOTING.md)
- 🤝 [Contributing](CONTRIBUTING.md)

### Technical Resources
- 📊 [Code Review](CODE_REVIEW_REPORT.md)
- ✅ [Fixes Applied](FIXES_APPLIED.md)
- ⚡ [Quick Reference](QUICK_FIX_REFERENCE.md)
- 🧪 [Testing Guide](tests/README.txt)

### External Links
- 📄 [Original Paper](https://arxiv.org/abs/2506.12982)
- 🔬 [Original Repo](https://github.com/xiaoyatang/duoformer_TCGA)
- 🚀 [This Repo](https://github.com/AliSerwat/Refactored-DuoFormer)

---

## 🔍 Search Guide

### By Topic

**Installation & Setup:**
- Installation → `docs/INSTALLATION.md`
- Dependencies → `requirements.txt`, `requirements.in`
- Environment → `setup_environment.py`

**Usage & Training:**
- Quick start → `README.md`
- Detailed guide → `GETTING_STARTED.md`
- Interactive → `demo_duoformer.ipynb`
- Training → `train.py`

**Configuration:**
- Overview → `README.md` - Configuration section
- Details → `config/model_config.py`
- Examples → `config/*.yaml`

**Troubleshooting:**
- Common issues → `docs/TROUBLESHOOTING.md`
- Quick fixes → `docs/QUICK_FIX_REFERENCE.md`
- Code review → `docs/CODE_REVIEW_REPORT.md`

**Development:**
- Contributing → `docs/CONTRIBUTING.md`
- Code structure → This document
- Testing → `tests/README.txt`

---

## 📞 Support Resources

### Self-Help
1. **Check Documentation** - Use this index
2. **Run Diagnostics** - `python scripts/check_system.py`
3. **Search Issues** - [GitHub Issues](https://github.com/AliSerwat/Refactored-DuoFormer/issues)

### Getting Help
1. **Troubleshooting Guide** - [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **GitHub Issues** - Report bugs or ask questions
3. **GitHub Discussions** - Community support

---

## 🗺️ Documentation Roadmap

### Coming Soon
- [ ] API Reference
- [ ] Advanced Configuration Guide
- [ ] Performance Tuning Guide
- [ ] Docker Deployment Guide
- [ ] Multi-GPU Training Guide
- [ ] Custom Dataset Guide
- [ ] Model Export Guide (ONNX, TorchScript)

---

## 📊 Documentation Statistics

- **Total Documents**: 15+
- **Lines of Documentation**: 5000+
- **Code Comments**: Comprehensive
- **Examples**: Multiple formats
- **Languages**: English
- **Format**: Markdown, Python, Jupyter

---

## ✅ Documentation Checklist

Before deploying:
- [✅] README.md - Complete and up-to-date
- [✅] INSTALLATION.md - All platforms covered
- [✅] GETTING_STARTED.md - Beginner friendly
- [✅] TROUBLESHOOTING.md - Common issues documented
- [✅] CONTRIBUTING.md - Clear guidelines
- [✅] CODE_REVIEW_REPORT.md - All issues documented
- [✅] QUICK_FIX_REFERENCE.md - Developer friendly
- [✅] FIXES_APPLIED.md - Changes tracked
- [✅] Inline code comments - Adequate coverage
- [✅] Docstrings - All public APIs
- [✅] Examples - Working and tested
- [✅] Configuration files - Well documented

---

**Last Updated**: October 15, 2025
**Repository**: https://github.com/AliSerwat/Refactored-DuoFormer
**Documentation Status**: ✅ Complete and Ready for GitHub

---

<div align="center">

**Made with ❤️ for the medical AI community**

[Original Paper](https://arxiv.org/abs/2506.12982) • [Original Repository](https://github.com/xiaoyatang/duoformer_TCGA) • [This Repository](https://github.com/AliSerwat/Refactored-DuoFormer)

</div>

