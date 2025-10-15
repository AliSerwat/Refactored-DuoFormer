# 📦 Installation Guide
## Refactored DuoFormer Setup

Complete installation guide for **Refactored DuoFormer** on Windows, Linux, and macOS.

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Manual Installation](#manual-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **GPU**: Optional (NVIDIA CUDA 11.x+ or Apple Silicon MPS)

### Recommended Requirements
- **Python**: 3.10 or 3.11
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10 GB free space

---

## ⚡ Quick Installation

### One-Command Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 2. Run automated setup
python setup_environment.py
```

This command will:
✅ Upgrade pip to latest version
✅ Install pip-tools for dependency management
✅ Compile requirements with pinned versions
✅ Install all dependencies
✅ Validate the installation

**That's it!** Skip to [Verification](#verification).

---

## 🛠️ Manual Installation

If you prefer manual control or the automated setup fails:

### Step 1: Clone Repository

```bash
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer
```

### Step 2: Create Virtual Environment (Recommended)

**Option A - Using venv:**
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate
```

**Option B - Using conda:**
```bash
# Create conda environment
conda create -n duoformer python=3.10
conda activate duoformer
```

### Step 3: Install Dependencies

**Option A - From requirements.txt (recommended):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B - From requirements.in (latest versions):**
```bash
pip install --upgrade pip pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt
```

### Step 4: Install PyTorch (If Not Already Installed)

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision
# MPS support is automatic in PyTorch 2.0+
```

---

## 🌍 Platform-Specific Instructions

### Windows

```powershell
# 1. Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
python setup_environment.py

# 4. Verify installation
python scripts\verify_installation.py
```

**Common Windows Issues:**
- If you get SSL errors, update certificates: `pip install --upgrade certifi`
- If PowerShell blocks scripts, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Linux (Ubuntu/Debian)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# 2. Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
python setup_environment.py

# 5. Verify installation
python scripts/verify_installation.py
```

**For CUDA support:**
```bash
# Check CUDA version
nvcc --version

# Install PyTorch with matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.10

# 3. Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
python setup_environment.py

# 6. Verify installation
python scripts/verify_installation.py
```

**For Apple Silicon (M1/M2/M3):**
- MPS (Metal Performance Shaders) support is automatic
- PyTorch 2.0+ includes native Apple Silicon support

---

## ✅ Verification

After installation, run these commands to verify everything works:

### 1. Basic Import Test
```bash
python -c "import torch; import timm; from models import build_model_no_extra_params; print('✅ All imports successful!')"
```

### 2. System Check
```bash
python scripts/check_system.py
```

This will display:
- ✅ Platform information
- ✅ Python version
- ✅ PyTorch version
- ✅ CUDA/MPS availability
- ✅ GPU information (if available)
- ✅ Recommended settings

### 3. Installation Verification
```bash
python scripts/verify_installation.py
```

### 4. Quick Unit Tests
```bash
python tests/run_tests.py --unit
```

Expected output:
```
✅ All unit tests passed (30 seconds or less)
```

### 5. Try Demo
```bash
# Option A: Jupyter notebook
jupyter notebook demo_duoformer.ipynb

# Option B: Python script
python examples/demo_robust.py
```

---

## 🔧 Advanced Installation Options

### Development Installation

For development with editable install:

```bash
# Clone repository
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### Docker Installation (Coming Soon)

```bash
# Build Docker image
docker build -t duoformer:latest .

# Run container
docker run --gpus all -it -v $(pwd)/data:/data duoformer:latest
```

### Conda Environment File

Create `environment.yml`:
```yaml
name: duoformer
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0
  - torchvision
  - cudatoolkit=11.8  # Or your CUDA version
  - pip
  - pip:
    - -r requirements.txt
```

Install:
```bash
conda env create -f environment.yml
conda activate duoformer
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Not Detected

**Symptoms:**
```
CUDA available: False
```

**Solutions:**
1. Check CUDA installation: `nvcc --version`
2. Install correct PyTorch version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU drivers are up to date

### Issue 2: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'timm'
```

**Solutions:**
1. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```
2. Check virtual environment is activated
3. Try manual installation:
   ```bash
   pip install timm einops matplotlib
   ```

### Issue 3: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```bash
   python train.py --batch_size 8  # or 4
   ```
2. Use CPU mode:
   ```bash
   python train.py --device cpu
   ```
3. Enable gradient checkpointing (coming soon)

### Issue 4: Windows SSL Errors

**Symptoms:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions:**
```bash
pip install --upgrade certifi
# Or
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Issue 5: Permission Errors (Linux/macOS)

**Symptoms:**
```
Permission denied
```

**Solutions:**
```bash
# Don't use sudo! Use virtual environment instead
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 Dependency Details

### Core Dependencies
- **torch** (>=2.0): Deep learning framework
- **torchvision**: Image transformations
- **timm** (>=0.9.0): Vision Transformer components
- **einops**: Tensor operations
- **numpy**, **scipy**: Numerical computing
- **matplotlib**, **seaborn**: Visualization

### Training Utilities
- **tqdm**: Progress bars
- **tensorboard**: Training monitoring
- **scikit-learn**: Metrics and utilities
- **pandas**: Data handling

### Optional Dependencies
- **jupyter**: Interactive notebooks
- **pytest**: Testing framework
- **psutil**: System monitoring

---

## 🔄 Updating Installation

To update to the latest version:

```bash
# Update repository
git pull origin main

# Update dependencies
python setup_environment.py --compile-only  # Compile new requirements
python setup_environment.py --install-only  # Install updates
```

Or manually:
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Next Steps

After successful installation:

1. ✅ **Read Getting Started**: See [../GETTING_STARTED.md](../GETTING_STARTED.md)
2. ✅ **Check System**: `python scripts/check_system.py`
3. ✅ **Run Tests**: `python tests/run_tests.py --unit`
4. ✅ **Try Demo**: `jupyter notebook demo_duoformer.ipynb`
5. ✅ **Train Model**: See [../README.md](../README.md) for training instructions

---

## 📞 Need Help?

- **System Check**: `python scripts/check_system.py`
- **Health Check**: `python scripts/health_check.py`
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Code Review**: See [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)
- **Quick Fixes**: See [QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)
- **GitHub Issues**: [Report a bug](https://github.com/AliSerwat/Refactored-DuoFormer/issues)

---

## ✨ Installation Complete!

You're now ready to use Refactored DuoFormer! 🎉

Start with:
```bash
python train.py --help
```

Or jump into the interactive demo:
```bash
jupyter notebook demo_duoformer.ipynb
```

---

**Last Updated**: October 15, 2025
**Repository**: https://github.com/AliSerwat/Refactored-DuoFormer

