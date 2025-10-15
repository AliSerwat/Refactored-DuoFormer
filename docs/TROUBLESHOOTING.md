# 🔧 Troubleshooting Guide
## Refactored DuoFormer Common Issues & Solutions

Complete troubleshooting guide for **Refactored DuoFormer**.

---

## 📋 Quick Diagnosis

Run these commands to diagnose issues:

```bash
# 1. Check system capabilities
python scripts/check_system.py

# 2. Verify installation
python scripts/verify_installation.py

# 3. Check code health
python scripts/health_check.py

# 4. Run quick tests
python tests/run_tests.py --unit
```

---

## 🔴 Critical Issues

### Issue 1: ImportError - Missing Attention Class

**Symptoms:**
```python
NameError: name 'Attention' is not defined
File "models/scale_attention.py", line 15
```

**Cause:** Missing import in `scale_attention.py`

**Solution:** ✅ FIXED in latest version
```bash
# Update to latest version
git pull origin main
```

If you're on an older version, manually fix:
```python
# In models/scale_attention.py, line 2:
from timm.models.vision_transformer import VisionTransformer, LayerScale, Attention
```

---

### Issue 2: RuntimeError - Device Mismatch

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

**Cause:** Index tensors on wrong device

**Solution:** ✅ FIXED in latest version
```bash
# Update to latest version
git pull origin main
```

The latest version automatically moves tensors to the correct device.

---

### Issue 3: KeyError When Using num_layers=2

**Symptoms:**
```
KeyError: '0'
File "models/model_wo_extra_params.py"
```

**Cause:** Attempting to access non-existent feature map

**Solution:** ✅ FIXED in latest version
```bash
git pull origin main
```

---

## 🟡 Installation Issues

### Issue 4: CUDA Not Available

**Symptoms:**
```
CUDA available: False
```

**Diagnosis:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Solutions:**

**Option A - Install CUDA-enabled PyTorch:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Option B - Check CUDA installation:**
```bash
# Linux/macOS
nvcc --version

# Windows
nvcc --version
```

**Option C - Use CPU mode:**
```bash
python train.py --device cpu --batch_size 8
```

---

### Issue 5: Package Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0
```

**Solutions:**

**Step 1 - Update pip:**
```bash
python -m pip install --upgrade pip
```

**Step 2 - Install PyTorch first:**
```bash
pip install torch torchvision
```

**Step 3 - Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

### Issue 6: SSL Certificate Errors (Windows)

**Symptoms:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions:**

**Option A - Update certificates:**
```bash
pip install --upgrade certifi
```

**Option B - Use trusted hosts:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Option C - Update pip:**
```bash
python -m pip install --upgrade pip setuptools
```

---

## 🟢 Runtime Issues

### Issue 7: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

**Option A - Reduce batch size:**
```bash
python train.py --batch_size 8  # or 4, 2
```

**Option B - Use fewer scales:**
```bash
python train.py --num_layers 2  # instead of 4
```

**Option C - Use lightweight config:**
```bash
python train.py --config config/lightweight_config.yaml
```

**Option D - Use mixed precision:**
```bash
python train.py --amp
```

**Option E - Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

---

### Issue 8: Slow Training Speed

**Symptoms:**
- Training takes very long per epoch
- Low GPU utilization

**Diagnosis:**
```bash
# Check GPU utilization
nvidia-smi -l 1  # Linux/Windows with NVIDIA

# Check CPU usage
python scripts/check_system.py
```

**Solutions:**

**Option A - Increase num_workers:**
```bash
# Auto-detect optimal
python train.py --data_dir ./data

# Or manually set
python train.py --num_workers 8
```

**Option B - Enable pin_memory (Linux):**
```python
# Automatic on Linux
train_loader = DataLoader(..., pin_memory=True)
```

**Option C - Use persistent_workers (Linux):**
```python
train_loader = DataLoader(..., persistent_workers=True)
```

**Option D - Check data loading:**
```python
# Profile data loading
import time
start = time.time()
for batch in train_loader:
    break
print(f"First batch load time: {time.time() - start:.2f}s")
```

---

### Issue 9: Model Not Learning

**Symptoms:**
- Loss not decreasing
- Validation accuracy stuck

**Diagnosis:**
```python
# Check if backbone is frozen
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

**Solutions:**

**Option A - Unfreeze backbone:**
```bash
python train.py --freeze_backbone false
```

**Option B - Adjust learning rate:**
```bash
# Try higher learning rate
python train.py --lr 1e-3

# Or lower if diverging
python train.py --lr 1e-5
```

**Option C - Check data:**
```python
# Verify labels are correct
from utils import create_dataloaders
train_loader, _, _ = create_dataloaders(data_dir='./data')
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
    break
```

**Option D - Use label smoothing:**
```yaml
# In config YAML
training:
  label_smoothing: 0.1
```

---

### Issue 10: NaN Loss

**Symptoms:**
```
Loss: nan
```

**Causes & Solutions:**

**Cause A - Learning rate too high:**
```bash
python train.py --lr 1e-5  # Reduce from default 1e-4
```

**Cause B - Gradient explosion:**
```python
# Gradient clipping is enabled by default
trainer = Trainer(..., gradient_clip_val=1.0)
```

**Cause C - Invalid data:**
```python
# Check for NaN/Inf in data
import torch
data_has_nan = torch.isnan(images).any()
data_has_inf = torch.isinf(images).any()
print(f"NaN: {data_has_nan}, Inf: {data_has_inf}")
```

---

## 🔵 Data Issues

### Issue 11: Dataset Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data'
```

**Solutions:**

**Option A - Check path:**
```bash
# Verify data directory exists
ls -la ./data  # Linux/macOS
dir data  # Windows
```

**Option B - Use absolute path:**
```bash
python train.py --data_dir /absolute/path/to/data
```

**Option C - Organize data correctly:**
```
data/
├── class_0/
│   ├── image1.png
│   └── image2.png
├── class_1/
│   └── ...
```

---

### Issue 12: CSV File Loading Error

**Symptoms:**
```
KeyError: 'image_path' not found in CSV
```

**Solution:**

CSV file must have these columns:
```csv
image_path,label
/path/to/image1.png,0
/path/to/image2.png,1
```

Example:
```python
import pandas as pd

# Create CSV
df = pd.DataFrame({
    'image_path': ['./data/img1.png', './data/img2.png'],
    'label': [0, 1]
})
df.to_csv('data.csv', index=False)

# Use CSV
python train.py --csv_file data.csv
```

---

### Issue 13: Image Loading Errors

**Symptoms:**
```
Error loading image: cannot identify image file
```

**Solutions:**

**Option A - Check image formats:**
```bash
# Supported: PNG, JPG, JPEG, TIF, TIFF
file data/class_0/*.png  # Linux/macOS
```

**Option B - Verify images aren't corrupted:**
```python
from PIL import Image
import os

data_dir = './data'
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except Exception as e:
                print(f"Corrupted: {file} - {e}")
```

---

## 🟣 Configuration Issues

### Issue 14: Config File Not Found

**Symptoms:**
```
FileNotFoundError: config/my_config.yaml not found
```

**Solutions:**

**Option A - Use relative path from project root:**
```bash
python train.py --config config/default_config.yaml
```

**Option B - Check file exists:**
```bash
ls -la config/  # Should show .yaml files
```

**Option C - Create custom config:**
```bash
cp config/default_config.yaml config/my_config.yaml
# Edit my_config.yaml
python train.py --config config/my_config.yaml
```

---

### Issue 15: Configuration Validation Errors

**Symptoms:**
```
AssertionError: embed_dim (768) must be divisible by num_heads (11)
```

**Solution:**

Ensure valid configuration:
```yaml
transformer:
  embed_dim: 768  # Must be divisible by num_heads
  num_heads: 12   # 768 / 12 = 64 ✓

# Invalid examples:
# embed_dim: 768, num_heads: 11  ✗ (768 / 11 = 69.8...)
# embed_dim: 784, num_heads: 12  ✗ (784 / 12 = 65.3...)

# Valid combinations:
# embed_dim: 384, num_heads: 6   ✓
# embed_dim: 512, num_heads: 8   ✓
# embed_dim: 768, num_heads: 12  ✓
```

---

## 🟤 Testing Issues

### Issue 16: Tests Failing

**Symptoms:**
```
FAILED tests/unit/test_config.py::test_model_config
```

**Solutions:**

**Option A - Update dependencies:**
```bash
pip install -r requirements.txt --upgrade
python tests/run_tests.py --unit
```

**Option B - Run specific tests:**
```bash
# Run one test file
python tests/unit/test_config.py

# Run with verbose output
pytest tests/unit/test_config.py -v -s
```

**Option C - Skip integration tests:**
```bash
# Only run fast unit tests
python tests/run_tests.py --unit
```

---

## 🔄 Update Issues

### Issue 17: Git Pull Conflicts

**Symptoms:**
```
error: Your local changes to the following files would be overwritten by merge
```

**Solutions:**

**Option A - Stash changes:**
```bash
git stash
git pull origin main
git stash pop
```

**Option B - Create new branch:**
```bash
git checkout -b my-changes
git pull origin main
```

**Option C - Discard local changes:**
```bash
git reset --hard origin/main
```

---

## 📊 Performance Issues

### Issue 18: High Memory Usage (RAM)

**Symptoms:**
- System becomes unresponsive
- Out of memory errors

**Solutions:**

**Option A - Reduce num_workers:**
```bash
python train.py --num_workers 2  # or 1
```

**Option B - Reduce batch size:**
```bash
python train.py --batch_size 8
```

**Option C - Monitor memory:**
```python
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```

---

### Issue 19: Checkpoint Files Too Large

**Symptoms:**
- Checkpoint files are several GB each
- Running out of disk space

**Solutions:**

**Option A - Keep fewer checkpoints:**
```yaml
# In config
training:
  keep_last_n: 2  # Only keep last 2 checkpoints
```

**Option B - Save less frequently:**
```yaml
training:
  save_freq: 10  # Save every 10 epochs instead of 5
```

**Option C - Compress checkpoints:**
```python
# When saving manually
torch.save(checkpoint, 'model.pt', _use_new_zipfile_serialization=True)
```

---

## 🆘 Emergency Procedures

### Complete Reset

If nothing works, perform a complete reset:

```bash
# 1. Backup your data and configs
cp -r config config_backup
cp -r checkpoints checkpoints_backup

# 2. Remove everything except data
rm -rf venv/
rm -rf __pycache__/
rm -rf .pytest_cache/
find . -name "*.pyc" -delete

# 3. Fresh clone
cd ..
rm -rf Refactored-DuoFormer
git clone https://github.com/AliSerwat/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# 4. Fresh installation
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
python setup_environment.py

# 5. Restore your configs
cp -r ../config_backup/* config/
cp -r ../checkpoints_backup/* checkpoints/

# 6. Verify
python scripts/verify_installation.py
```

---

## 📞 Getting Help

### Self-Diagnosis Checklist

Before asking for help, check:

- [ ] ✅ Ran `python scripts/check_system.py`
- [ ] ✅ Ran `python scripts/verify_installation.py`
- [ ] ✅ Ran `python tests/run_tests.py --unit`
- [ ] ✅ Checked this troubleshooting guide
- [ ] ✅ Updated to latest version: `git pull origin main`
- [ ] ✅ Tried with latest dependencies: `python setup_environment.py`

### Reporting Issues

When reporting issues on GitHub, include:

1. **System Information:**
   ```bash
   python scripts/check_system.py > system_info.txt
   ```

2. **Error Message:**
   - Full error traceback
   - Command that caused the error

3. **Reproducible Example:**
   ```bash
   # Minimal command that reproduces the issue
   python train.py --data_dir ./data --batch_size 32
   ```

4. **Configuration:**
   - Attach your config file
   - List any custom modifications

---

## 📚 Additional Resources

- **Code Review Report**: [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md) - All known issues and fixes
- **Quick Fix Reference**: [QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md) - Quick solutions
- **Fixes Applied**: [FIXES_APPLIED.md](FIXES_APPLIED.md) - What's been fixed
- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md) - Detailed setup instructions
- **Getting Started**: [../GETTING_STARTED.md](../GETTING_STARTED.md) - Usage guide
- **GitHub Issues**: https://github.com/AliSerwat/Refactored-DuoFormer/issues

---

## ✅ Issue Resolution Status

| Issue | Status | Version Fixed |
|-------|--------|---------------|
| Missing Attention import | ✅ Fixed | Current |
| Device mismatch | ✅ Fixed | Current |
| KeyError num_layers=2 | ✅ Fixed | Current |
| Dtype inconsistency | ✅ Fixed | Current |
| Input validation | ✅ Added | Current |

---

**Last Updated**: October 15, 2025
**Repository**: https://github.com/AliSerwat/Refactored-DuoFormer
**Status**: All critical issues resolved ✅

