# 🌐 **Cloud Access Guide for DuoFormer**

## 🚀 **Quick Start - Universal Solution**

**One command works everywhere:**

```bash
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
```

This automatically detects your cloud environment and provides platform-specific instructions!

---

## ⚡ **Lightning.ai Studio** (Your Current Environment)

### **Method 1: Cloud Script (Recommended)**
```bash
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
```

**Follow the printed instructions:**
1. Click the "Port viewer" icon in the sidebar (bottom icon)
2. Click "+ New Port" button
3. Set Port: `8888` (or whatever port is shown)
4. Set Address: `localhost`
5. Click "Create" to expose the port
6. Click the generated URL to access Jupyter

### **Method 2: Manual Setup**
```bash
# Start Jupyter with cloud settings
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Then use Port Viewer extension to expose port 8888
```

---

## 🌐 **Other Cloud Platforms**

### **Google Colab**
```python
# Upload demo_duoformer.ipynb to Colab
!pip install -r requirements.txt
# Run cells directly
```

### **Kaggle Notebooks**
```python
# Upload demo_duoformer.ipynb to Kaggle
# Install dependencies:
!pip install -r requirements.txt
# Run cells directly
```

### **AWS SageMaker**
```bash
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
# Follow printed instructions for port forwarding
```

### **Azure ML / Google Cloud**
```bash
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
# Follow printed instructions for port forwarding
```

---

## 🐳 **Docker Environment**

### **Option 1: Docker Build**
```bash
# Build containerized Jupyter
docker build -f Dockerfile.jupyter -t duoformer-jupyter .

# Run with port mapping
docker run -p 8888:8888 -v $(pwd):/workspace duoformer-jupyter

# Access: http://localhost:8888
```

### **Option 2: Docker Compose**
```bash
# Start with Docker Compose
docker-compose up duoformer-jupyter

# Access: http://localhost:8888
```

---

## 💻 **Local Development**

### **Standard Jupyter**
```bash
jupyter notebook demo_duoformer.ipynb
# or
jupyter lab demo_duoformer.ipynb
```

### **Cloud Script (Works Locally Too)**
```bash
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
# Access: http://localhost:8888
```

---

## 🔧 **Enhanced Features**

### **Install Jupyter Extensions**
```bash
python scripts/install_jupyter_extensions.py
```

**Extensions included:**
- 📊 Git integration
- 📑 Table of contents
- 🎨 Code formatting
- 🎛️ Interactive widgets
- 📈 Enhanced plotting

### **Monitor Training with TensorBoard**
```bash
# Start TensorBoard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# For cloud environments, expose port 6006 using Port Viewer
```

---

## 🆘 **Troubleshooting**

### **Port Already in Use**
```bash
# The cloud script automatically finds available ports
python scripts/start_jupyter_cloud.py demo_duoformer.ipynb
```

### **Jupyter Not Found**
```bash
# Install Jupyter
pip install jupyterlab

# Or use our setup script
python setup_environment.py
```

### **Permission Issues**
```bash
# Make scripts executable
chmod +x scripts/start_jupyter_cloud.py
chmod +x scripts/install_jupyter_extensions.py
```

### **Browser Doesn't Open**
This is normal in cloud environments! Use the Port Viewer extension or follow the printed instructions.

---

## 🎯 **Best Practices**

### **For Lightning.ai Studio:**
1. ✅ Use `python scripts/start_jupyter_cloud.py demo_duoformer.ipynb`
2. ✅ Use Port Viewer extension for port access
3. ✅ Install extensions for better UX: `python scripts/install_jupyter_extensions.py`

### **For Other Cloud Platforms:**
1. ✅ Use the cloud script - it auto-detects your environment
2. ✅ Follow the printed instructions
3. ✅ Set up port forwarding as instructed

### **For Local Development:**
1. ✅ Use standard Jupyter commands
2. ✅ Or use cloud script for consistency

---

## 📞 **Need Help?**

1. **Check System**: `python scripts/check_system.py`
2. **Verify Installation**: `python scripts/verify_installation.py`
3. **Run Tests**: `python tests/run_tests.py --unit`
4. **Health Check**: `python scripts/health_check.py`

---

**🎉 Happy coding with DuoFormer!**
