# 📁 Project Structure Guide
## Refactored DuoFormer - Organized Codebase

This document explains the new, organized structure of the Refactored DuoFormer codebase, designed to follow software engineering and GitOps best practices.

---

## 🏗️ **New Directory Structure**

```
Refactored-DuoFormer/
│
├── 📦 src/                          # Source code (main package)
│   └── duoformer/                   # Main package
│       ├── __init__.py              # Package exports and metadata
│       ├── models/                  # Model components
│       │   ├── __init__.py         # Model exports
│       │   ├── duoformer_model.py # Main DuoFormer model
│       │   ├── model_wo_extra_params.py # Legacy model (backward compatibility)
│       │   ├── backbone/           # Backbone networks
│       │   │   ├── __init__.py
│       │   │   └── backbone.py    # ResNet implementations
│       │   ├── transformer/        # Transformer components
│       │   │   ├── __init__.py
│       │   │   └── multi_vision_transformer.py
│       │   ├── attention/          # Attention mechanisms
│       │   │   ├── __init__.py
│       │   │   ├── scale_attention.py
│       │   │   └── multiscale_attn.py
│       │   └── projection/         # Projection heads
│       │       ├── __init__.py
│       │       └── projection_head.py
│       ├── config/                 # Configuration management
│       │   ├── __init__.py
│       │   ├── model_config.py    # Configuration classes
│       │   ├── default_config.yaml
│       │   ├── lightweight_config.yaml
│       │   └── performance_config.yaml
│       ├── data/                   # Data handling
│       │   ├── loaders/           # Data loaders
│       │   │   ├── __init__.py
│       │   │   └── dataset.py
│       │   ├── augmentation/      # Data augmentation (future)
│       │   └── preprocessing/     # Data preprocessing (future)
│       ├── training/              # Training components
│       │   ├── trainer/           # Training logic
│       │   │   ├── __init__.py
│       │   │   └── trainer.py
│       │   ├── optimizers/        # Optimizers (future)
│       │   ├── schedulers/        # Learning rate schedulers (future)
│       │   └── metrics/           # Training metrics (future)
│       └── utils/                 # Utility functions
│           ├── device/            # Device management
│           │   ├── __init__.py
│           │   └── device_utils.py
│           ├── logging/            # Logging utilities
│           │   ├── __init__.py
│           │   └── logging_config.py
│           ├── platform/           # Platform-specific utilities
│           │   ├── __init__.py
│           │   └── platform_utils.py
│           └── exceptions/         # Custom exceptions
│               ├── __init__.py
│               └── exceptions.py
│
├── 🧪 tests/                       # Test suite (unchanged)
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── run_tests.py              # Test runner
│
├── 🛠️ tools/                      # Development and operational tools
│   ├── testing/                  # Testing tools
│   │   ├── __init__.py
│   │   ├── check_system.py       # System capability check
│   │   ├── health_check.py       # Code health check
│   │   └── verify_installation.py # Installation verification
│   ├── devops/                   # DevOps tools
│   │   ├── __init__.py
│   │   ├── setup_environment.py # Environment setup
│   │   ├── start_jupyter_cloud.py # Cloud Jupyter launcher
│   │   └── install_jupyter_extensions.py # Jupyter extensions
│   └── analysis/                 # Analysis tools (future)
│       └── __init__.py
│
├── 🏗️ infrastructure/             # Infrastructure as Code
│   ├── docker/                   # Docker configurations
│   │   ├── __init__.py
│   │   ├── Dockerfile.jupyter    # Jupyter Dockerfile
│   │   └── docker-compose.yml   # Docker Compose
│   ├── kubernetes/               # Kubernetes manifests (future)
│   └── terraform/                # Terraform configurations (future)
│
├── 🚀 deployment/                 # Deployment configurations
│   ├── scripts/                  # Deployment scripts (future)
│   ├── configs/                  # Deployment configs (future)
│   └── monitoring/               # Monitoring configs (future)
│
├── 📓 notebooks/                  # Jupyter notebooks
│   ├── tutorials/                # Tutorial notebooks (future)
│   ├── examples/                 # Example notebooks
│   │   ├── __init__.py
│   │   ├── demo_duoformer.ipynb  # Main demo notebook
│   │   ├── demo_robust.py        # Robust demo script
│   │   └── example_usage.py      # Usage examples
│   └── research/                 # Research notebooks (future)
│
├── 📚 docs/                       # Documentation (unchanged)
│   ├── README.md
│   ├── INSTALLATION.md
│   ├── TROUBLESHOOTING.md
│   ├── TESTING_GUIDE.md
│   ├── FIXES_APPLIED.md
│   ├── QUICK_FIX_REFERENCE.md
│   └── CONTRIBUTING.md
│
├── 🗂️ legacy/                     # Legacy code (unchanged)
│   ├── model_legacy.py
│   └── resnet50ssl_legacy.py
│
├── 📊 assets/                     # Static assets
│   ├── images/                    # Images (future)
│   ├── icons/                     # Icons (future)
│   └── logos/                     # Logos (future)
│
├── 📄 Configuration Files
│   ├── pyproject.toml            # Python project configuration
│   ├── requirements.in           # Direct dependencies
│   ├── requirements.txt          # Locked dependencies
│   ├── mypy.ini                  # Type checking configuration
│   └── .gitignore               # Git ignore rules
│
├── 📋 Documentation Files
│   ├── README.md                 # Main project documentation
│   ├── GETTING_STARTED.md       # Getting started guide
│   ├── SECURITY.md              # Security policy
│   └── LICENSE                  # License file
│
└── 🎯 Entry Points
    ├── train.py                 # Main training script
    └── setup.py                 # Package setup (future)
```

---

## 🎯 **Key Organizational Principles**

### **1. Separation of Concerns**
- **Models**: All model-related code in `src/duoformer/models/`
- **Configuration**: All config management in `src/duoformer/config/`
- **Training**: All training logic in `src/duoformer/training/`
- **Data**: All data handling in `src/duoformer/data/`
- **Utils**: All utility functions in `src/duoformer/utils/`

### **2. Functional Grouping**
- **Backbone Networks**: ResNet implementations in `models/backbone/`
- **Attention Mechanisms**: All attention code in `models/attention/`
- **Transformers**: Transformer components in `models/transformer/`
- **Projections**: Projection heads in `models/projection/`

### **3. Tool Organization**
- **Testing Tools**: All testing utilities in `tools/testing/`
- **DevOps Tools**: All operational tools in `tools/devops/`
- **Analysis Tools**: Future analysis tools in `tools/analysis/`

### **4. Infrastructure Separation**
- **Docker**: Container configurations in `infrastructure/docker/`
- **Kubernetes**: Future K8s manifests in `infrastructure/kubernetes/`
- **Terraform**: Future IaC in `infrastructure/terraform/`

### **5. Deployment Management**
- **Scripts**: Deployment scripts in `deployment/scripts/`
- **Configs**: Deployment configurations in `deployment/configs/`
- **Monitoring**: Monitoring setups in `deployment/monitoring/`

---

## 🔄 **Migration Benefits**

### **Before (Old Structure)**
```
models/
├── backbone.py
├── projection_head.py
├── scale_attention.py
├── multiscale_attn.py
├── multi_vision_transformer.py
├── duoformer_model.py
└── model_wo_extra_params.py

utils/
├── dataset.py
├── device_utils.py
├── exceptions.py
├── logging_config.py
├── platform_utils.py
└── trainer.py

scripts/
├── check_system.py
├── health_check.py
├── verify_installation.py
├── start_jupyter_cloud.py
├── install_jupyter_extensions.py
└── setup_environment.py
```

### **After (New Structure)**
```
src/duoformer/
├── models/
│   ├── backbone/
│   ├── transformer/
│   ├── attention/
│   ├── projection/
│   └── [main models]
├── config/
├── data/
├── training/
└── utils/

tools/
├── testing/
├── devops/
└── analysis/

infrastructure/
├── docker/
├── kubernetes/
└── terraform/
```

---

## 🚀 **Usage Examples**

### **Import from New Structure**
```python
# Main package imports
from duoformer import build_model_no_extra_params, ModelConfig, Trainer

# Specific component imports
from duoformer.models.backbone import ResNetBackbone
from duoformer.models.attention import ScaleAttention
from duoformer.config import DEFAULT_CONFIG
from duoformer.utils.device import setup_device_environment
```

### **Running Tools**
```bash
# Testing tools
python tools/testing/health_check.py
python tools/testing/check_system.py
python tools/testing/verify_installation.py

# DevOps tools
python tools/devops/setup_environment.py
python tools/devops/start_jupyter_cloud.py

# Training
python train.py --data_dir /path/to/data
```

### **Infrastructure**
```bash
# Docker
docker build -f infrastructure/docker/Dockerfile.jupyter .
docker-compose -f infrastructure/docker/docker-compose.yml up

# Future Kubernetes
kubectl apply -f infrastructure/kubernetes/
```

---

## 📋 **Migration Checklist**

- [✅] **Source Code Organized**: All source code moved to `src/duoformer/`
- [✅] **Models Categorized**: Models organized by functionality
- [✅] **Tools Separated**: Tools organized by purpose
- [✅] **Infrastructure Created**: Infrastructure directories created
- [✅] **Entry Points Updated**: Main scripts updated for new structure
- [✅] **Package Exports**: Proper `__init__.py` files with exports
- [⏳] **Import Updates**: All imports updated to new structure
- [⏳] **Documentation Updated**: Documentation reflects new structure
- [⏳] **Tests Updated**: Tests updated for new imports
- [⏳] **CI/CD Updated**: CI/CD pipelines updated for new structure

---

## 🎯 **Next Steps**

1. **Update Import Statements**: Update all imports to use new structure
2. **Update Documentation**: Update all documentation to reflect new structure
3. **Update Tests**: Update test imports and paths
4. **Update CI/CD**: Update any CI/CD configurations
5. **Create Setup Script**: Create proper package setup script
6. **Add Type Stubs**: Add proper type stubs for better IDE support

---

**This new structure follows software engineering best practices and makes the codebase much more maintainable and user-friendly!** 🎉
