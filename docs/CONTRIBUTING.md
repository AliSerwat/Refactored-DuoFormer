# 🤝 Contributing to Refactored DuoFormer

Thank you for your interest in contributing to **Refactored DuoFormer**! This guide will help you get started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## 🌟 Code of Conduct

We are committed to providing a welcoming and inclusive environment. By participating in this project, you agree to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

---

## 🚀 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Found a bug? Let us know!
- ✨ **Feature Requests**: Have an idea? Share it!
- 📝 **Documentation**: Improve guides, add examples
- 🔧 **Bug Fixes**: Fix existing issues
- ⚡ **Performance Improvements**: Make it faster!
- 🧪 **Tests**: Add test coverage
- 💡 **New Features**: Implement new functionality

---

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Refactored-DuoFormer.git
cd Refactored-DuoFormer

# Add upstream remote
git remote add upstream https://github.com/AliSerwat/Refactored-DuoFormer.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python setup_environment.py

# Install development tools
pip install pytest black flake8 mypy
```

### 3. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# Or: git checkout -b fix/your-bug-fix
```

---

## 📏 Coding Standards

### Code Style

We follow these standards:

**Python Style:**
- Follow [PEP 8](https://pep8.org/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where possible

**Example:**
```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Train the model.

    Args:
        model: PyTorch model to train
        data_loader: Training data loader
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Dictionary with training metrics
    """
    # Implementation
    pass
```

### Code Formatting

**Before committing, format your code:**

```bash
# Auto-format with black
black models/ utils/ tests/

# Check style with flake8
flake8 models/ utils/ tests/ --max-line-length=100

# Type checking with mypy (optional)
mypy models/ utils/ --ignore-missing-imports
```

### Documentation

- **Docstrings**: All public functions/classes must have docstrings
- **Comments**: Explain *why*, not *what*
- **Type Hints**: Add type hints to function signatures
- **README**: Update relevant documentation

**Good Comment:**
```python
# Move index tensors to same device as input to prevent device mismatch errors
if not hasattr(self, '_indices_device_moved'):
    device = x[next(iter(x.keys()))].device
    for key in self.index:
        self.index[key] = self.index[key].to(device)
```

**Bad Comment:**
```python
# Moving indices to device
self.index[key] = self.index[key].to(device)
```

---

## 🧪 Testing

### Run Tests Before Submitting

```bash
# Run all unit tests (fast, required)
python tests/run_tests.py --unit

# Run integration tests (slower, recommended)
python tests/run_tests.py --integration

# Run specific test file
python tests/unit/test_config.py

# Run with pytest
pytest tests/ -v
```

### Writing Tests

Add tests for new functionality:

```python
# tests/unit/test_your_feature.py
import pytest
import torch
from models import build_model_no_extra_params

def test_your_feature():
    """Test description."""
    # Arrange
    model = build_model_no_extra_params(
        depth=6,
        embed_dim=384,
        num_heads=6,
        num_classes=10
    )

    # Act
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    # Assert
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
```

---

## 📤 Pull Request Process

### 1. Make Your Changes

```bash
# Make changes to code
# Add tests
# Update documentation

# Check code quality
black .
flake8 . --max-line-length=100
python tests/run_tests.py --unit
```

### 2. Commit Your Changes

**Commit Message Format:**
```
type(scope): short description

Longer description if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi-colons, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(models): add gradient checkpointing support"
git commit -m "fix(utils): resolve device mismatch in data loading"
git commit -m "docs(readme): update installation instructions"
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Include:
# - Clear title and description
# - Reference related issues
# - List of changes
# - Test results
```

### 4. PR Review Process

Your PR will be reviewed for:

- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation
- ✅ Performance impact
- ✅ Backward compatibility

**Checklist:**
- [ ] Tests pass (`python tests/run_tests.py --unit`)
- [ ] Code is formatted (`black .`)
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No linter errors (`flake8 .`)

---

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, include:

**1. System Information:**
```bash
python scripts/check_system.py
```

**2. Description:**
- Clear, concise description
- Expected vs. actual behavior

**3. Reproduction Steps:**
```bash
# Step-by-step commands to reproduce
python train.py --data_dir ./data --batch_size 32
```

**4. Error Messages:**
```
Full error traceback
```

**5. Additional Context:**
- Screenshots (if UI-related)
- Configuration files
- Sample data (if possible)

### Feature Requests

Include:

1. **Problem**: What problem does this solve?
2. **Proposed Solution**: How would you implement it?
3. **Alternatives**: What alternatives have you considered?
4. **Use Case**: Provide real-world scenarios

---

## 🏗️ Project Structure

Understanding the codebase:

```
refactored-duoformer/
├── models/          # Model architectures
│   ├── model.py                    # Original implementation
│   ├── model_wo_extra_params.py   # Main model (use this)
│   ├── multi_vision_transformer.py
│   └── ...
├── utils/           # Training utilities
│   ├── trainer.py              # Training loop
│   ├── dataset.py              # Data loading
│   ├── device_utils.py         # Hardware detection
│   └── platform_utils.py       # Platform optimization
├── config/          # Configuration management
├── tests/           # Test suite
│   ├── unit/                   # Fast tests
│   └── integration/            # Slow tests
└── scripts/         # Utility scripts
```

---

## 🔍 Code Review Guidelines

### For Reviewers

- Be constructive and respectful
- Focus on code, not the person
- Explain the "why" behind suggestions
- Acknowledge good work

### For Contributors

- Respond to all comments
- Don't take feedback personally
- Ask questions if unclear
- Update PR based on feedback

---

## 📚 Development Resources

### Useful Commands

```bash
# Format code
black models/ utils/ tests/

# Check style
flake8 . --max-line-length=100

# Run tests
python tests/run_tests.py --unit

# Check system
python scripts/check_system.py

# Verify installation
python scripts/verify_installation.py
```

### Documentation

- **Code Review Report**: [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)
- **Quick Fixes**: [QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)
- **Installation**: [INSTALLATION.md](INSTALLATION.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Documentation Index**: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

---

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- 🐛 Bug fixes (see [Issues](https://github.com/AliSerwat/Refactored-DuoFormer/issues))
- 📝 Documentation improvements
- 🧪 Test coverage expansion
- 🏎️ Performance optimizations

### Medium Priority
- ✨ New features (discuss first via Issues)
- 🎨 Code refactoring
- 📊 Visualization improvements
- 🔧 Configuration options

### Low Priority
- 💅 UI/UX improvements
- 📖 Tutorial creation
- 🌍 Internationalization

---

## 📞 Questions?

- **Issues**: [GitHub Issues](https://github.com/AliSerwat/Refactored-DuoFormer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AliSerwat/Refactored-DuoFormer/discussions)
- **Documentation**: See `docs/` directory

---

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Acknowledged in documentation

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## 🙏 Thank You!

Every contribution, no matter how small, helps make Refactored DuoFormer better for everyone. We appreciate your time and effort!

**Happy Contributing!** 🎉

---

**Repository**: https://github.com/AliSerwat/Refactored-DuoFormer
**Original Work**: [xiaoyatang/duoformer_TCGA](https://github.com/xiaoyatang/duoformer_TCGA)
**Paper**: [arXiv:2506.12982](https://arxiv.org/abs/2506.12982)

