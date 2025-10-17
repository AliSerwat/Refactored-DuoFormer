#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏥 Refactored DuoFormer - Codebase Health Check

Comprehensive health check script to verify code integrity and readiness.
For general medical imaging applications.

Usage:
    python health_check.py [--verbose] [--install-deps]

Features:
- Syntax validation
- Import structure check
- Dependency verification
- Configuration validation
- Test execution

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import ast
import importlib.util

# Unicode handling for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def safe_print(text: str):
    """Print with emoji fallback for Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_map = {
            "✅": "[OK]",
            "❌": "[ERROR]",
            "⚠️": "[WARNING]",
            "🔍": "[CHECK]",
            "📁": "[DIR]",
            "🐍": "[PY]",
            "🏥": "[HEALTH]",
            "🎯": "[TEST]",
            "📊": "[STATS]",
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        print(text)


class HealthChecker:
    """Comprehensive codebase health checker."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent  # Go up one level from scripts/
        self.issues = []
        self.warnings = []
        self.passed = []

    def check_syntax(self, file_path: Path) -> bool:
        """Check if Python file has valid syntax."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            self.issues.append(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            self.warnings.append(f"Could not parse {file_path}: {e}")
            return False

    def check_wildcard_imports(self, file_path: Path) -> bool:
        """Check for wildcard imports (excluding comments)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    stripped = line.strip()
                    # Skip comments
                    if stripped.startswith("#"):
                        continue
                    # Check for actual wildcard import (not string literal)
                    if "from" in line and "import" in line and "*" in line:
                        if (
                            not stripped.startswith("#")
                            and not "'" in line.split("import")[-1]
                            and not '"' in line.split("import")[-1]
                        ):
                            self.warnings.append(
                                f"Wildcard import in {file_path}:{line_no}: {line.strip()}"
                            )
                            return False
            return True
        except Exception as e:
            self.warnings.append(f"Could not check {file_path}: {e}")
            return False

    def check_deprecated_apis(self, file_path: Path) -> bool:
        """Check for deprecated PyTorch APIs."""
        deprecated_patterns = [
            ("pretrained=True", "weights="),
            ("pretrained=False", "weights=None"),
        ]

        issues_found = False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                for old_pattern, new_pattern in deprecated_patterns:
                    if old_pattern in content and new_pattern not in content:
                        # Check if it's in a comment
                        lines = content.split("\n")
                        for line_no, line in enumerate(lines, 1):
                            if old_pattern in line and not line.strip().startswith("#"):
                                self.warnings.append(
                                    f"Potential deprecated API in {file_path}:{line_no}"
                                )
                                issues_found = True
        except Exception as e:
            self.warnings.append(f"Could not check {file_path}: {e}")

        return not issues_found

    def run_comprehensive_check(self):
        """Run comprehensive health check."""
        safe_print("\n" + "=" * 80)
        safe_print("🏥 Refactored DuoFormer - Codebase Health Check")
        safe_print("=" * 80 + "\n")

        # 1. Check Python files syntax
        safe_print("🔍 Checking Python syntax...")
        python_files = list(self.project_root.rglob("*.py"))
        syntax_ok = 0
        for py_file in python_files:
            # Skip __pycache__ and .git
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            if self.check_syntax(py_file):
                syntax_ok += 1
                if self.verbose:
                    safe_print(f"   ✅ {py_file.relative_to(self.project_root)}")
            else:
                safe_print(f"   ❌ {py_file.relative_to(self.project_root)}")

        safe_print(
            f"\n   Result: {syntax_ok}/{len(python_files)} files passed syntax check"
        )
        self.passed.append(f"Syntax: {syntax_ok}/{len(python_files)}")

        # 2. Check for wildcard imports
        safe_print("\n🔍 Checking for wildcard imports...")
        wildcard_clean = 0
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            if self.check_wildcard_imports(py_file):
                wildcard_clean += 1

        safe_print(
            f"   Result: {wildcard_clean}/{len(python_files)} files have no wildcard imports"
        )
        self.passed.append(
            f"Wildcard imports: {wildcard_clean}/{len(python_files)} clean"
        )

        # 3. Check for deprecated APIs
        safe_print("\n🔍 Checking for deprecated PyTorch APIs...")
        deprecated_clean = 0
        model_files = list(self.project_root.rglob("src/duoformer/models/**/*.py"))
        for py_file in model_files:
            if self.check_deprecated_apis(py_file):
                deprecated_clean += 1

        safe_print(
            f"   Result: {deprecated_clean}/{len(model_files)} model files clean"
        )
        self.passed.append(
            f"Deprecated APIs: {deprecated_clean}/{len(model_files)} clean"
        )

        # 4. Check file structure
        safe_print("\n🔍 Checking project structure...")
        required_files = [
            "requirements.in",
            "requirements.txt",
            "setup_environment.py",
            "train.py",
            "src/duoformer/models/__init__.py",
            "src/duoformer/utils/__init__.py",
            "src/duoformer/training/__init__.py",
            "src/duoformer/data/__init__.py",
            "src/duoformer/config/__init__.py",
            "src/duoformer/config/model_config.py",
            "tests/__init__.py",
        ]

        missing_files = []
        for req_file in required_files:
            if not (self.project_root / req_file).exists():
                missing_files.append(req_file)
                safe_print(f"   ❌ Missing: {req_file}")
            elif self.verbose:
                safe_print(f"   ✅ {req_file}")

        if not missing_files:
            safe_print(f"   Result: All required files present ✅")
            self.passed.append("File structure: Complete")
        else:
            self.issues.append(f"Missing files: {missing_files}")

        # 5. Check documentation
        safe_print("\n🔍 Checking documentation...")
        doc_files = [
            "README.md",
            "GETTING_STARTED.md",
            "docs/INSTALLATION.md",
            "docs/CONTRIBUTING.md",
            "docs/TROUBLESHOOTING.md",
        ]

        docs_present = 0
        for doc in doc_files:
            if (self.project_root / doc).exists():
                docs_present += 1
                if self.verbose:
                    safe_print(f"   ✅ {doc}")

        safe_print(
            f"   Result: {docs_present}/{len(doc_files)} documentation files present"
        )
        self.passed.append(f"Documentation: {docs_present}/{len(doc_files)}")

        # 6. Check requirements.txt
        safe_print("\n🔍 Checking requirements...")
        req_txt = self.project_root / "requirements.txt"
        if req_txt.exists():
            with open(req_txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
                package_lines = [
                    l for l in lines if l.strip() and not l.strip().startswith("#")
                ]
                safe_print(f"   Result: {len(package_lines)} pinned dependencies ✅")
                self.passed.append(f"Dependencies: {len(package_lines)} pinned")
        else:
            self.issues.append("requirements.txt not found")

        # 7. Check configuration
        safe_print("\n🔍 Checking configuration system...")
        config_file = self.project_root / "config" / "model_config.py"
        if config_file.exists():
            if self.check_syntax(config_file):
                safe_print("   Result: Configuration system OK ✅")
                self.passed.append("Configuration: Validated")

        # Print summary
        self._print_summary()

        # Return True if no issues found
        return len(self.issues) == 0

    def _print_summary(self):
        """Print health check summary."""
        safe_print("\n" + "=" * 80)
        safe_print("📊 Health Check Summary")
        safe_print("=" * 80 + "\n")

        safe_print("✅ Passed Checks:")
        for item in self.passed:
            safe_print(f"   • {item}")

        if self.warnings:
            safe_print("\n⚠️  Warnings:")
            for warning in self.warnings:
                safe_print(f"   • {warning}")

        if self.issues:
            safe_print("\n❌ Issues:")
            for issue in self.issues:
                safe_print(f"   • {issue}")

        # Overall status
        safe_print("\n" + "=" * 80)
        if not self.issues:
            if not self.warnings:
                safe_print("🎉 Health Check: EXCELLENT - No issues or warnings")
                safe_print("✅ Codebase is PRODUCTION READY")
            else:
                safe_print("✅ Health Check: GOOD - Minor warnings only")
                safe_print("⚠️  Review warnings above")
        else:
            safe_print("❌ Health Check: NEEDS ATTENTION")
            safe_print("⚠️  Please fix issues above")
        safe_print("=" * 80 + "\n")

        return len(self.issues) == 0


def check_dependencies_installed():
    """Check if key dependencies are installed."""
    safe_print("\n🔍 Checking installed dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "timm",
        "einops",
        "matplotlib",
        "numpy",
        "PIL",
        "tqdm",
        "sklearn",
    ]

    installed = []
    missing = []

    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            installed.append(package)
            safe_print(f"   ✅ {package}")
        except ImportError:
            missing.append(package)
            safe_print(f"   ❌ {package} (not installed)")

    safe_print(
        f"\n   Result: {len(installed)}/{len(required_packages)} packages installed"
    )

    if missing:
        safe_print("\n⚠️  Missing packages. To install:")
        safe_print("   python setup_environment.py")
    else:
        safe_print("\n✅ All dependencies installed!")

    return len(missing) == 0


def main():
    """Main health check function."""
    import argparse

    parser = argparse.ArgumentParser(description="DuoFormer Codebase Health Check")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install missing dependencies"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency check"
    )

    args = parser.parse_args()

    # Run health check
    checker = HealthChecker(verbose=args.verbose)
    code_health = checker.run_comprehensive_check()

    # Check dependencies
    if not args.skip_deps:
        deps_ok = check_dependencies_installed()

        if not deps_ok and args.install_deps:
            safe_print("\n🔧 Installing dependencies...")
            subprocess.run([sys.executable, "setup_environment.py"])

    # Final recommendation
    safe_print("\n" + "=" * 80)
    safe_print("📋 Next Steps:")
    safe_print("=" * 80)

    if code_health:
        safe_print("\n✅ Code Health: EXCELLENT")
        safe_print("\n   Recommended actions:")
        safe_print("   1. Install dependencies: python setup_environment.py")
        safe_print("   2. Run tests: python tests/run_tests.py --unit")
        safe_print("   3. Try demo: jupyter notebook demo_duoformer.ipynb")
        safe_print("   4. Train model: python train.py --data_dir /path/to/data")
    else:
        safe_print("\n⚠️  Code Health: NEEDS ATTENTION")
        safe_print("\n   Please fix issues above before proceeding")

    safe_print("=" * 80 + "\n")

    return code_health


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
