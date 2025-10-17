#!/usr/bin/env python3
"""
🔧 JupyterLab Extensions Installer for Enhanced UX
Installs useful JupyterLab extensions for better cloud experience

This script enhances the Jupyter experience with useful extensions
that improve productivity and user experience in cloud environments.

Usage:
    python scripts/install_jupyter_extensions.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """🔄 Run command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"📤 Output: {e.stdout}")
        if e.stderr:
            print(f"📤 Error: {e.stderr}")
        return False


def install_jupyterlab_extensions():
    """📦 Install JupyterLab extensions for enhanced UX"""

    print("🔧" + "=" * 60)
    print("🚀 INSTALLING JUPYTERLAB EXTENSIONS")
    print("🔧" + "=" * 60)

    print("📦 Installing JupyterLab extensions from requirements.txt...")
    print("   📝 Extensions are managed via pip-compile for compatibility")

    # Install from requirements.txt (which includes the extensions)
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing JupyterLab extensions from requirements.txt",
    )

    if success:
        print(f"\n🎉 JupyterLab extensions installed successfully!")
        print(f"💡 Extensions included:")
        print(f"   ✅ jupyterlab-git - Git integration")
        print(f"   ✅ jupyterlab-code-formatter - Code formatting")
        print(f"   ✅ jupyterlab-lsp - Language Server Protocol")
        print(f"   ✅ jupyterlab-snippets - Code snippets")
        print(f"   ✅ jupyterlab-widgets - Interactive widgets")
        print(f"💡 Restart JupyterLab to see the new extensions")
    else:
        print(f"\n⚠️  Extension installation failed")
        print(f"💡 Try running: pip install -r requirements.txt")


def install_jupyter_widgets():
    """🎛️ Install Jupyter widgets for interactive functionality"""

    print(f"\n🎛️ Setting up Jupyter Widgets...")
    print("   📝 Widgets are already included in requirements.txt")

    # Enable widget extensions (widgets are already installed via requirements.txt)
    # Note: In JupyterLab 4.x, widgets are automatically enabled
    print("   ✅ Widget extensions are automatically enabled in JupyterLab 4.x")


def create_jupyter_config():
    """⚙️ Create enhanced Jupyter configuration"""

    print(f"\n⚙️ Creating enhanced Jupyter configuration...")

    config_dir = Path.home() / ".jupyter"
    config_dir.mkdir(exist_ok=True)

    # Enhanced configuration
    config_content = """
# Enhanced Jupyter configuration for better UX
c = get_config()

# Server configuration
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.allow_remote_access = True

# Notebook configuration
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
c.NotebookApp.allow_remote_access = True

# Enhanced features
c.NotebookApp.iopub_data_rate_limit = 1000000000  # 1GB
c.ServerApp.iopub_data_rate_limit = 1000000000    # 1GB

# Widget support
c.NotebookApp.nbserver_extensions = {
    'jupyterlab_git': True,
    'jupyterlab_lsp': True,
}

# Theme and appearance
c.ServerApp.theme = 'light'
c.ServerApp.collaborative = True

# Security (for cloud environments)
c.ServerApp.disable_check_xsrf = True
c.ServerApp.allow_origin = '*'
"""

    config_file = config_dir / "jupyter_lab_config.py"
    with open(config_file, "w") as f:
        f.write(config_content.strip())

    print(f"✅ Enhanced Jupyter configuration created at: {config_file}")


def main():
    """🎯 Main function"""

    print("🌐 JupyterLab Extensions & UX Enhancement")
    print("=" * 60)

    # Install extensions
    install_jupyterlab_extensions()

    # Install widgets
    install_jupyter_widgets()

    # Create enhanced config
    create_jupyter_config()

    print(f"\n🎉 JupyterLab enhancement completed!")
    print(
        f"💡 Use 'python scripts/start_jupyter_cloud.py' to start Jupyter with enhanced features"
    )
    print(f"🔄 Restart JupyterLab to see all new extensions and features")


if __name__ == "__main__":
    main()
