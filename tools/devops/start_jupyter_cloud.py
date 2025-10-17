#!/usr/bin/env python3
"""
🌐 Cloud-Optimized Jupyter Starter
Automatically configures Jupyter for cloud environments (lightning.ai, Colab, AWS, etc.)

This script provides the best UX for accessing Jupyter notebooks in cloud environments
by automatically detecting the platform and configuring Jupyter appropriately.

Usage:
    python scripts/start_jupyter_cloud.py [notebook_path]
    python scripts/start_jupyter_cloud.py demo_duoformer.ipynb

Features:
- ✅ Auto-detects cloud environment (lightning.ai, Colab, AWS, Azure, GCP, local)
- ✅ Automatically finds available ports
- ✅ Configures Jupyter with optimal settings for each platform
- ✅ Provides platform-specific access instructions
- ✅ Supports both Jupyter Notebook and JupyterLab
- ✅ Handles port conflicts gracefully
"""

import os
import sys
import subprocess
import socket
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CloudJupyterManager:
    """🌐 Manages Jupyter configuration for cloud environments"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = Path.home() / ".jupyter"
        self.cloud_env = self.detect_cloud_environment()
        self.port = self.get_available_port()

    def detect_cloud_environment(self) -> str:
        """🔍 Auto-detect cloud environment"""
        cloud_indicators = {
            "lightning.ai": [
                "lightning.ai",
                "LIGHTNING_STUDIO",
                "LIGHTNING_CLOUD_PROJECT_ID",
            ],
            "google_colab": ["colab", "COLAB_GPU", "COLAB_KAGGLE_TOKEN"],
            "kaggle": ["kaggle", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DATASET_PATH"],
            "aws_sagemaker": [
                "sagemaker",
                "AWS_DEFAULT_REGION",
                "SAGEMAKER_TRAINING_JOB_NAME",
            ],
            "azure": ["azure", "AZURE_CLI_VERSION", "AZURE_RESOURCE_GROUP"],
            "gcp": ["gcp", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"],
            "jupyter_hub": ["JUPYTERHUB_API_URL", "JUPYTERHUB_SERVICE_PREFIX"],
            "binder": ["BINDER_REPO_URL", "BINDER_SERVICE_URL"],
        }

        # Check environment variables
        for cloud, indicators in cloud_indicators.items():
            if any(os.environ.get(indicator) for indicator in indicators):
                return cloud

        # Check hostname patterns
        hostname = socket.gethostname().lower()
        if "lightning" in hostname or "studio" in hostname:
            return "lightning.ai"
        elif "colab" in hostname:
            return "google_colab"
        elif "kaggle" in hostname:
            return "kaggle"
        elif "sagemaker" in hostname:
            return "aws_sagemaker"
        elif "azure" in hostname:
            return "azure"
        elif "gcp" in hostname or "google" in hostname:
            return "gcp"

        return "local"

    def get_available_port(self, start_port: int = 8888, max_attempts: int = 20) -> int:
        """🔌 Find available port with intelligent fallback"""
        # Common Jupyter ports to try
        port_candidates = [8888, 8889, 8890, 8891, 8892, 8080, 8081, 8082, 8083, 8084]

        for port in port_candidates:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.bind(("", port))
                    return port
            except (OSError, socket.error):
                continue

        # If no common ports work, try sequential
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.bind(("", port))
                    return port
            except (OSError, socket.error):
                continue

        return 8888  # Ultimate fallback

    def create_jupyter_config(self) -> Path:
        """⚙️ Create cloud-optimized Jupyter configuration"""
        self.config_dir.mkdir(exist_ok=True)

        # Base configuration for all cloud environments
        base_config = {
            "ServerApp": {
                "ip": "0.0.0.0",
                "port": self.port,
                "open_browser": False,
                "allow_root": True,
                "allow_remote_access": True,
                "disable_check_xsrf": True,
                "token": "",  # Disable token for easier access
                "password": "",  # Disable password
                "base_url": "/",
                "root_dir": str(self.project_root),
            },
            "NotebookApp": {
                "ip": "0.0.0.0",
                "port": self.port,
                "open_browser": False,
                "allow_root": True,
                "allow_remote_access": True,
                "disable_check_xsrf": True,
                "token": "",
                "password": "",
                "base_url": "/",
                "notebook_dir": str(self.project_root),
            },
        }

        # Cloud-specific configurations
        if self.cloud_env == "lightning.ai":
            base_config["ServerApp"].update(
                {
                    "base_url": "/",
                    "root_dir": str(self.project_root),
                    "allow_origin": "*",
                }
            )

        elif self.cloud_env == "google_colab":
            base_config["ServerApp"].update(
                {
                    "base_url": "/",
                    "allow_origin": "*",
                }
            )

        elif self.cloud_env in ["aws_sagemaker", "azure", "gcp"]:
            base_config["ServerApp"].update(
                {
                    "allow_origin": "*",
                    "trust_xheaders": True,
                }
            )

        elif self.cloud_env == "kaggle":
            base_config["ServerApp"].update(
                {
                    "base_url": "/",
                    "allow_origin": "*",
                }
            )

        # Write configuration file
        config_file = self.config_dir / "jupyter_lab_config.py"
        with open(config_file, "w") as f:
            f.write(f"# Auto-generated cloud config for {self.cloud_env}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("c = get_config()\n\n")

            for section, settings in base_config.items():
                f.write(f"# {section} configuration\n")
                for key, value in settings.items():
                    if isinstance(value, str):
                        f.write(f"c.{section}.{key} = '{value}'\n")
                    else:
                        f.write(f"c.{section}.{key} = {value}\n")
                f.write("\n")

        return config_file

    def get_platform_instructions(self) -> Dict[str, Any]:
        """📋 Get platform-specific access instructions"""

        instructions = {
            "lightning.ai": {
                "title": "⚡ Lightning.ai Studio Instructions",
                "steps": [
                    'Click the "Port viewer" icon in the sidebar (bottom icon)',
                    'Click "+ New Port" button',
                    f"Set Port: {self.port}",
                    "Set Address: localhost",
                    'Click "Create" to expose the port',
                    "Click the generated URL to access Jupyter",
                    "The notebook will open automatically in the new tab",
                ],
                "note": "Lightning.ai will automatically handle the port forwarding for you.",
            },
            "google_colab": {
                "title": "📓 Google Colab Instructions",
                "steps": [
                    "Colab will automatically handle port forwarding",
                    "Access URL will be provided automatically",
                    "If running in Colab, upload the notebook directly",
                    "Or use the provided URL to access the notebook",
                ],
                "note": "Colab has built-in Jupyter support - no additional setup needed.",
            },
            "kaggle": {
                "title": "🏆 Kaggle Notebook Instructions",
                "steps": [
                    "Kaggle will automatically handle port forwarding",
                    "Access URL will be provided automatically",
                    "Upload the notebook to Kaggle and run directly",
                    "Or use the provided URL to access the notebook",
                ],
                "note": "Kaggle has built-in Jupyter support.",
            },
            "aws_sagemaker": {
                "title": "☁️ AWS SageMaker Instructions",
                "steps": [
                    "Use SageMaker Studio or SageMaker Notebook instances",
                    "Set up port forwarding: ssh -L 8888:localhost:8888 your-instance",
                    f"Access: http://localhost:8888",
                    "Or use SageMaker Studio's built-in Jupyter interface",
                ],
                "note": "SageMaker provides built-in Jupyter access through Studio.",
            },
            "azure": {
                "title": "☁️ Azure Instructions",
                "steps": [
                    "Use Azure Machine Learning Studio",
                    "Set up port forwarding: ssh -L 8888:localhost:8888 your-vm",
                    f"Access: http://localhost:8888",
                    "Or use Azure ML's built-in Jupyter interface",
                ],
                "note": "Azure ML provides built-in Jupyter access.",
            },
            "gcp": {
                "title": "☁️ Google Cloud Platform Instructions",
                "steps": [
                    "Use Google Cloud AI Platform Notebooks",
                    'Set up port forwarding: gcloud compute ssh --ssh-flag="-L 8888:localhost:8888" your-instance',
                    f"Access: http://localhost:8888",
                    "Or use Vertex AI Workbench's built-in Jupyter interface",
                ],
                "note": "GCP provides built-in Jupyter access through Vertex AI Workbench.",
            },
            "local": {
                "title": "💻 Local Development Instructions",
                "steps": [
                    f"Access: http://localhost:{self.port}",
                    "The browser should open automatically",
                    "If not, manually open the URL above",
                    "Use Ctrl+C to stop the Jupyter server",
                ],
                "note": "Local development - browser should open automatically.",
            },
        }

        return instructions.get(self.cloud_env, instructions["local"])

    def print_startup_banner(self, notebook_path: Optional[str] = None):
        """🚀 Print startup banner with instructions"""

        print("🌐" + "=" * 60)
        print("🚀 CLOUD-OPTIMIZED JUPYTER SETUP")
        print("🌐" + "=" * 60)
        print(f"🔍 Detected Environment: {self.cloud_env}")
        print(f"🔌 Port: {self.port}")
        print(f"📁 Project Root: {self.project_root}")
        if notebook_path:
            print(f"📓 Target Notebook: {notebook_path}")
        print()

        # Get platform instructions
        instructions = self.get_platform_instructions()

        print(f"📋 {instructions['title']}")
        print("-" * 50)
        for i, step in enumerate(instructions["steps"], 1):
            print(f"{i}. {step}")
        print()

        if "note" in instructions:
            print(f"💡 {instructions['note']}")
            print()

        print("🚀 Starting JupyterLab...")
        print("🌐" + "=" * 60)
        print("Press Ctrl+C to stop the server")
        print()

    def start_jupyter(self, notebook_path: Optional[str] = None, use_lab: bool = True):
        """🚀 Start Jupyter with optimal configuration"""

        # Create configuration
        config_file = self.create_jupyter_config()

        # Print startup information
        self.print_startup_banner(notebook_path)

        # Prepare command
        cmd = ["jupyter", "lab" if use_lab else "notebook"]
        cmd.extend(
            [
                f"--port={self.port}",
                "--no-browser",
                "--ip=0.0.0.0",
                "--allow-root",
                f"--config={config_file}",
            ]
        )

        if notebook_path:
            notebook_full_path = self.project_root / notebook_path
            if notebook_full_path.exists():
                cmd.extend(["--notebook-dir", str(notebook_full_path.parent)])
            else:
                print(f"⚠️  Warning: Notebook {notebook_path} not found")

        try:
            # Start Jupyter
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n👋 Jupyter stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting Jupyter: {e}")
            print("💡 Try installing JupyterLab: pip install jupyterlab")
            print("💡 Or try Jupyter Notebook: pip install jupyter notebook")
        except FileNotFoundError:
            print("❌ Jupyter not found. Installing...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "jupyterlab"], check=True
                )
                print("✅ JupyterLab installed. Please run the script again.")
            except subprocess.CalledProcessError as install_error:
                print(f"❌ Failed to install JupyterLab: {install_error}")
                print("💡 Please install manually: pip install jupyterlab")


def main():
    """🎯 Main function"""

    # Parse command line arguments
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Create manager and start Jupyter
    manager = CloudJupyterManager()

    # Check if notebook exists
    if notebook_path:
        notebook_full_path = manager.project_root / notebook_path
        if not notebook_full_path.exists():
            print(f"❌ Error: Notebook '{notebook_path}' not found")
            print(f"📁 Available notebooks in project:")
            for nb in manager.project_root.glob("*.ipynb"):
                print(f"   - {nb.name}")
            sys.exit(1)

    # Start Jupyter
    manager.start_jupyter(notebook_path)


if __name__ == "__main__":
    main()
