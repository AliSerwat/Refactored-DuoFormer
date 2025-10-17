#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored DuoFormer Environment Setup Script

This script automates the dependency management process using pip-tools
to ensure reproducible and conflict-free installations.

Refactored for general medical imaging applications.

Usage:
    python setup_environment.py [--compile-only] [--install-only] [--verbose]

Arguments:
    --compile-only: Only compile requirements.in to requirements.txt
    --install-only: Only install from existing requirements.txt
    --verbose: Show detailed output
    --dry-run: Simulate installation without actually installing

Original work: https://github.com/xiaoyatang/duoformer_TCGA
Paper: https://arxiv.org/abs/2506.12982
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, List

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        # Try to set UTF-8 encoding for Windows console
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Fallback: disable emoji if encoding fails
        pass


# Emoji support with fallback for Windows
def safe_print(text: str):
    """Print with emoji support and Windows fallback."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace emojis with text equivalents
        emoji_map = {
            "🔧": "[SETUP]",
            "✅": "[OK]",
            "❌": "[ERROR]",
            "📦": "[PACKAGE]",
            "🔍": "[CHECK]",
            "⚠️": "[WARNING]",
            "🚀": "[START]",
            "🎉": "[SUCCESS]",
            "📝": "[NOTE]",
            "📊": "[INFO]",
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        print(text)


class DependencyManager:
    """Manages Python dependencies using pip-tools for reproducible environments."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        self.requirements_in = self.project_root / "requirements.in"
        self.requirements_txt = self.project_root / "requirements.txt"

    def _run_command(
        self, command: List[str], description: str, capture_output: bool = False
    ) -> Optional[str]:
        """Execute a shell command with error handling."""
        safe_print(f"\n{'=' * 70}")
        safe_print(f"🔧 {description}")
        safe_print(f"{'=' * 70}")

        if self.verbose:
            safe_print(f"Command: {' '.join(command)}\n")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root,
            )

            if capture_output and result.stdout:
                return result.stdout

            safe_print(f"✅ {description} completed successfully!\n")
            return None

        except subprocess.CalledProcessError as e:
            safe_print(f"❌ Error during: {description}")
            safe_print(f"Error message: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            sys.exit(1)
        except FileNotFoundError:
            safe_print(f"❌ Command not found: {command[0]}")
            safe_print("Please ensure Python and pip are properly installed.")
            sys.exit(1)

    def upgrade_pip(self):
        """Upgrade pip to the latest version."""
        self._run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrading pip to latest version",
        )

    def install_pip_tools(self):
        """Install pip-tools for dependency management."""
        self._run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip-tools"],
            "Installing pip-tools",
        )

    def compile_requirements(self, upgrade: bool = True):
        """Compile requirements.in to requirements.txt with pinned versions."""
        if not self.requirements_in.exists():
            safe_print(f"❌ Error: {self.requirements_in} not found!")
            sys.exit(1)

        command = [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            str(self.requirements_in),
            "-o",
            str(self.requirements_txt),
            "--resolver=backtracking",
            "--generate-hashes",  # Add hashes for security
        ]

        if upgrade:
            command.append("--upgrade")

        if self.verbose:
            command.append("--verbose")

        self._run_command(command, "Compiling requirements.in to requirements.txt")

        # Display summary
        if self.requirements_txt.exists():
            with open(self.requirements_txt, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if not l.strip().startswith("#")]
                safe_print(f"📦 Generated {len(lines)} pinned dependencies")

    def preview_installation(self):
        """Preview what will be installed/upgraded without actually installing."""
        safe_print("\n" + "=" * 70)
        safe_print("🔍 PREVIEWING INSTALLATION (DRY RUN)")
        safe_print("=" * 70 + "\n")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(self.requirements_txt),
                    "--upgrade-strategy",
                    "only-if-needed",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=self.project_root,
            )

            if result.stdout:
                safe_print(result.stdout)
            if result.returncode == 0:
                safe_print("✅ Preview completed. Review the changes above.")
            else:
                safe_print(
                    "⚠️  Some issues detected during preview. Proceeding anyway..."
                )

        except Exception as e:
            safe_print(f"⚠️  Could not complete preview: {e}")

    def install_requirements(self, dry_run: bool = False):
        """Install dependencies from requirements.txt."""
        if not self.requirements_txt.exists():
            safe_print(f"❌ Error: {self.requirements_txt} not found!")
            safe_print("Run with --compile-only first to generate requirements.txt")
            sys.exit(1)

        if dry_run:
            self.preview_installation()
            return

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(self.requirements_txt),
            "--upgrade-strategy",
            "only-if-needed",
        ]

        if self.verbose:
            command.append("--verbose")

        self._run_command(command, "Installing dependencies from requirements.txt")

    def validate_environment(self):
        """Validate the installed environment for conflicts."""
        self._run_command(
            [sys.executable, "-m", "pip", "check"],
            "Validating environment for conflicts",
        )

    def show_summary(self):
        """Display a summary of installed packages."""
        safe_print("\n" + "=" * 70)
        safe_print("📊 INSTALLATION SUMMARY")
        safe_print("=" * 70 + "\n")

        result = self._run_command(
            [sys.executable, "-m", "pip", "list", "--format", "columns"],
            "Listing installed packages",
            capture_output=True,
        )

        if result:
            safe_print(result)

    def setup_complete_workflow(self, dry_run: bool = False):
        """Execute the complete dependency setup workflow."""
        safe_print("\n" + "=" * 70)
        safe_print("🚀 DUOFORMER ENVIRONMENT SETUP")
        safe_print("=" * 70)
        safe_print(f"Project: {self.project_root}")
        safe_print(f"Python: {sys.version.split()[0]}")
        safe_print(f"Executable: {sys.executable}\n")

        # Step 1: Upgrade pip
        self.upgrade_pip()

        # Step 2: Install pip-tools
        self.install_pip_tools()

        # Step 3: Compile requirements
        self.compile_requirements(upgrade=True)

        # Step 4: Install requirements
        if dry_run:
            self.preview_installation()
        else:
            self.install_requirements()

        # Step 5: Validate environment
        if not dry_run:
            self.validate_environment()

        # Step 6: Show summary
        if not dry_run and self.verbose:
            self.show_summary()

        safe_print("\n" + "=" * 70)
        safe_print("🎉 ENVIRONMENT SETUP COMPLETE!")
        safe_print("=" * 70)
        safe_print("\n✅ Your DuoFormer environment is ready to use!")
        safe_print("📝 Next steps:")
        safe_print("   1. Open demo_duoformer.ipynb")
        safe_print("   2. Run the cells to explore DuoFormer architecture")
        safe_print("   3. Adapt to your medical imaging dataset\n")


def main():
    """Main entry point for the setup script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DuoFormer Environment Setup - Automated dependency management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_environment.py                    # Full setup
  python setup_environment.py --compile-only     # Only compile requirements
  python setup_environment.py --install-only     # Only install dependencies
  python setup_environment.py --dry-run          # Preview changes
  python setup_environment.py --verbose          # Detailed output

For more information, visit:
  GitHub: https://github.com/xiaoyatang/duoformer_TCGA
  Paper: https://arxiv.org/abs/2506.12982
        """,
    )

    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile requirements.in to requirements.txt",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install from existing requirements.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate installation without actually installing",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Initialize manager
    manager = DependencyManager(verbose=args.verbose)

    try:
        if args.compile_only:
            manager.upgrade_pip()
            manager.install_pip_tools()
            manager.compile_requirements(upgrade=True)
        elif args.install_only:
            manager.install_requirements(dry_run=args.dry_run)
            if not args.dry_run:
                manager.validate_environment()
        else:
            # Full workflow
            manager.setup_complete_workflow(dry_run=args.dry_run)

    except KeyboardInterrupt:
        safe_print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
