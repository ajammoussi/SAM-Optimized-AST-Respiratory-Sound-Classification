#!/usr/bin/env python
"""
setup.py
========
Quick setup script for local development:
  1. Ensures dataset is downloaded
  2. Optionally creates a virtual environment
  3. Installs requirements

Usage:
    python scripts/setup.py
    python scripts/setup.py --no-venv
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    if description:
        print(f"\n{description}...")
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup ICBHI AST-SAM project")
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Skip virtual environment setup (use system Python)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip package installation",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ICBHI AST-SAM — Local Setup")
    print("=" * 70)

    # Step 1: Download dataset
    print("\n[OK] Step 1: Ensuring dataset...")
    if not run_command(
        [sys.executable, str(SCRIPT_DIR / "download_data.py")],
        "Downloading dataset",
    ):
        print("[WARNING] Dataset setup had issues. Continuing anyway...")

    # Step 2: Create virtual environment (optional)
    if not args.no_venv:
        venv_name = "venv"
        if not os.path.exists(venv_name):
            print(f"\n[OK] Step 2: Creating virtual environment '{venv_name}'...")
            if not run_command([sys.executable, "-m", "venv", venv_name]):
                print("[ERROR] Failed to create virtual environment")
                sys.exit(1)
        else:
            print(f"\n[OK] Step 2: Virtual environment '{venv_name}' already exists")

        # Activate venv and install requirements
        if platform.system() == "Windows":
            python_cmd = os.path.join(venv_name, "Scripts", "python")
            activate_cmd = os.path.join(venv_name, "Scripts", "activate.bat")
        else:
            python_cmd = os.path.join(venv_name, "bin", "python")
            activate_cmd = os.path.join(venv_name, "bin", "activate")

        print(f"\n  Virtual environment created at: {venv_name}/")
        print(f"  Activate with:")
        if platform.system() == "Windows":
            print(f"    {activate_cmd}")
        else:
            print(f"    source {activate_cmd}")

    else:
        python_cmd = sys.executable
        print("\n[OK] Step 2: Skipping virtual environment (using system Python)")

    # Step 3: Install requirements
    if not args.no_install:
        print("\n[OK] Step 3: Installing requirements...")
        requirements_path = REPO_ROOT / "requirements.txt"
        if not run_command([python_cmd, "-m", "pip", "install", "-r", str(requirements_path), "-q"]):
            print("[WARNING] Package installation had issues")

    print("\n" + "=" * 70)
    print("[SUCCESS] Setup complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Preprocess data: python scripts/preprocess.py")
    print("  2. Train model:     python scripts/train.py --epochs 20 --batch_size 16")
    print("  3. Evaluate:        python scripts/evaluate.py")


if __name__ == "__main__":
    main()
