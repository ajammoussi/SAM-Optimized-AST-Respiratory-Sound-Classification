"""
download_data.py
================
Automated ICBHI 2017 dataset downloader.

This script checks if the ICBHI dataset is already present. If not, it:
  1. Downloads the official dataset zip from the ICBHI Challenge website
  2. Extracts it to data/ICBHI_final_database/
  3. Downloads the train/test split file
  4. Verifies file integrity

Works on both local machines and Google Colab.

Usage:
    python scripts/download_data.py [--data_dir ./data]
"""

import os
import sys
import argparse
import ssl
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from tqdm import tqdm


# ================================================================
# Configuration
# ================================================================

DATASET_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
SPLIT_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt"

EXPECTED_FILES = 920  # approximate number of .wav/.txt pairs


# ================================================================
# Helpers
# ================================================================

class ProgressBar(urllib.request.FancyURLopener):
    """Custom URL opener with progress bar."""
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise urllib.error.HTTPError(url, errcode, errmsg, headers, fp)


def download_file(url: str, destination: str, description: str = "Downloading"):
    """Download a file with a progress bar."""
    print(f"\n{description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")
    
    try:
        def _stream_download(context):
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, context=context) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                bar_length = 40

                with open(destination, "wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = min(downloaded * 100 // total_size, 100)
                            filled = int(bar_length * percent // 100)
                            bar = "#" * filled + "-" * (bar_length - filled)
                            print(
                                f"\r  [{bar}] {percent}% "
                                f"({downloaded / 1e9:.1f}GB / {total_size / 1e9:.1f}GB)",
                                end="",
                                flush=True,
                            )

        try:
            _stream_download(ssl.create_default_context())
        except ssl.SSLError as ssl_error:
            print(f"\n  [WARNING] SSL verification failed: {ssl_error}")
            print("  [WARNING] Retrying with certificate verification disabled...")
            _stream_download(ssl._create_unverified_context())

        print()
        print(f"  [OK] Downloaded successfully")

    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise


def extract_zip(zip_path: str, extract_to: str, description: str = "Extracting"):
    """Extract a zip file with progress feedback."""
    print(f"\n{description}...")
    print(f"  Source: {zip_path}")
    print(f"  Target: {extract_to}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc="  Extracting", unit="file"):
                zip_ref.extract(member, extract_to)
        print(f"  [OK] Extraction complete")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        raise


def verify_dataset(data_dir: str) -> bool:
    """Verify that the dataset has the expected structure."""
    db_dir = os.path.join(data_dir, "ICBHI_final_database")
    split_file = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
    
    if not os.path.isdir(db_dir):
        print(f"  [ERROR] Database directory missing: {db_dir}")
        return False
    
    if not os.path.isfile(split_file):
        print(f"  [ERROR] Split file missing: {split_file}")
        return False
    
    # Count files in the database directory
    wav_files = [f for f in os.listdir(db_dir) if f.endswith('.wav')]
    txt_files = [f for f in os.listdir(db_dir) if f.endswith('.txt')]
    
    print(f"  [OK] Database directory found: {db_dir}")
    print(f"    - WAV files: {len(wav_files)}")
    print(f"    - TXT files: {len(txt_files)}")
    print(f"  [OK] Split file found: {split_file}")
    
    if len(wav_files) < 100 or len(txt_files) < 100:
        print(f"  [WARNING] Warning: Fewer than 100 files found. Dataset may be incomplete.")
        return False
    
    return True


# ================================================================
# Main
# ================================================================

def download_data(data_dir: str = "./data", force: bool = False):
    """
    Main function: check and download dataset if needed.
    
    Args:
        data_dir: Path to data directory
        force: If True, re-download even if files exist
    
    Returns:
        True if dataset is ready, False otherwise
    """
    print("=" * 70)
    print("ICBHI 2017 Dataset Downloader")
    print("=" * 70)
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if dataset already exists
    if not force and verify_dataset(data_dir):
        print("\n[SUCCESS] Dataset is ready!")
        return True
    
    if force:
        print("\n[WARNING] Force download requested. Removing existing files...")
        db_dir = os.path.join(data_dir, "ICBHI_final_database")
        split_file = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
        if os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir)
            print(f"  Removed {db_dir}")
        if os.path.exists(split_file):
            os.remove(split_file)
            print(f"  Removed {split_file}")
    else:
        print("\n[WARNING] Dataset not found. Downloading...")
    
    # Download dataset zip
    zip_path = os.path.join(data_dir, "ICBHI_final_database.zip")
    try:
        download_file(DATASET_URL, zip_path, "Downloading ICBHI dataset zip")
        
        # Extract
        extract_zip(zip_path, data_dir, "Extracting dataset")
        
        # Handle nested folder if the zip contains ICBHI_final_database/ICBHI_final_database/
        nested_path = os.path.join(data_dir, "ICBHI_final_database", "ICBHI_final_database")
        if os.path.isdir(nested_path):
            import shutil
            # Move files up one level
            target = os.path.join(data_dir, "ICBHI_final_database_tmp")
            shutil.move(nested_path, target)
            # Remove the old parent dir
            shutil.rmtree(os.path.join(data_dir, "ICBHI_final_database"))
            # Rename back
            shutil.move(target, os.path.join(data_dir, "ICBHI_final_database"))
        
        # Clean up zip
        os.remove(zip_path)
        print(f"  Cleaned up zip file")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to download/extract dataset: {e}")
        return False
    
    # Download split file
    split_file = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
    try:
        download_file(SPLIT_URL, split_file, "Downloading train/test split file")
    except Exception as e:
        print(f"\n[ERROR] Failed to download split file: {e}")
        return False
    
    # Final verification
    print("\nVerifying downloaded dataset...")
    if verify_dataset(data_dir):
        print("\n[SUCCESS] Dataset is ready!")
        return True
    else:
        print("\n[ERROR] Dataset verification failed!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify ICBHI 2017 dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    
    args = parser.parse_args()
    
    success = download_data(args.data_dir, force=args.force)
    sys.exit(0 if success else 1)
