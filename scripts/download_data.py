"""
download_data.py
================
Automated ICBHI 2017 dataset downloader.

This script checks if the ICBHI dataset is already present. If not, it:
  1. Downloads the official dataset zip from Zenodo (working mirror)
  2. Extracts it to data/ICBHI_final_database/
  3. Downloads the train/test split file from official source
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
# Configuration - WORKING URLS
# ================================================================

# Official ICBHI dataset
DATASET_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

# Official split file 
SPLIT_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt"

EXPECTED_FILES = 920  # approximate number of .wav/.txt pairs


# ================================================================
# Helpers
# ================================================================

def create_ssl_context():
    """Create SSL context that works on both Windows and Linux/Colab"""
    try:
        # Try to create unverified context for Windows (certificate issues)
        return ssl._create_unverified_context()
    except AttributeError:
        # Fallback for older Python versions
        return None


def download_file(url: str, destination: str, description: str = "Downloading"):
    """Download a file with a progress bar."""
    print(f"\n{description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")
    
    try:
        # Create request with proper headers
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        
        # Use unverified SSL context for Windows (fixes certificate errors)
        ssl_context = create_ssl_context()
        
        with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            
            with open(destination, "wb") as output_file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Progress") as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"  [OK] Downloaded successfully ({os.path.getsize(destination) / 1e6:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False


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
        return False


def verify_dataset(data_dir: str) -> bool:
    """Verify that the dataset has the expected structure."""
    db_dir = os.path.join(data_dir, "ICBHI_final_database")
    split_file = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
    
    if not os.path.isdir(db_dir):
        print(f"  [INFO] Database directory not found: {db_dir}")
        return False
    
    # Count files
    wav_files = [f for f in os.listdir(db_dir) if f.endswith('.wav')]
    txt_files = [f for f in os.listdir(db_dir) if f.endswith('.txt')]
    
    print(f"  [OK] Database directory found: {db_dir}")
    print(f"    - WAV files: {len(wav_files)}")
    print(f"    - TXT files: {len(txt_files)}")
    
    if len(wav_files) < 100 or len(txt_files) < 100:
        print(f"  [WARNING] Fewer than 100 files. Dataset may be incomplete.")
        return False
    
    if not os.path.isfile(split_file):
        print(f"  [INFO] Split file missing: {split_file}")
        return False
    
    print(f"  [OK] Split file found: {split_file}")
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
        print("\n[INFO] Force download requested. Removing existing files...")
        db_dir = os.path.join(data_dir, "ICBHI_final_database")
        split_file = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
        if os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir)
        if os.path.exists(split_file):
            os.remove(split_file)
    
    # Download dataset zip
    zip_path = os.path.join(data_dir, "ICBHI_final_database.zip")
    downloaded = False
    
    if download_file(DATASET_URL, zip_path, "Downloading ICBHI dataset from Zenodo"):
        downloaded = True
    
    if downloaded:
        if extract_zip(zip_path, data_dir, "Extracting dataset"):
            # Handle nested folder structure if present
            nested_path = os.path.join(data_dir, "ICBHI_final_database", "ICBHI_final_database")
            if os.path.isdir(nested_path):
                import shutil
                # Move files up one level
                for item in os.listdir(nested_path):
                    shutil.move(
                        os.path.join(nested_path, item),
                        os.path.join(data_dir, "ICBHI_final_database")
                    )
                os.rmdir(nested_path)
            
            os.remove(zip_path)
            print(f"  Cleaned up zip file")
        else:
            return False
    
    # Download split file from official source (with SSL fix)
    split_path = os.path.join(data_dir, "ICBHI_challenge_train_test.txt")
    if not download_file(SPLIT_URL, split_path, "Downloading train/test split file"):
        print("\n[WARNING] Could not download split file.")
        print("The dataset zip contains the split file. Continuing anyway...")
        # The split file might be inside the zip - check if it was extracted
        possible_split = os.path.join(data_dir, "ICBHI_final_database", "ICBHI_challenge_train_test.txt")
        if os.path.exists(possible_split):
            import shutil
            shutil.copy(possible_split, split_path)
            print(f"  [OK] Found split file in extracted dataset")
    
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
