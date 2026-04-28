"""
preprocess.py
=============
Reads the raw ICBHI 2017 audio files, segments them into respiratory cycles,
applies cyclic padding to 8 seconds, then runs the ASTFeatureExtractor **once**
for every sample and saves the resulting log-mel spectrogram tensors to a .npz.

Why run the feature extractor here instead of inside the DataLoader?
  Running ASTFeatureExtractor inside __getitem__ was the biggest training
  bottleneck: it recomputed log-mel spectrograms on the CPU for every sample
  on every epoch, starving the GPU while it waited.  By doing it once here we
  pay the CPU cost once (a few minutes) and the DataLoader becomes trivially fast.

Output .npz keys:
  X_train  : float32, (N_train, freq_bins, time_frames)  — pre-computed spectrograms
  y_train  : int64,   (N_train,)
  device_train : int64, (N_train,)
  X_test   : float32, (N_test,  freq_bins, time_frames)
  y_test   : int64,   (N_test,)
  device_test  : int64, (N_test,)

Usage:
  python preprocess.py \
      --data_dir  ./data/ICBHI_final_database \
      --split_file ./data/ICBHI_challenge_train_test.txt \
      --output    ./icbhi_ast_16k_8s_spectrograms.npz
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import ASTFeatureExtractor

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #
TARGET_SR = 16_000          # Hz — required by the AST pre-trained weights
TARGET_DURATION = 8         # seconds
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION   # 128 000 samples

DEVICE_MAP = {
    "AKGC417L": 0,
    "LittC2SE": 1,
    "Litt3200": 2,
    "Meditron": 3,
}


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def get_device_id(filename: str) -> int:
    """Extract the recording device from the filename convention."""
    parts = filename.split("_")
    dev_name = parts[-1]
    return DEVICE_MAP.get(dev_name, -1)


def cyclic_padding(wav: np.ndarray, target_len: int) -> np.ndarray:
    """
    Repeat the signal until it reaches target_len samples.
    Much better than zero-padding because the model always sees a
    signal-dense input, which prevents dilution of pathological features.
    """
    curr_len = len(wav)
    if curr_len >= target_len:
        return wav[:target_len]
    repeat_count = (target_len // curr_len) + 1
    return np.tile(wav, repeat_count)[:target_len]


# --------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------- #

def process_data(args):
    print("=" * 60)
    print("ICBHI 2017 Preprocessing — Pre-computing AST Spectrograms")
    print("=" * 60)

    # Load the feature extractor once (downloads ~few MB from HuggingFace)
    print("\nLoading ASTFeatureExtractor …")
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    split_df = pd.read_csv(
        args.split_file, sep="\t", names=["filename", "set_type"]
    )

    X_train, y_train, device_train = [], [], []
    X_test,  y_test,  device_test  = [], [], []

    stats = {"Normal": 0, "Crackle": 0, "Wheeze": 0, "Both": 0}
    skipped = 0

    print(f"\nProcessing {len(split_df)} recordings …\n")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        fname    = row["filename"]
        set_type = row["set_type"]

        wav_path = os.path.join(args.data_dir, fname + ".wav")
        txt_path = os.path.join(args.data_dir, fname + ".txt")

        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            skipped += 1
            continue

        # Load audio at 16 kHz
        audio, _ = librosa.load(wav_path, sr=TARGET_SR)
        dev_id   = get_device_id(fname)

        # Read cycle annotations
        anns = pd.read_csv(
            txt_path, sep="\t", names=["start", "end", "crackle", "wheeze"]
        )

        for _, ann in anns.iterrows():
            start = int(ann["start"] * TARGET_SR)
            end   = int(ann["end"]   * TARGET_SR)

            chunk = audio[start:end]
            if len(chunk) < 100:       # skip degenerate cycles
                continue

            # Cyclic padding to exactly 8 s
            padded_wav = cyclic_padding(chunk, TARGET_SAMPLES)

            # ---- Run ASTFeatureExtractor ONCE per cycle ----
            # Returns dict with "input_values": shape (1, freq_bins, time_frames)
            feat = processor(
                padded_wav,
                sampling_rate=TARGET_SR,
                return_tensors="np",
            )
            # Squeeze the batch dimension → (freq_bins, time_frames)
            spectrogram = feat["input_values"].squeeze(0).astype(np.float32)

            # Class label
            c, w = int(ann["crackle"]), int(ann["wheeze"])
            if   c == 0 and w == 0: label = 0; stats["Normal"]  += 1
            elif c == 1 and w == 0: label = 1; stats["Crackle"] += 1
            elif c == 0 and w == 1: label = 2; stats["Wheeze"]  += 1
            else:                   label = 3; stats["Both"]     += 1

            if set_type == "train":
                X_train.append(spectrogram)
                y_train.append(label)
                device_train.append(dev_id)
            else:
                X_test.append(spectrogram)
                y_test.append(label)
                device_test.append(dev_id)

    # Convert to arrays
    X_train      = np.array(X_train,      dtype=np.float32)
    y_train      = np.array(y_train,      dtype=np.int64)
    device_train = np.array(device_train, dtype=np.int64)
    X_test       = np.array(X_test,       dtype=np.float32)
    y_test       = np.array(y_test,       dtype=np.int64)
    device_test  = np.array(device_test,  dtype=np.int64)

    print(f"\nDone.  Skipped {skipped} missing recordings.")
    print(f"   Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"   Class distribution: {stats}")
    print(f"   Spectrogram shape per sample: {X_train.shape[1:]}")

    np.savez_compressed(
        args.output,
        X_train=X_train, y_train=y_train, device_train=device_train,
        X_test=X_test,   y_test=y_test,   device_test=device_test,
    )
    print(f"\nSaved -> {args.output}")


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process ICBHI 2017 dataset into AST spectrograms"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ICBHI_final_database",
        help="Folder containing .wav and .txt annotation files",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="./data/ICBHI_challenge_train_test.txt",
        help="Official train/test split file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./icbhi_ast_16k_8s_spectrograms.npz",
        help="Output .npz file path",
    )
    args = parser.parse_args()
    process_data(args)
