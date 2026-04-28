"""
ASTDataset
----------
Loads pre-computed log-mel spectrogram tensors from the .npz file produced
by preprocess.py (which now calls ASTFeatureExtractor once, offline).

Why pre-computed?
  The original implementation ran ASTFeatureExtractor inside __getitem__,
  meaning the CPU computed log-mel spectrograms for every sample on every
  epoch.  By pre-computing once we eliminate the biggest DataLoader bottleneck
  (the GPU was starved waiting for CPU-side feature extraction).

Augmentation (training only):
  • Random gain jitter   (±10% amplitude, p=0.5)
  • Additive Gaussian noise (σ=0.005 on the spectrogram, p=0.5)
  • Random frequency masking  – zeros out up to 20 frequency bins (p=0.4)
  • Random time masking       – zeros out up to 30 time frames  (p=0.4)

  SpecAugment-style masking is applied in spectrogram space (after the
  offline feature extraction), so it remains fast and does not require
  re-running the feature extractor.

Data format in the .npz:
  X_train / X_test  : float32, shape (N, freq_bins, time_frames)
                      i.e. the output of ASTFeatureExtractor already squeezed.
  y_train / y_test  : int64, class labels 0-3
  device_train / device_test : int64, recording device IDs (for analysis only)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ASTDataset(Dataset):
    def __init__(self, X, y, device_ids, train: bool = True):
        """
        Parameters
        ----------
        X          : np.ndarray, shape (N, freq_bins, time_frames), float32
                     Pre-computed log-mel spectrograms.
        y          : np.ndarray, shape (N,), int64  class labels
        device_ids : np.ndarray, shape (N,), int64  recording device IDs
        train      : bool  — whether to apply augmentation
        """
        self.X = X
        self.y = y
        self.device_ids = device_ids
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Load the pre-computed spectrogram (copy to avoid mutating the array)
        spec = self.X[idx]
        if isinstance(spec, torch.Tensor):
            spec = spec.clone()
        else:
            spec = spec.copy()   # shape: (freq_bins, time_frames)

        if self.train:
            # --- Gain jitter ---
            if np.random.random() < 0.5:
                spec = spec * np.random.uniform(0.9, 1.1)

            # --- Additive Gaussian noise on spectrogram ---
            if np.random.random() < 0.5:
                spec = spec + np.random.normal(0, 0.005, spec.shape).astype(np.float32)

            # --- Frequency masking (SpecAugment) ---
            if np.random.random() < 0.4:
                freq_bins = spec.shape[0]
                f = np.random.randint(0, min(20, freq_bins))
                f0 = np.random.randint(0, freq_bins - f + 1)
                spec[f0: f0 + f, :] = 0.0

            # --- Time masking (SpecAugment) ---
            if np.random.random() < 0.4:
                time_frames = spec.shape[1]
                t = np.random.randint(0, min(30, time_frames))
                t0 = np.random.randint(0, time_frames - t + 1)
                spec[:, t0: t0 + t] = 0.0

        # ASTModel expects shape (freq_bins, time_frames) as a float tensor
        if isinstance(spec, torch.Tensor):
            spec_tensor = spec.to(dtype=torch.float32)
        else:
            spec_tensor = torch.from_numpy(spec.astype(np.float32, copy=False))
        label_tensor = torch.as_tensor(self.y[idx], dtype=torch.long)
        return spec_tensor, label_tensor, self.device_ids[idx]
