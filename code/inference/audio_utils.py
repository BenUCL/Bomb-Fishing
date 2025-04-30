# inference/audio_utilspy

import os
from typing import List, Tuple
import numpy as np
import librosa

from config import SAMPLE_RATE, WINDOW_LENGTH_SEC, STAGGER_SEC


def find_audio_files(directory: str) -> List[str]:
    """Return list of .wav/.WAV files in `directory`."""
    return [f for f in os.listdir(directory) if f.lower().endswith(".wav")]


def load_and_resample(
        filepath: str,
        target_sr: int = SAMPLE_RATE
    ) -> Tuple[np.ndarray, int]:
    """Load audio from `filepath` and resample to `target_sr`."""
    audio, sr = librosa.load(filepath, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def sliding_windows(
        audio: np.ndarray,
        sr: int = SAMPLE_RATE
    ) -> List[Tuple[np.ndarray, float]]:
    """
    Generate overlapping windows of `WINDOW_LENGTH_SEC` from `audio`.
    This is so if a bomb is cut in half we don't miss it.
    Returns list of (window, start_time_s).
    """
    win_len = int(WINDOW_LENGTH_SEC * sr)
    stagger = int(STAGGER_SEC * sr)
    total = len(audio)
    windows: List[Tuple[np.ndarray, float]] = []

    # Stream 1
    for i in range(0, total, win_len):
        if i + win_len <= total:
            windows.append((audio[i:i + win_len], i / sr))

    # Stream 2 (staggered)
    for i in range(stagger, total, win_len):
        if i + win_len <= total:
            windows.append((audio[i:i + win_len], i / sr))

    return windows


def compute_mfcc(
        window: np.ndarray,
        sr: int = SAMPLE_RATE,
        n_mfcc: int = 32
    ) -> np.ndarray:
    """Compute MFCCs for `window` and return array shaped (n_mfcc, T, 1)."""
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=n_mfcc)
    # add channel & batch dims
    return mfcc.astype(np.float32)[..., np.newaxis][np.newaxis, ...]
