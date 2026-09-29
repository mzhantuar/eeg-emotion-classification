"""Leakage-safe preprocessing primitives."""

from __future__ import annotations

import numpy as np


LABEL_COLUMNS = {
    "valence": 0,
    "arousal": 1,
    "dominance": 2,
    "liking": 3,
}


def binary_emotion_labels(
    labels: np.ndarray,
    target: str = "valence",
    threshold: float = 5.0,
) -> np.ndarray:
    """Convert DEAP ratings to 0/1 using `rating > threshold` for class 1."""
    if target not in LABEL_COLUMNS:
        choices = ", ".join(LABEL_COLUMNS)
        raise ValueError(f"Unknown target '{target}'. Choose one of: {choices}")
    ratings = np.asarray(labels)[:, LABEL_COLUMNS[target]]
    return (ratings > threshold).astype(np.int64)


def remove_baseline(eeg: np.ndarray, sampling_rate: int = 128, seconds: int = 3) -> np.ndarray:
    """Remove the leading DEAP baseline interval from the time axis."""
    samples = sampling_rate * seconds
    if eeg.shape[-1] <= samples:
        raise ValueError("Signal is shorter than the requested baseline interval")
    return eeg[..., samples:]


def zscore_per_channel(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each channel using only values inside the current window."""
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    return ((window - mean) / np.maximum(std, eps)).astype(np.float32)


def window_starts(signal_samples: int, window_samples: int, step_samples: int) -> list[int]:
    """Return valid inclusive starting indices for fixed-length windows."""
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_samples and step_samples must be positive")
    if signal_samples < window_samples:
        return []
    return list(range(0, signal_samples - window_samples + 1, step_samples))

