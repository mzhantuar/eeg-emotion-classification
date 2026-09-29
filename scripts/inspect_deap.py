"""Inspect one preprocessed DEAP participant from the command line."""

from __future__ import annotations

import argparse

import numpy as np

from eeg_emotion.data import available_subject_ids, load_deap_subject
from eeg_emotion.preprocessing import binary_emotion_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/deap")
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()

    available = available_subject_ids(args.data_dir)
    print(f"Found {len(available)} subject files: {available}")

    subject = load_deap_subject(args.data_dir, args.subject)
    valence = binary_emotion_labels(subject.labels, "valence")
    arousal = binary_emotion_labels(subject.labels, "arousal")

    print(f"Subject: s{subject.subject_id:02d}")
    print(f"EEG shape (trials, channels, samples): {subject.eeg.shape}")
    print(f"Labels shape: {subject.labels.shape}")
    print(f"EEG dtype: {subject.eeg.dtype}")
    print(f"EEG finite: {bool(np.isfinite(subject.eeg).all())}")
    print(f"Valence class counts [low, high]: {np.bincount(valence, minlength=2).tolist()}")
    print(f"Arousal class counts [low, high]: {np.bincount(arousal, minlength=2).tolist()}")


if __name__ == "__main__":
    main()

