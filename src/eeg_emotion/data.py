"""Loading and validation helpers for preprocessed DEAP Python files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DEAPSubject:
    """One participant from the preprocessed DEAP dataset."""

    subject_id: int
    eeg: np.ndarray
    labels: np.ndarray


def subject_path(data_dir: str | Path, subject_id: int) -> Path:
    """Return the conventional DEAP path for a one-based participant id."""
    if not 1 <= subject_id <= 32:
        raise ValueError("DEAP subject_id must be between 1 and 32")
    return Path(data_dir) / f"s{subject_id:02d}.dat"


def load_deap_subject(data_dir: str | Path, subject_id: int) -> DEAPSubject:
    """Load one preprocessed DEAP participant and keep only 32 EEG channels."""
    path = subject_path(data_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(
            f"DEAP file not found: {path}. Put s01.dat ... s32.dat in {Path(data_dir)}."
        )

    with path.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")

    if not isinstance(payload, dict) or "data" not in payload or "labels" not in payload:
        raise ValueError(f"{path} does not contain DEAP 'data' and 'labels' arrays")

    data = np.asarray(payload["data"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.float32)

    if data.ndim != 3 or data.shape[1] < 32:
        raise ValueError(f"Unexpected DEAP data shape {data.shape} in {path}")
    if labels.ndim != 2 or labels.shape[0] != data.shape[0] or labels.shape[1] < 4:
        raise ValueError(f"Unexpected DEAP labels shape {labels.shape} in {path}")

    return DEAPSubject(subject_id=subject_id, eeg=data[:, :32, :], labels=labels[:, :4])


def available_subject_ids(data_dir: str | Path) -> list[int]:
    """Return sorted participant ids detected in a DEAP directory."""
    ids: list[int] = []
    for path in Path(data_dir).glob("s??.dat"):
        suffix = path.stem.removeprefix("s")
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)

