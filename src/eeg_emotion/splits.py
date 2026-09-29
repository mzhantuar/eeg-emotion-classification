"""Participant-level dataset splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def make_subject_split(
    subject_ids: list[int] | tuple[int, ...],
    train_size: int = 24,
    validation_size: int = 4,
    seed: int = 42,
) -> SubjectSplit:
    """Shuffle ids reproducibly and split subjects, never individual windows."""
    unique_ids = np.asarray(sorted(set(subject_ids)), dtype=np.int64)
    if len(unique_ids) != len(subject_ids):
        raise ValueError("subject_ids contains duplicates")
    if train_size <= 0 or validation_size <= 0:
        raise ValueError("train_size and validation_size must be positive")
    if train_size + validation_size >= len(unique_ids):
        raise ValueError("At least one test subject is required")

    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    train_end = train_size
    validation_end = train_size + validation_size
    return SubjectSplit(
        train=tuple(int(x) for x in shuffled[:train_end]),
        validation=tuple(int(x) for x in shuffled[train_end:validation_end]),
        test=tuple(int(x) for x in shuffled[validation_end:]),
    )

