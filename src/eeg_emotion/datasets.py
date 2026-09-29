"""PyTorch dataset definitions for DEAP windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import DEAPSubject
from .preprocessing import binary_emotion_labels, window_starts, zscore_per_channel


@dataclass(frozen=True)
class WindowReference:
    subject_index: int
    trial_id: int
    start: int


class DEAPWindowDataset(Dataset):
    """Create virtual windows without duplicating the underlying EEG arrays."""

    def __init__(
        self,
        subjects: list[DEAPSubject],
        target: str = "valence",
        threshold: float = 5.0,
        sampling_rate: int = 128,
        window_seconds: float = 4.0,
        overlap: float = 0.5,
        baseline_seconds: float = 3.0,
        normalize: bool = True,
    ) -> None:
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")

        self.subjects = subjects
        self.normalize = normalize
        self.window_samples = int(round(window_seconds * sampling_rate))
        self.step_samples = max(1, int(round(self.window_samples * (1 - overlap))))
        self.baseline_samples = int(round(baseline_seconds * sampling_rate))
        self.labels = [binary_emotion_labels(s.labels, target, threshold) for s in subjects]
        self.index: list[WindowReference] = []

        for subject_index, subject in enumerate(subjects):
            for trial_id in range(subject.eeg.shape[0]):
                for start in window_starts(
                    subject.eeg.shape[-1] - self.baseline_samples,
                    self.window_samples,
                    self.step_samples,
                ):
                    self.index.append(
                        WindowReference(subject_index, trial_id, start + self.baseline_samples)
                    )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        ref = self.index[item]
        subject = self.subjects[ref.subject_index]
        stop = ref.start + self.window_samples
        eeg = subject.eeg[ref.trial_id, :, ref.start:stop]
        if self.normalize:
            eeg = zscore_per_channel(eeg)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(eeg)).float(),
            "y": torch.tensor(self.labels[ref.subject_index][ref.trial_id], dtype=torch.long),
            "subject_id": torch.tensor(subject.subject_id, dtype=torch.long),
            "trial_id": torch.tensor(ref.trial_id, dtype=torch.long),
        }
