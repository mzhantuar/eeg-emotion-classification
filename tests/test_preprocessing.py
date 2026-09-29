import numpy as np

from eeg_emotion.preprocessing import binary_emotion_labels, window_starts, zscore_per_channel


def test_binary_labels_use_strictly_greater_than_five():
    labels = np.array([[5.0, 8.0, 1.0, 1.0], [5.1, 2.0, 1.0, 1.0]])
    assert binary_emotion_labels(labels, "valence").tolist() == [0, 1]


def test_window_starts_include_last_exact_window():
    assert window_starts(1024, 512, 256) == [0, 256, 512]


def test_zscore_is_per_channel():
    window = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
    normalized = zscore_per_channel(window)
    np.testing.assert_allclose(normalized.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=-1), 1.0, atol=1e-6)

