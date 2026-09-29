import torch

from eeg_emotion.models import SimpleEEGCNN


def test_cnn_output_shape():
    model = SimpleEEGCNN(eeg_channels=32, num_classes=2)
    batch = torch.randn(3, 32, 512)
    assert model(batch).shape == (3, 2)

