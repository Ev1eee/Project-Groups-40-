import torch.nn as nn

import config


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """Small baseline CNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """Deeper Conv-BN-ReLU CNN with dropout."""

    def __init__(self, dropout=config.DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class StridedCNN(nn.Module):
    """CNN using learned stride-2 convolution for downsampling."""

    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            conv_block(32, 32, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            conv_block(32, 64),
            conv_block(64, 64),
            conv_block(64, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 2 * 2, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeotteCNN(nn.Module):
    """Compact MNIST-style CNN: Conv-BN-ReLU blocks, learned downsampling, dropout."""

    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            conv_block(32, 32, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            conv_block(32, 64),
            conv_block(64, 64),
            conv_block(64, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "improved_cnn": ImprovedCNN,
    "strided_cnn": StridedCNN,
    "deotte_cnn": DeotteCNN,
}


def build_model(model_name=config.MODEL_NAME):
    try:
        return MODEL_REGISTRY[model_name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown model '{model_name}'. Use one of: {', '.join(MODEL_REGISTRY)}"
        ) from error
