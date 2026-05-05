import torch.nn as nn

import config


class SimpleCNN(nn.Module):
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
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    def __init__(self, dropout=config.DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(1, 32),
            self._conv_block(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            self._conv_block(32, 64),
            self._conv_block(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, config.NUM_CLASSES),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class StridedCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(1, 32, kernel_size=3),
            self._conv_block(32, 32, kernel_size=3),
            self._conv_block(32, 32, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            self._conv_block(32, 64, kernel_size=3),
            self._conv_block(64, 64, kernel_size=3),
            self._conv_block(64, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(dropout),
            self._conv_block(64, 128, kernel_size=4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 2 * 2, config.NUM_CLASSES),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name=config.MODEL_NAME):
    if model_name == "simple_cnn":
        return SimpleCNN()
    if model_name == "improved_cnn":
        return ImprovedCNN()
    if model_name == "strided_cnn":
        return StridedCNN()

    raise ValueError(
        f"Unknown model '{model_name}'. Use 'simple_cnn', 'improved_cnn', or "
        "'strided_cnn'."
    )
