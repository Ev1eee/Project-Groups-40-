import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_transform(augment=False):
    transform_list = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]

    if augment:
        transform_list.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                fill=0,
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transforms.Compose(transform_list)


def get_transform():
    return get_eval_transform()


class DigitTrainDataset(Dataset):
    def __init__(self, root=config.TRAIN_DIR, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Training image folder not found: {self.root}")

        for label in range(config.NUM_CLASSES):
            class_dir = self.root / str(label)
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No training images found in: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DigitTestDataset(Dataset):
    def __init__(self, csv_file=config.TEST_CSV, root=config.TEST_DIR, transform=None):
        self.csv_file = Path(csv_file)
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Test CSV not found: {self.csv_file}")
        if not self.root.exists():
            raise FileNotFoundError(f"Test image folder not found: {self.root}")

        with self.csv_file.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "Id" not in reader.fieldnames:
                raise ValueError(f"Test CSV must contain an Id column: {self.csv_file}")

            for row in reader:
                image_id = str(row["Id"])
                image_path = self._find_image_path(image_id)
                self.samples.append((image_path, image_id))

        if not self.samples:
            raise ValueError(f"No test rows found in: {self.csv_file}")

    def _find_image_path(self, image_id):
        image_path = self.root / image_id
        if image_path.suffix:
            candidates = [image_path]
        else:
            candidates = [image_path.with_suffix(".png")]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Missing test image for Id {image_id}. Expected: {candidates[0]}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, image_id = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id
