import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import config
from preprocess import get_eval_transform, get_train_transform, get_transform, process_image


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


class BootstrappedTrainDataset(Dataset):
    """Training subset with deterministic synthetic affine copies.

    Length = number of selected original images * (1 + copies_per_original).
    Copy 0 is the unmodified preprocessed image. Copies 1..N are synthetic affine
    variants generated from the bounds in config.AFFINE_SETTINGS.
    """

    def __init__(self, base_dataset, indices, preprocess_profile, affine_setting, seed):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.preprocess_profile = preprocess_profile
        self.affine_setting = affine_setting
        self.seed = seed
        self.copies_per_original = int(config.AFFINE_SETTINGS[affine_setting]["copies_per_original"])
        self.samples = [base_dataset.samples[index] for index in self.indices]

    def __len__(self):
        return len(self.indices) * (1 + self.copies_per_original)

    def __getitem__(self, index):
        original_position = index % len(self.indices)
        copy_number = index // len(self.indices)
        original_index = self.indices[original_position]
        image_path, label = self.base_dataset.samples[original_index]
        image = Image.open(image_path).convert("L")

        if copy_number == 0:
            affine_setting = "none"
            affine_seed = self.seed
        else:
            affine_setting = self.affine_setting
            affine_seed = self.seed + original_index * 1009 + copy_number * 9176

        image_tensor = process_image(
            image,
            preprocess_profile=self.preprocess_profile,
            affine_setting=affine_setting,
            seed=affine_seed,
        )
        return image_tensor, label


class ValidationSubset(Dataset):
    """Validation subset that keeps original image paths available for reports."""

    def __init__(self, base_dataset, indices, preprocess_profile):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.preprocess_profile = preprocess_profile
        self.samples = [base_dataset.samples[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image_path, label = self.base_dataset.samples[original_index]
        image = Image.open(image_path).convert("L")
        image_tensor = process_image(
            image,
            preprocess_profile=self.preprocess_profile,
            affine_setting="none",
            seed=0,
        )
        return image_tensor, label


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
        candidates = [image_path] if image_path.suffix else [image_path.with_suffix(".png")]

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
