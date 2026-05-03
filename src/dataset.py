from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


class DigitTrainDataset(Dataset):
    def __init__(self, root=config.TRAIN_DIR, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for label in range(config.NUM_CLASSES):
            class_dir = self.root / str(label)
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
