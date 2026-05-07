import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BootstrappedTrainDataset, DigitTrainDataset, ValidationSubset


def make_split_indices(dataset, seed=config.SEED):
    labels_to_indices = {label: [] for label in range(config.NUM_CLASSES)}
    for index, (_, label) in enumerate(dataset.samples):
        labels_to_indices[label].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for label in range(config.NUM_CLASSES):
        indices = labels_to_indices[label]
        val_size = int(len(indices) * config.VAL_RATIO)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in permutation]
        val_indices.extend(shuffled[:val_size])
        train_indices.extend(shuffled[val_size:])

    return train_indices, val_indices


def make_data_loaders(seed=config.SEED, preprocess_profile=None, affine_setting="none"):
    if preprocess_profile is None:
        preprocess_profile = config.PREPROCESS_PROFILE

    base_dataset = DigitTrainDataset(transform=None)
    train_indices, val_indices = make_split_indices(base_dataset, seed=seed)

    train_data = BootstrappedTrainDataset(
        base_dataset=base_dataset,
        indices=train_indices,
        preprocess_profile=preprocess_profile,
        affine_setting=affine_setting,
        seed=seed,
    )
    val_data = ValidationSubset(
        base_dataset=base_dataset,
        indices=val_indices,
        preprocess_profile=preprocess_profile,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader


def make_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def make_scheduler(optimizer, scheduler_name, epochs=None):
    if scheduler_name == "none":
        return None
    if scheduler_name == "step_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LR_GAMMA)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs or config.EPOCHS))
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
