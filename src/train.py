import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset import DigitTrainDataset, get_transform
from model import SimpleCNN
from utils import save_log, save_model, set_seed


def make_data_loaders():
    dataset = DigitTrainDataset(transform=get_transform())
    val_size = int(len(dataset) * config.VAL_RATIO)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config.SEED)
    train_data, val_data = random_split(
        dataset, [train_size, val_size], generator=generator
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


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    set_seed(config.SEED)
    train_loader, val_loader = make_data_loaders()

    model = SimpleCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_accuracy = 0.0
    best_epoch = 0
    best_path = config.EXPERIMENT_DIR / "best_simple_cnn.pt"
    log_path = config.EXPERIMENT_DIR / "simple_cnn_training_log.csv"
    log_rows = []

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"train samples: {n_train}")
    print(f"val samples: {n_val}")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )
        val_loss, val_acc = validate(model, val_loader, loss_fn)

        print(
            f"epoch {epoch}/{config.EPOCHS} | "
            f"train acc {train_acc:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        save_log(log_rows, log_path)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            save_model(model, best_path, epoch, val_acc)

    print(f"best val acc {best_accuracy:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
