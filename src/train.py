import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import DigitTrainDataset, get_eval_transform, get_train_transform
from model import build_model
from utils import experiment_name, get_device, save_log, save_model, set_seed


MODEL_CHOICES = ["simple_cnn", "improved_cnn", "strided_cnn"]
SCHEDULER_CHOICES = ["none", "step_decay"]


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
        shuffled_indices = [indices[i] for i in permutation]
        val_indices.extend(shuffled_indices[:val_size])
        train_indices.extend(shuffled_indices[val_size:])

    return train_indices, val_indices


def make_data_loaders(augment=None, seed=config.SEED):
    if augment is None:
        augment = config.AUGMENT

    base_dataset = DigitTrainDataset(transform=None)
    train_indices, val_indices = make_split_indices(base_dataset, seed=seed)

    train_dataset = DigitTrainDataset(transform=get_train_transform(augment=augment))
    val_dataset = DigitTrainDataset(transform=get_eval_transform())
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)

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


def make_scheduler(optimizer, scheduler_name):
    if scheduler_name == "none":
        return None
    if scheduler_name == "step_decay":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.LR_GAMMA,
        )

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
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
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
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train a digit recognition CNN.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=config.MODEL_NAME,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use mild affine augmentation for the training split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.WEIGHT_DECAY,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default=config.SCHEDULER,
        help="Learning-rate schedule.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for split and model initialization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    train_loader, val_loader = make_data_loaders(
        augment=args.augment,
        seed=args.seed,
    )

    device = get_device()
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.scheduler)

    best_accuracy = 0.0
    best_epoch = 0
    run_name = experiment_name(args.model, args.augment, seed=args.seed)
    best_path = config.EXPERIMENT_DIR / f"best_{run_name}.pt"
    log_path = config.EXPERIMENT_DIR / f"{run_name}_training_log.csv"
    log_rows = []

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"model: {args.model}")
    print(f"augmentation: {args.augment}")
    print(f"scheduler: {args.scheduler}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"train samples: {n_train}")
    print(f"val samples: {n_val}")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train acc {train_acc:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })
        save_log(log_rows, log_path)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            save_model(
                model,
                best_path,
                epoch,
                val_acc,
                model_name=args.model,
                metadata={
                    "augment": args.augment,
                    "learning_rate": args.lr,
                    "scheduler": args.scheduler,
                    "lr_gamma": config.LR_GAMMA,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "val_ratio": config.VAL_RATIO,
                },
            )

        if scheduler is not None:
            scheduler.step()

    print(f"best val acc {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"best model saved to {best_path}")


if __name__ == "__main__":
    main()
