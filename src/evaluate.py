import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from model import build_model
from train import make_data_loaders
from utils import (
    experiment_name,
    get_device,
    load_checkpoint,
    save_rows_csv,
    set_seed,
)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.long,
    )

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for true_label, predicted_label in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                confusion[true_label, predicted_label] += 1

    return total_loss / total, correct / total, confusion


def save_confusion_matrix(confusion, path):
    fieldnames = ["true_label"] + [str(label) for label in range(config.NUM_CLASSES)]
    rows = []
    for true_label in range(config.NUM_CLASSES):
        row = {"true_label": true_label}
        for predicted_label in range(config.NUM_CLASSES):
            row[str(predicted_label)] = int(confusion[true_label, predicted_label])
        rows.append(row)

    save_rows_csv(rows, path, fieldnames)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved digit CNN.")
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "improved_cnn", "strided_cnn"],
        default=config.MODEL_NAME,
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use the augmented experiment name when choosing the default checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to experiments/best_<model>.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed used for the validation split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = experiment_name(args.model, args.augment, seed=args.seed)
    checkpoint_path = (
        config.EXPERIMENT_DIR / f"best_{run_name}.pt"
        if args.checkpoint is None
        else config.ROOT_DIR / args.checkpoint
    )
    metrics_path = config.EXPERIMENT_DIR / f"{run_name}_evaluation.csv"
    confusion_path = config.EXPERIMENT_DIR / f"{run_name}_confusion_matrix.csv"

    _, val_loader = make_data_loaders(augment=False, seed=args.seed)
    device = get_device()
    model = build_model(args.model).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    loss_fn = nn.CrossEntropyLoss()

    val_loss, val_acc, confusion = evaluate(model, val_loader, loss_fn, device)

    checkpoint_epoch = checkpoint.get("epoch", "")
    checkpoint_accuracy = checkpoint.get("accuracy", "")
    save_rows_csv(
        [{
            "model": args.model,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_accuracy": checkpoint_accuracy,
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "val_samples": len(val_loader.dataset),
        }],
        metrics_path,
        [
            "model",
            "checkpoint",
            "checkpoint_epoch",
            "checkpoint_accuracy",
            "val_loss",
            "val_acc",
            "val_samples",
        ],
    )
    save_confusion_matrix(confusion, confusion_path)

    print(f"checkpoint: {checkpoint_path}")
    print(f"validation loss: {val_loss:.6f}")
    print(f"validation accuracy: {val_acc:.6f}")
    print(f"metrics saved to {metrics_path}")
    print(f"confusion matrix saved to {confusion_path}")


if __name__ == "__main__":
    main()

