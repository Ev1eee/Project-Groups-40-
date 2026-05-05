import csv
import random

import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_name(model_name, augment=False, seed=None):
    name = model_name
    if augment:
        name = f"{name}_aug"
    if seed is not None and seed != 42:
        name = f"{name}_seed{seed}"
    return name


def save_model(model, path, epoch, accuracy, model_name=None, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }
    if model_name is not None:
        checkpoint["model_name"] = model_name
    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)


def load_checkpoint(model, path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint


def save_rows_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_log(rows, path):
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    if rows and "lr" in rows[0]:
        fieldnames.append("lr")

    formatted_rows = []
    for row in rows:
        formatted_row = {
            "epoch": row["epoch"],
            "train_loss": f"{row['train_loss']:.6f}",
            "train_acc": f"{row['train_acc']:.6f}",
            "val_loss": f"{row['val_loss']:.6f}",
            "val_acc": f"{row['val_acc']:.6f}",
        }
        if "lr" in row:
            formatted_row["lr"] = f"{row['lr']:.8f}"
        formatted_rows.append(formatted_row)

    save_rows_csv(formatted_rows, path, fieldnames)
