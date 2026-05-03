import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, path, epoch, accuracy):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }, path)


def save_log(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in rows:
            file.write(
                f"{row['epoch']},"
                f"{row['train_loss']:.6f},"
                f"{row['train_acc']:.6f},"
                f"{row['val_loss']:.6f},"
                f"{row['val_acc']:.6f}\n"
            )
