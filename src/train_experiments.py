"""Train all configured model/data-setting combinations.

Run:
    python train_experiments.py

Edit config.py to change models, preprocessing, affine settings, epochs, or optimizer.
"""

import time

import torch
import torch.nn as nn

import config
from experiment_common import (
    HISTORY_FIELDS,
    TRAIN_SUMMARY_FIELDS,
    count_parameters,
    iter_experiments,
    run_dir,
    save_metric_rows,
    synchronize_if_needed,
)
from model import build_model
from train import make_data_loaders, make_optimizer, make_scheduler, train_one_epoch, validate
from utils import get_device, save_model, set_seed


def train_condition(model_name, data_setting, run, device):
    set_seed(config.SEED)
    preprocess_profile = data_setting["preprocess_profile"]
    affine_setting = data_setting["affine_setting"]
    affine_config = config.AFFINE_SETTINGS[affine_setting]
    output_dir = run_dir(run)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_data_loaders(
        seed=config.SEED,
        preprocess_profile=preprocess_profile,
        affine_setting=affine_setting,
    )

    model = build_model(model_name).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = make_optimizer(model, config.OPTIMIZER, config.LEARNING_RATE, config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, config.SCHEDULER, epochs=config.EPOCHS)

    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "training_log.csv"
    best_accuracy = 0.0
    best_loss = float("inf")
    best_epoch = 0
    history = []

    print(f"\nTraining {run}")
    print(f"  model: {model_name}")
    print(f"  preprocessing: {preprocess_profile}")
    print(f"  affine setting: {affine_setting}")
    print(f"  train samples: {len(train_loader.dataset)}")
    print(f"  validation samples: {len(val_loader.dataset)}")
    print(f"  device: {device}")

    synchronize_if_needed(device)
    start = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        history.append({
            "run_name": run,
            "model": model_name,
            "data_setting": data_setting["name"],
            "preprocess_profile": preprocess_profile,
            "affine_setting": affine_setting,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        })
        save_metric_rows(history, log_path, HISTORY_FIELDS)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_loss = val_loss
            best_epoch = epoch
            save_model(
                model,
                checkpoint_path,
                epoch,
                val_acc,
                model_name=model_name,
                metadata={
                    "run_name": run,
                    "data_setting": data_setting,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "optimizer": config.OPTIMIZER,
                    "scheduler": config.SCHEDULER,
                    "weight_decay": config.WEIGHT_DECAY,
                    "label_smoothing": config.LABEL_SMOOTHING,
                    "seed": config.SEED,
                    "val_ratio": config.VAL_RATIO,
                },
            )

        if scheduler is not None:
            scheduler.step()

        print(f"  epoch {epoch:02d}/{config.EPOCHS} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

    synchronize_if_needed(device)
    training_seconds = time.perf_counter() - start
    final = history[-1]
    original_train_images = int(len(train_loader.dataset) / (1 + affine_config["copies_per_original"]))

    summary = {
        "run_name": run,
        "model": model_name,
        "data_setting": data_setting["name"],
        "preprocess_profile": preprocess_profile,
        "affine_setting": affine_setting,
        "bootstrap_copies_per_original": affine_config["copies_per_original"],
        "parameters": count_parameters(model),
        "device": str(device),
        "epochs": config.EPOCHS,
        "best_epoch": best_epoch,
        "best_val_acc": best_accuracy,
        "best_val_loss": best_loss,
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "generalization_gap": final["train_acc"] - final["val_acc"],
        "train_samples": len(train_loader.dataset),
        "original_train_images": original_train_images,
        "val_samples": len(val_loader.dataset),
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / max(1, config.EPOCHS),
        "checkpoint": str(checkpoint_path),
        "training_log": str(log_path),
    }
    save_metric_rows([summary], output_dir / "training_summary.csv", TRAIN_SUMMARY_FIELDS)
    return summary, history


def main():
    set_seed(config.SEED)
    device = get_device()
    all_summaries = []
    all_history = []

    for model_name, data_setting, run in iter_experiments():
        summary, history = train_condition(model_name, data_setting, run, device)
        all_summaries.append(summary)
        all_history.extend(history)

    output_dir = config.REPORT_DIR / config.REPORT_NAME
    save_metric_rows(all_summaries, output_dir / "training_summary.csv", TRAIN_SUMMARY_FIELDS)
    save_metric_rows(all_history, output_dir / "training_history.csv", HISTORY_FIELDS)
    print(f"\nTraining summaries saved to {output_dir}")


if __name__ == "__main__":
    main()
