"""Evaluate all trained experiment checkpoints.

Run after training:
    python evaluate_experiments.py

This script does validation evaluation and inference timing. It does not train.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DigitTestDataset
from evaluate import evaluate, save_confusion_matrix
from experiment_common import (
    EVAL_SUMMARY_FIELDS,
    MISCLASSIFIED_FIELDS,
    confusion_to_class_metrics,
    iter_experiments,
    measure_inference_time,
    read_csv_rows,
    report_dir,
    require_file,
    run_dir,
    save_metric_rows,
)
from model import build_model
from predict import predict_with_models
from preprocess import get_eval_transform
from train import make_data_loaders
from utils import get_device, load_checkpoint, save_rows_csv, set_seed


def collect_misclassified_examples(model, val_dataset, device, run, model_name, data_setting):
    model.eval()
    rows = []
    with torch.no_grad():
        for index in range(len(val_dataset)):
            image_tensor, true_label = val_dataset[index]
            image_path, _ = val_dataset.samples[index]
            outputs = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu()
            predicted = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted].item())
            if predicted != int(true_label):
                rows.append({
                    "run_name": run,
                    "model": model_name,
                    "data_setting": data_setting["name"],
                    "preprocess_profile": data_setting["preprocess_profile"],
                    "affine_setting": data_setting["affine_setting"],
                    "image_file": str(image_path),
                    "true_label": int(true_label),
                    "predicted_label": predicted,
                    "predicted_confidence": confidence,
                })
    rows.sort(key=lambda row: row["predicted_confidence"], reverse=True)
    return rows


def evaluate_condition(model_name, data_setting, run, device):
    output_dir = run_dir(run)
    checkpoint_path = require_file(
        output_dir / "best.pt",
        "Run python train_experiments.py before python evaluate_experiments.py.",
    )

    _, val_loader = make_data_loaders(
        seed=config.SEED,
        preprocess_profile=data_setting["preprocess_profile"],
        affine_setting="none",
    )

    model = build_model(model_name).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    val_loss, val_acc, confusion = evaluate(model, val_loader, loss_fn, device)
    class_rows, macro_f1 = confusion_to_class_metrics(confusion)
    inference_seconds, inference_ms_per_image = measure_inference_time(
        model,
        val_loader,
        device,
        repeats=config.INFERENCE_TIMING_REPEATS,
    )
    misclassified_rows = collect_misclassified_examples(
        model,
        val_loader.dataset,
        device,
        run,
        model_name,
        data_setting,
    )

    if config.DETAILED_CLASS_FILES:
        save_confusion_matrix(confusion, output_dir / "confusion_matrix.csv")
        save_metric_rows(class_rows, output_dir / "class_metrics.csv", ["class", "support", "precision", "recall", "f1"])

    test_submission = ""
    if config.MAKE_TEST_SUBMISSIONS:
        submission_path = config.SUBMISSION_DIR / f"submission_{run}_report.csv"
        test_dataset = DigitTestDataset(transform=get_eval_transform(profile=data_setting["preprocess_profile"]))
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        rows = predict_with_models([model], test_loader, device, use_tta=config.TTA)
        save_rows_csv(rows, submission_path, ["Id", "Category"])
        test_submission = str(submission_path)

    summary = {
        "run_name": run,
        "model": model_name,
        "data_setting": data_setting["name"],
        "preprocess_profile": data_setting["preprocess_profile"],
        "affine_setting": data_setting["affine_setting"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "checkpoint_accuracy": checkpoint.get("accuracy", ""),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_error_percent": (1.0 - val_acc) * 100.0,
        "macro_f1": macro_f1,
        "misclassified_count": len(misclassified_rows),
        "val_samples": len(val_loader.dataset),
        "inference_seconds": inference_seconds,
        "inference_ms_per_image": inference_ms_per_image,
        "test_submission": test_submission,
    }

    save_metric_rows([summary], output_dir / "evaluation_summary.csv", EVAL_SUMMARY_FIELDS)
    save_metric_rows(misclassified_rows, output_dir / "misclassified_examples.csv", MISCLASSIFIED_FIELDS)
    return summary, misclassified_rows


def main():
    require_file(report_dir() / "training_summary.csv", "Run python train_experiments.py first.")
    set_seed(config.SEED)
    device = get_device()
    summaries = []
    misclassified = []

    for model_name, data_setting, run in iter_experiments():
        print(f"\nEvaluating {run}")
        summary, rows = evaluate_condition(model_name, data_setting, run, device)
        summaries.append(summary)
        misclassified.extend(rows)
        print(f"  val acc: {summary['val_acc']:.4f}")
        print(f"  inference ms/image: {summary['inference_ms_per_image']:.4f}")

    output_dir = report_dir()
    save_metric_rows(summaries, output_dir / "evaluation_summary.csv", EVAL_SUMMARY_FIELDS)
    save_metric_rows(misclassified, output_dir / "misclassified_examples.csv", MISCLASSIFIED_FIELDS)
    print(f"\nEvaluation summaries saved to {output_dir}")


if __name__ == "__main__":
    main()
