import csv
import math
import time
from pathlib import Path

import torch

import config
from utils import save_rows_csv


MODEL_DESCRIPTIONS = {
    "simple_cnn": "Baseline CNN with two convolution-pooling stages and a dense classifier. It tests whether shallow local features are sufficient for the binary digit task.",
    "improved_cnn": "Deeper CNN with Conv-BatchNorm-ReLU blocks, max-pooling, and dropout. It assumes extra capacity plus regularization can improve shape recognition without overfitting too strongly.",
    "strided_cnn": "CNN with stride-2 convolution for learned downsampling. It assumes the model can learn better spatial reduction than fixed max-pooling alone.",
    "deotte_cnn": "Compact MNIST-style CNN with repeated Conv-BatchNorm-ReLU blocks, learned downsampling, adaptive pooling, and dropout. It assumes several small convolution stages are useful for distorted handwritten-style symbols.",
}

PREPROCESS_DESCRIPTIONS = {
    "plain": "Resize to the configured image size, convert to tensor, and normalize. This keeps the original location and scale information mostly intact.",
    "centered": "Threshold the foreground, crop the non-background bounding box, pad to a square, resize, convert to tensor, and normalize. This treats translation and scale as nuisance variation.",
}

SCIENTIFIC_ASSUMPTIONS = [
    "The input images are treated as single-channel binary symbols: foreground shape carries the useful class information.",
    "The class label should not change under small translations, rotations, scale changes, or mild shear.",
    "Foreground centering is valid only if absolute position inside the 32x32 frame is not meaningful for the class label.",
    "Affine bootstrapping adds synthetic variants of existing images; it improves robustness but does not add independent new information.",
    "The affine ranges must remain bounded because aggressive transformations can create unrealistic or label-ambiguous digits.",
    "The stratified validation split is used as a proxy for hidden-test generalization, not as proof of universal model quality.",
    "Accuracy is the primary metric for single-label digit classification; macro F1 is reported to expose class-specific imbalance or failure.",
    "Training and inference timings are hardware-dependent and should only be compared between runs executed on the same machine.",
]

TRAIN_SUMMARY_FIELDS = [
    "run_name", "model", "data_setting", "preprocess_profile", "affine_setting",
    "bootstrap_copies_per_original", "parameters", "device", "epochs", "best_epoch",
    "best_val_acc", "best_val_loss", "final_train_acc", "final_val_acc",
    "generalization_gap", "train_samples", "original_train_images", "val_samples",
    "training_seconds", "seconds_per_epoch", "checkpoint", "training_log",
]

HISTORY_FIELDS = [
    "run_name", "model", "data_setting", "preprocess_profile", "affine_setting",
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
]

EVAL_SUMMARY_FIELDS = [
    "run_name", "model", "data_setting", "preprocess_profile", "affine_setting",
    "checkpoint", "checkpoint_epoch", "checkpoint_accuracy", "val_loss", "val_acc",
    "val_error_percent", "macro_f1", "misclassified_count", "val_samples",
    "inference_seconds", "inference_ms_per_image", "test_submission",
]

MISCLASSIFIED_FIELDS = [
    "run_name", "model", "data_setting", "preprocess_profile", "affine_setting",
    "image_file", "true_label", "predicted_label", "predicted_confidence",
]


def report_dir():
    return config.REPORT_DIR / config.REPORT_NAME


def run_name(model_name, data_setting_name):
    name = f"{model_name}_{data_setting_name}"
    if config.SEED != 42:
        name = f"{name}_seed{config.SEED}"
    return name


def iter_experiments():
    for model_name in config.REPORT_MODELS:
        for data_setting in config.REPORT_DATA_SETTINGS:
            yield model_name, data_setting, run_name(model_name, data_setting["name"])


def run_dir(run):
    return config.RUN_DIR / run


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def synchronize_if_needed(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def measure_inference_time(model, loader, device, repeats):
    model.eval()
    total_images = 0
    synchronize_if_needed(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            total_images = 0
            for images, _labels in loader:
                images = images.to(device)
                _ = model(images)
                total_images += images.size(0)
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    seconds_per_pass = elapsed / max(1, repeats)
    ms_per_image = seconds_per_pass * 1000.0 / max(1, total_images)
    return seconds_per_pass, ms_per_image


def confusion_to_class_metrics(confusion):
    rows = []
    for label in range(config.NUM_CLASSES):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item()) - tp
        fn = int(confusion[label, :].sum().item()) - tp
        support = int(confusion[label, :].sum().item())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    macro_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0.0
    return rows, macro_f1


def format_value(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return value


def save_metric_rows(rows, path, fieldnames):
    formatted = [{field: format_value(row.get(field, "")) for field in fieldnames} for row in rows]
    save_rows_csv(formatted, path, fieldnames)


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def require_file(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\n{hint}")
    return path
