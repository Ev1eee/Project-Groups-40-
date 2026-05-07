"""Create the scientific report from existing training/evaluation outputs.

Run order:
    python train_experiments.py
    python evaluate_experiments.py
    python report.py

This script does not train or evaluate models. It only reads saved CSV files,
creates plots/example grids, and writes report.md.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

import config
from dataset import DigitTrainDataset
from experiment_common import (
    MODEL_DESCRIPTIONS,
    PREPROCESS_DESCRIPTIONS,
    SCIENTIFIC_ASSUMPTIONS,
    read_csv_rows,
    report_dir,
    require_file,
    save_metric_rows,
    write_text,
)
from preprocess import make_preview_image
from utils import save_rows_csv


NUMERIC_FIELDS = {
    "bootstrap_copies_per_original", "parameters", "epochs", "best_epoch",
    "best_val_acc", "best_val_loss", "final_train_acc", "final_val_acc",
    "generalization_gap", "train_samples", "original_train_images", "val_samples",
    "training_seconds", "seconds_per_epoch", "checkpoint_epoch", "checkpoint_accuracy",
    "val_loss", "val_acc", "val_error_percent", "macro_f1", "misclassified_count",
    "inference_seconds", "inference_ms_per_image", "epoch", "train_loss", "train_acc",
    "lr", "predicted_confidence", "wrong_count", "max_confidence",
}

HARD_EXAMPLE_FIELDS = [
    "image_file", "true_label", "wrong_count", "predicted_labels", "conditions",
    "max_confidence", "preprocess_profile",
]


def coerce_numbers(rows):
    for row in rows:
        for field in NUMERIC_FIELDS:
            if field in row and row[field] != "":
                try:
                    if field in {"parameters", "epochs", "best_epoch", "checkpoint_epoch", "train_samples", "original_train_images", "val_samples", "misclassified_count", "wrong_count"}:
                        row[field] = int(float(row[field]))
                    else:
                        row[field] = float(row[field])
                except ValueError:
                    pass
    return rows


def condition_label(row):
    return f"{row['model']}\n{row['data_setting']}"


def load_inputs(output_dir):
    training_summary = coerce_numbers(read_csv_rows(require_file(output_dir / "training_summary.csv", "Run python train_experiments.py first.")))
    evaluation_summary = coerce_numbers(read_csv_rows(require_file(output_dir / "evaluation_summary.csv", "Run python evaluate_experiments.py after training.")))
    training_history = coerce_numbers(read_csv_rows(require_file(output_dir / "training_history.csv", "Run python train_experiments.py first.")))
    misclassified_path = output_dir / "misclassified_examples.csv"
    misclassified = coerce_numbers(read_csv_rows(misclassified_path)) if misclassified_path.exists() else []

    training_by_run = {row["run_name"]: row for row in training_summary}
    combined = []
    for eval_row in evaluation_summary:
        train_row = training_by_run.get(eval_row["run_name"], {})
        combined.append({**train_row, **eval_row})
    return combined, training_history, misclassified


def plot_bar(rows, output_path, key, ylabel, title):
    labels = [condition_label(row) for row in rows]
    values = [float(row[key]) for row in rows]
    x_positions = list(range(len(labels)))
    plt.figure(figsize=(max(11, len(labels) * 0.85), 5))
    bars = plt.bar(x_positions, values)
    plt.xticks(x_positions, labels, fontsize=7)
    plt.ylabel(ylabel)
    plt.title(title)
    if values:
        plt.ylim(0.0, max(values) * 1.20 if max(values) > 0 else 1.0)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_curves(history_rows, output_path, metric_a, metric_b, ylabel, title):
    grouped = {}
    for row in history_rows:
        grouped.setdefault(row["run_name"], []).append(row)
    plt.figure(figsize=(11, 5))
    for run, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row[metric_a] for row in rows], linestyle="--", linewidth=1, alpha=0.7, label=f"{run} train")
        plt.plot(epochs, [row[metric_b] for row in rows], linewidth=1, label=f"{run} val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_preprocessing_examples(base_dataset, output_path):
    selected = base_dataset.samples[:config.N_PREPROCESS_EXAMPLES]
    if not selected:
        return False
    columns = ["raw", "plain", "centered", "centered + mild translate"]
    plt.figure(figsize=(len(columns) * 2.4, len(selected) * 2.0))
    for row_index, (image_path, label) in enumerate(selected):
        from PIL import Image, ImageOps
        images = [
            ImageOps.autocontrast(Image.open(image_path).convert("L")),
            make_preview_image(image_path, "plain"),
            make_preview_image(image_path, "centered"),
            make_preview_image(image_path, "centered", "mild", seed=config.SEED + row_index),
        ]
        for col_index, image in enumerate(images):
            ax = plt.subplot(len(selected), len(columns), row_index * len(columns) + col_index + 1)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            if row_index == 0:
                ax.set_title(columns[col_index], fontsize=8)
            if col_index == 0:
                ax.set_ylabel(f"label {label}\n{Path(image_path).name}", fontsize=8)
    plt.suptitle("Preprocessing and translation examples", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()
    return True


def save_bootstrap_examples(base_dataset, output_path):
    selected = base_dataset.samples[:config.N_BOOTSTRAP_EXAMPLES]
    affine_names = [name for name in config.AFFINE_SETTINGS if name != "none"]
    if not selected or not affine_names:
        return False
    columns = ["centered original"] + [f"bootstrap: {name}" for name in affine_names]
    plt.figure(figsize=(len(columns) * 2.4, len(selected) * 2.0))
    for row_index, (image_path, label) in enumerate(selected):
        images = [make_preview_image(image_path, "centered")]
        for affine_name in affine_names:
            images.append(make_preview_image(image_path, "centered", affine_name, seed=config.SEED + row_index * 19))
        for col_index, image in enumerate(images):
            ax = plt.subplot(len(selected), len(columns), row_index * len(columns) + col_index + 1)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            if row_index == 0:
                ax.set_title(columns[col_index], fontsize=8)
            if col_index == 0:
                ax.set_ylabel(f"label {label}\n{Path(image_path).name}", fontsize=8)
    plt.suptitle("Bootstrapped affine training examples", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()
    return True


def save_image_grid(rows, output_path, title, max_images):
    selected = rows[:max_images]
    if not selected:
        return False
    n_cols = min(6, len(selected))
    n_rows = math.ceil(len(selected) / n_cols)
    plt.figure(figsize=(n_cols * 2.25, n_rows * 2.35))
    for i, row in enumerate(selected, start=1):
        image = make_preview_image(row["image_file"], row.get("preprocess_profile", "centered"))
        ax = plt.subplot(n_rows, n_cols, i)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        file_name = Path(row["image_file"]).name
        if "predicted_label" in row:
            label = f"true {row['true_label']} | pred {row['predicted_label']}\n{file_name}"
        else:
            label = f"true {row['true_label']} | wrong {row['wrong_count']}x\n{file_name}"
        ax.set_title(label, fontsize=8)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()
    return True


def summarize_hard_examples(misclassified_rows):
    grouped = {}
    for row in misclassified_rows:
        key = row["image_file"]
        item = grouped.setdefault(key, {
            "image_file": row["image_file"],
            "true_label": row["true_label"],
            "wrong_count": 0,
            "predicted_labels": [],
            "conditions": [],
            "max_confidence": 0.0,
            "preprocess_profile": row.get("preprocess_profile", "centered"),
        })
        item["wrong_count"] += 1
        item["predicted_labels"].append(str(row["predicted_label"]))
        item["conditions"].append(row["run_name"])
        item["max_confidence"] = max(item["max_confidence"], float(row["predicted_confidence"]))
    rows = []
    for item in grouped.values():
        rows.append({
            "image_file": item["image_file"],
            "true_label": item["true_label"],
            "wrong_count": item["wrong_count"],
            "predicted_labels": ";".join(item["predicted_labels"]),
            "conditions": ";".join(item["conditions"]),
            "max_confidence": item["max_confidence"],
            "preprocess_profile": item["preprocess_profile"],
        })
    rows.sort(key=lambda row: (row["wrong_count"], row["max_confidence"]), reverse=True)
    return rows


def markdown_table(rows, fields, max_rows=None):
    shown = rows[:max_rows] if max_rows else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(rows, history_rows, output_dir):
    best = min(rows, key=lambda row: row["val_error_percent"])
    lines = []
    lines.append("# Scientific Comparison of CNN Models and Preprocessing Strategies")
    lines.append("")
    lines.append("## Research question")
    lines.append("This experiment evaluates whether CNN architecture changes, foreground centering, and affine bootstrapping reduce validation error for binary digit classification.")
    lines.append("")
    lines.append("## Experimental protocol")
    lines.append("Training, evaluation, and reporting are separated into three scripts. `train_experiments.py` trains all configured conditions and saves checkpoints/training logs. `evaluate_experiments.py` loads those checkpoints and computes validation/inference statistics. `report.py` reads the saved outputs and generates this report. No command-line flags are required; all settings are in `config.py`.")
    lines.append("")
    lines.append("## Assumptions")
    for assumption in SCIENTIFIC_ASSUMPTIONS:
        lines.append(f"- {assumption}")
    lines.append("")
    lines.append("## Model assumptions")
    for model_name in config.REPORT_MODELS:
        lines.append(f"- **{model_name}**: {MODEL_DESCRIPTIONS.get(model_name, 'No description available.')}")
    lines.append("")
    lines.append("## Preprocessing and bootstrapping assumptions")
    for profile in config.PREPROCESS_PROFILE_CHOICES:
        lines.append(f"- **{profile}**: {PREPROCESS_DESCRIPTIONS.get(profile, '')}")
    for name, setting in config.AFFINE_SETTINGS.items():
        lines.append(
            f"- **affine `{name}`**: {setting['description']} "
            f"copies_per_original={setting['copies_per_original']}, degrees={setting['degrees']}, "
            f"translate={setting['translate']}, scale={setting['scale']}, shear={setting['shear']}."
        )
    lines.append("")
    lines.append("The strong affine condition remains in the default report grid. It is useful as a stress test, but it should be accepted only if validation error improves; otherwise it is probably creating unrealistic samples.")
    lines.append("")
    lines.append("## Configuration used")
    lines.append(f"- Seed: `{config.SEED}`")
    lines.append(f"- Epochs: `{config.EPOCHS}`")
    lines.append(f"- Batch size: `{config.BATCH_SIZE}`")
    lines.append(f"- Optimizer: `{config.OPTIMIZER}`")
    lines.append(f"- Learning rate: `{config.LEARNING_RATE}`")
    lines.append(f"- Weight decay: `{config.WEIGHT_DECAY}`")
    lines.append(f"- Scheduler: `{config.SCHEDULER}`")
    lines.append(f"- Label smoothing: `{config.LABEL_SMOOTHING}`")
    lines.append("")
    lines.append("## Image examples")
    lines.append("The preprocessing grid shows raw inputs, plain resizing, foreground centering, and a mild translated affine variant.")
    lines.append("")
    lines.append("![Preprocessing examples](preprocessing_examples.png)")
    lines.append("")
    lines.append("The bootstrap grid shows synthetic affine samples used during training. These samples are generated from existing labeled images, not from external data.")
    lines.append("")
    lines.append("![Bootstrap examples](bootstrap_examples.png)")
    lines.append("")
    lines.append("## Results summary")
    summary_fields = [
        "model", "data_setting", "val_acc", "val_error_percent", "macro_f1",
        "training_seconds", "inference_ms_per_image", "train_samples", "misclassified_count",
    ]
    lines.append(markdown_table(sorted(rows, key=lambda row: row["val_error_percent"]), summary_fields))
    lines.append("")
    lines.append("## Graphs")
    lines.append("![Validation error comparison](validation_error_comparison.png)")
    lines.append("")
    lines.append("![Training time comparison](training_time_comparison.png)")
    lines.append("")
    lines.append("![Inference time comparison](inference_time_comparison.png)")
    lines.append("")
    lines.append("![Accuracy curves](accuracy_curves.png)")
    lines.append("")
    lines.append("![Loss curves](loss_curves.png)")
    lines.append("")
    lines.append("## Best condition")
    lines.append(f"The best validation condition is **{best['run_name']}**, with validation accuracy `{best['val_acc']:.4f}` and validation error `{best['val_error_percent']:.4f}%`.")
    lines.append("")
    lines.append("## Misclassification analysis")
    lines.append("The first grid shows validation images misclassified by the best condition. The second grid shows hard examples that many conditions misclassified.")
    lines.append("")
    lines.append("![Best-condition misclassified examples](best_misclassified_examples.png)")
    lines.append("")
    lines.append("![Hard examples](hard_examples.png)")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- A single validation split can overestimate or underestimate true generalization. Repeating the experiment over multiple seeds would strengthen the conclusion.")
    lines.append("- Affine bootstrapping can help robustness, but strong settings can create samples that are no longer realistic digits.")
    lines.append("- Timing measurements are machine-specific. They are useful for comparing these local runs, not for universal speed claims.")
    lines.append("- The final model should be selected by validation performance and error inspection, not by architecture complexity alone.")
    lines.append("")
    lines.append("## Files generated")
    lines.append("- `training_summary.csv`: training statistics and checkpoint paths.")
    lines.append("- `training_history.csv`: per-epoch losses and accuracies.")
    lines.append("- `evaluation_summary.csv`: validation metrics, misclassification counts, and inference timing.")
    lines.append("- `misclassified_examples.csv`: validation mistakes with image paths and predicted labels.")
    lines.append("- `hard_examples.csv`: validation images missed by multiple conditions.")
    write_text(output_dir / "report.md", "\n".join(lines))


def main():
    output_dir = report_dir()
    rows, history_rows, misclassified_rows = load_inputs(output_dir)
    rows = sorted(rows, key=lambda row: (row["model"], row["data_setting"]))

    base_dataset = DigitTrainDataset(transform=None)
    save_preprocessing_examples(base_dataset, output_dir / "preprocessing_examples.png")
    save_bootstrap_examples(base_dataset, output_dir / "bootstrap_examples.png")

    plot_bar(rows, output_dir / "validation_error_comparison.png", "val_error_percent", "Validation error (%)", "Validation error by model and data setting")
    plot_bar(rows, output_dir / "training_time_comparison.png", "training_seconds", "Training time (seconds)", "Training time by model and data setting")
    plot_bar(rows, output_dir / "inference_time_comparison.png", "inference_ms_per_image", "Inference time (ms/image)", "Inference speed by model and data setting")
    plot_curves(history_rows, output_dir / "accuracy_curves.png", "train_acc", "val_acc", "Accuracy", "Training and validation accuracy")
    plot_curves(history_rows, output_dir / "loss_curves.png", "train_loss", "val_loss", "Loss", "Training and validation loss")

    save_metric_rows(rows, output_dir / "model_summary.csv", list(rows[0].keys()))

    best = min(rows, key=lambda row: row["val_error_percent"])
    best_mistakes = [row for row in misclassified_rows if row["run_name"] == best["run_name"]]
    save_image_grid(best_mistakes, output_dir / "best_misclassified_examples.png", "Misclassified examples from best condition", config.MAX_MISCLASSIFIED_IMAGES)

    hard_examples = summarize_hard_examples(misclassified_rows)
    save_metric_rows(hard_examples, output_dir / "hard_examples.csv", HARD_EXAMPLE_FIELDS)
    save_image_grid(hard_examples, output_dir / "hard_examples.png", "Hard examples across conditions", config.MAX_MISCLASSIFIED_IMAGES)

    build_report(rows, history_rows, output_dir)
    print(f"Report saved to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
