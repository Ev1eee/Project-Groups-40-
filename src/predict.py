import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional
from tqdm import tqdm

import config
from dataset import DigitTestDataset, get_eval_transform
from model import build_model
from utils import (
    experiment_name,
    get_device,
    load_checkpoint,
    save_rows_csv,
    set_seed,
)


def apply_tta(images, angle=0, translate=(0, 0), scale=1.0):
    transformed_images = []
    for image in images:
        transformed_images.append(
            transform_functional.affine(
                image,
                angle=angle,
                translate=list(translate),
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=-1.0,
            )
        )

    return torch.stack(transformed_images)


def tta_batches(images):
    variants = [
        images,
        apply_tta(images, angle=7),
        apply_tta(images, angle=-7),
        apply_tta(images, translate=(2, 0)),
        apply_tta(images, translate=(-2, 0)),
        apply_tta(images, translate=(0, 2)),
        apply_tta(images, translate=(0, -2)),
        apply_tta(images, scale=0.95),
        apply_tta(images, scale=1.05),
    ]
    return variants


def predict_with_models(models, loader, device, use_tta=False):
    for model in models:
        model.eval()

    rows = []

    with torch.no_grad():
        for images, image_ids in tqdm(loader, leave=False):
            image_variants = tta_batches(images) if use_tta else [images]
            logits_sum = None
            n_predictions = 0

            for model in models:
                for variant in image_variants:
                    outputs = model(variant.to(device))
                    logits_sum = outputs if logits_sum is None else logits_sum + outputs
                    n_predictions += 1

            averaged_logits = logits_sum / n_predictions
            predictions = averaged_logits.argmax(dim=1).cpu().tolist()

            for image_id, prediction in zip(image_ids, predictions):
                rows.append({
                    "Id": image_id,
                    "Category": int(prediction),
                })

    return rows


def predict(model, loader, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, image_ids in tqdm(loader, leave=False):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            for image_id, prediction in zip(image_ids, predictions):
                rows.append({
                    "Id": image_id,
                    "Category": int(prediction),
                })

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Create a digit challenge submission.")
    parser.add_argument(
        "--model",
        choices=config.MODEL_CHOICES,
        default=config.MODEL_NAME,
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use the augmented experiment name when choosing the default checkpoint.",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=config.PREPROCESS_PROFILE_CHOICES,
        default=config.PREPROCESS_PROFILE,
        help="Preprocessing profile used for test images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to experiments/best_<model>.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Submission CSV path. Defaults to submissions/submission.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Seed used in the default checkpoint filename.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=config.TTA,
        help="Average predictions over deterministic test-time augmentations.",
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint paths to average. Uses --model architecture for all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    run_name = experiment_name(args.model, args.augment, seed=args.seed)
    checkpoint_path = (
        config.EXPERIMENT_DIR / f"best_{run_name}.pt"
        if args.checkpoint is None
        else config.ROOT_DIR / args.checkpoint
    )
    default_output_name = "submission.csv"
    if args.ensemble_checkpoints is not None:
        default_output_name = "submission_ensemble_strided3.csv"
    elif args.tta:
        default_output_name = f"submission_{run_name}_tta.csv"

    output_path = (
        config.SUBMISSION_DIR / default_output_name
        if args.output is None
        else config.ROOT_DIR / args.output
    )

    dataset = DigitTestDataset(transform=get_eval_transform(profile=args.preprocess_profile))
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    device = get_device()
    checkpoint_paths = (
        [config.ROOT_DIR / path for path in args.ensemble_checkpoints]
        if args.ensemble_checkpoints is not None
        else [checkpoint_path]
    )
    models = []
    for path in checkpoint_paths:
        model = build_model(args.model).to(device)
        load_checkpoint(model, path, device)
        models.append(model)

    rows = predict_with_models(models, loader, device, use_tta=args.tta)
    save_rows_csv(rows, output_path, ["Id", "Category"])

    print(f"checkpoints: {[str(path) for path in checkpoint_paths]}")
    print(f"preprocess profile: {args.preprocess_profile}")
    print(f"tta: {args.tta}")
    print(f"test samples: {len(rows)}")
    print(f"submission saved to {output_path}")


if __name__ == "__main__":
    main()
