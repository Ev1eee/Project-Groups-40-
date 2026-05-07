"""Deterministic preprocessing and affine bootstrapping for binary images.

The report uses this file for two jobs:
1. deterministic preprocessing: raw resize or foreground-centering;
2. synthetic affine bootstrapping: create a configured number of extra training
   examples with bounded random translation, scale, rotation, and shear.

The code deliberately avoids many profiles. Scientific comparison is controlled
from config.py by combining a preprocessing profile with an affine setting.
"""

import random
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

import config


PREPROCESS_DESCRIPTIONS = {
    "plain": "Resize image to the CNN input size, convert to tensor, normalize to roughly [-1, 1]. No centering is applied.",
    "centered": "Threshold foreground, crop non-black pixels, pad to a square canvas, resize, then normalize. This assumes the class identity is carried by the white foreground shape rather than its absolute position in the frame.",
}


class Binarize:
    def __init__(self, threshold=config.FOREGROUND_THRESHOLD):
        self.threshold = threshold

    def __call__(self, image):
        image = image.convert("L")
        return image.point(lambda pixel: 255 if pixel > self.threshold else 0)


class CropForegroundAndPad:
    def __init__(self, margin=config.CENTER_MARGIN, fill=0):
        self.margin = margin
        self.fill = fill

    def __call__(self, image):
        image = image.convert("L")
        bbox = image.getbbox()
        if bbox is None:
            return image

        cropped = image.crop(bbox)
        width, height = cropped.size
        side = max(width, height) + 2 * self.margin
        canvas = Image.new("L", (side, side), color=self.fill)
        left = (side - width) // 2
        top = (side - height) // 2
        canvas.paste(cropped, (left, top))
        return canvas


def validate_preprocess_profile(profile):
    if profile not in config.PREPROCESS_PROFILE_CHOICES:
        raise ValueError(
            f"Unknown preprocessing profile '{profile}'. Use one of: "
            f"{', '.join(config.PREPROCESS_PROFILE_CHOICES)}"
        )


def validate_affine_setting(setting_name):
    if setting_name not in config.AFFINE_SETTINGS:
        raise ValueError(
            f"Unknown affine setting '{setting_name}'. Use one of: "
            f"{', '.join(config.AFFINE_SETTINGS)}"
        )


def deterministic_preprocess(image, profile="plain"):
    validate_preprocess_profile(profile)
    image = image.convert("L")

    if profile == "centered":
        image = Binarize()(image)
        image = CropForegroundAndPad()(image)

    return image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), resample=Image.BILINEAR)


def affine_params(setting_name, seed, image_size=config.IMAGE_SIZE):
    validate_affine_setting(setting_name)
    setting = config.AFFINE_SETTINGS[setting_name]
    rng = random.Random(seed)

    degrees = float(setting["degrees"])
    translate_x, translate_y = setting["translate"]
    scale_min, scale_max = setting["scale"]
    shear = float(setting["shear"])

    angle = rng.uniform(-degrees, degrees) if degrees else 0.0
    max_dx = int(round(float(translate_x) * image_size))
    max_dy = int(round(float(translate_y) * image_size))
    translate = (
        rng.randint(-max_dx, max_dx) if max_dx else 0,
        rng.randint(-max_dy, max_dy) if max_dy else 0,
    )
    scale = rng.uniform(float(scale_min), float(scale_max))
    shear_value = rng.uniform(-shear, shear) if shear else 0.0
    return angle, translate, scale, shear_value


def apply_affine(image, setting_name="none", seed=0):
    validate_affine_setting(setting_name)
    if setting_name == "none":
        return image

    angle, translate, scale, shear_value = affine_params(setting_name, seed)
    return F.affine(
        image,
        angle=angle,
        translate=list(translate),
        scale=scale,
        shear=[shear_value, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def to_model_tensor(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])
    return transform(image)


def process_image(image, preprocess_profile="plain", affine_setting="none", seed=0):
    image = deterministic_preprocess(image, preprocess_profile)
    image = apply_affine(image, affine_setting, seed)
    return to_model_tensor(image)


def make_preview_image(image_path, preprocess_profile="centered", affine_setting="none", seed=0):
    image = Image.open(Path(image_path)).convert("L")
    image = deterministic_preprocess(image, preprocess_profile)
    image = apply_affine(image, affine_setting, seed)
    return ImageOps.autocontrast(image)


def get_eval_transform(profile="plain"):
    def transform(image):
        return process_image(image, preprocess_profile=profile, affine_setting="none", seed=0)
    return transform


def get_train_transform(profile="plain", augment=None):
    # Backward-compatible adapter for train.py/evaluate.py/predict.py.
    affine_setting = "mild" if augment else "none"
    def transform(image):
        return process_image(image, preprocess_profile=profile, affine_setting=affine_setting, seed=random.randint(0, 10**9))
    return transform


def get_transform():
    return get_eval_transform("plain")
