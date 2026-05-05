# IIVP 2026 Digit Recognition

This project builds a convolutional neural network pipeline for the IIVP 2026 digit recognition challenge. It loads the provided grayscale digit images, trains and evaluates baseline and improved CNN models, records experiment results, and generates Kaggle-ready submission files with the required `Id,Category` format. The current workflow includes a simple CNN baseline, an improved CNN with augmentation, and a stronger strided CNN with learning-rate decay, optional test-time augmentation, and optional checkpoint ensembling.

## Setup

Install the packages first:

```bash
python -m pip install -r requirements.txt
```

The data folder should be in the project folder and named:

```text
iivp-2026-challenge/
```


## Files

- `src/config.py`: paths and training settings
- `src/dataset.py`: load training images
- `src/model.py`: simple CNN baseline and improved CNN
- `src/train.py`: train and validate the baseline
- `src/evaluate.py`: evaluate a saved checkpoint
- `src/predict.py`: create a submission CSV
- `src/utils.py`: small helper functions

## Run

Train the baseline:

```bash
python src/train.py
```

Check validation accuracy and save a confusion matrix:

```bash
python src/evaluate.py
```

Create the challenge submission file:

```bash
python src/predict.py
```

Try the improved model with augmentation:

```bash
python src/train.py --model improved_cnn --augment --epochs 10
python src/evaluate.py --model improved_cnn --augment
python src/predict.py --model improved_cnn --augment
```

Try the stronger strided CNN with learning-rate decay:

```bash
python src/train.py --model strided_cnn --augment --epochs 30 --scheduler step_decay
python src/evaluate.py --model strided_cnn --augment
python src/predict.py --model strided_cnn --augment --output submissions/submission_strided_cnn_aug.csv
```

Create a test-time augmentation submission:

```bash
python src/predict.py --model strided_cnn --augment --tta
```

Train three seeds for a small ensemble:

```bash
python src/train.py --model strided_cnn --augment --epochs 30 --scheduler step_decay --seed 42
python src/train.py --model strided_cnn --augment --epochs 30 --scheduler step_decay --seed 43
python src/train.py --model strided_cnn --augment --epochs 30 --scheduler step_decay --seed 44
python src/predict.py --model strided_cnn --ensemble-checkpoints experiments/best_strided_cnn_aug.pt experiments/best_strided_cnn_aug_seed43.pt experiments/best_strided_cnn_aug_seed44.pt
```
