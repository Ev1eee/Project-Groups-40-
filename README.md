# IIVP 2026 Digit Recognition

This is our digit recognition project. The current version is the first simple baseline.

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
- `src/model.py`: simple CNN baseline
- `src/train.py`: train and validate the baseline
- `src/utils.py`: small helper functions
