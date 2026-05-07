import torch
from tqdm import tqdm

import config
from utils import save_rows_csv


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for true_label, predicted_label in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[true_label, predicted_label] += 1

    return total_loss / total, correct / total, confusion


def save_confusion_matrix(confusion, path):
    fieldnames = ["true_label"] + [str(label) for label in range(config.NUM_CLASSES)]
    rows = []
    for true_label in range(config.NUM_CLASSES):
        row = {"true_label": true_label}
        for predicted_label in range(config.NUM_CLASSES):
            row[str(predicted_label)] = int(confusion[true_label, predicted_label])
        rows.append(row)
    save_rows_csv(rows, path, fieldnames)
