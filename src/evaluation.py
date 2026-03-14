from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc
)

def evaluate_subject_level(model, dataset, device):
    """
    Evaluación a nivel de sujeto.

    Promedia las probabilidades de todas las ventanas de cada
    paciente antes de tomar la decisión de clasificación.

    Args:
        model:   modelo entrenado
        dataset: AugmentedWindowedDataset con subject_ids
        device:  'mps', 'cuda' o 'cpu'

    Returns:
        y_true_subj: np.array con label real por sujeto
        y_prob_subj: np.array con probabilidad promedio por sujeto
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    subject_ids = dataset.subject_ids

    # ── Agrupar por sujeto ───────────────────────────────
    unique_subjects = np.unique(subject_ids)
    y_true_subj = []
    y_prob_subj = []

    for subj in unique_subjects:
        mask = subject_ids == subj
        y_true_subj.append(all_labels[mask][0])
        y_prob_subj.append(all_probs[mask].mean())

    return np.array(y_true_subj), np.array(y_prob_subj)

def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_probs)

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensibilidad": recall_score(y_true, y_pred),
        "Especificidad": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }

    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    metrics["AUC"] = auc(fpr, tpr)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr

    return metrics
