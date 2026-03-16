from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from .evaluation import evaluate_subject_level, compute_metrics, roc_curve
from .datasets import WindowedDataset
from sklearn.metrics import auc
from .datasets import build_windowed_dataset
from sklearn.model_selection import StratifiedKFold
import polars as pl

def train_and_evaluate(
        model,
        train_loader,
        val_loader,
        device,
        epochs=50,
        lr=1e-3,
        patience=7,
        model_name="model"
):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----------------------------------------------
    # TENSORBOARD: crear escritor con carpeta por modelo
    # ----------------------------------------------
    writer = SummaryWriter(log_dir=f"runs/{model_name}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"{'Época':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train Acc':>12} {'Val Acc':>12}")
    print("-" * 58)

    for epoch in range(1, epochs + 1):
        # ----------------------------------------------
        # Train
        # ----------------------------------------------
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y)
            t_correct += ((torch.sigmoid(out) >= 0.5).float() == y).sum().item()
            t_total += len(y)

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * len(y)
                v_correct += ((torch.sigmoid(out) >= 0.5).float() == y).sum().item()
                v_total += len(y)

        train_loss = t_loss / t_total
        val_loss = v_loss / v_total
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ----------------------------------------------
        # TENSORBOARD: registrar métricas por época
        # ----------------------------------------------
        writer.add_scalars("Loss", {
            "train": train_loss,
            "val": val_loss
        }, epoch)

        writer.add_scalars("Accuracy", {
            "train": train_acc,
            "val": val_acc
        }, epoch)

        # Brecha de generalización (útil para detectar overfitting)
        writer.add_scalar("Gap/loss_gap", val_loss - train_loss, epoch)

        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>12.4f} {train_acc:>12.3f} {val_acc:>12.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping en época {epoch}")
                break

    # ----------------------------------------------
    # TENSORBOARD: cerrar escritor
    # ----------------------------------------------
    writer.close()

    model.load_state_dict(best_state)

    return model, history, best_val_loss

def train_one_fold(model_class, X_train_f, y_train_f, subj_train_f,
                   X_val_f, y_val_f, subj_val_f,
                   lr, batch_size, patience, epochs, device="mps"):
    """
    Entrena un modelo en un fold y retorna métricas a nivel de sujeto.
    """
    # Crear datasets (sin augmentation)
    train_ds = WindowedDataset(X_train_f, y_train_f, subj_train_f)
    val_ds = WindowedDataset(X_val_f, y_val_f, subj_val_f)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Entrenar (silencioso)
    model = model_class().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        v_loss = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * len(y)
                v_total += len(y)

        val_loss = v_loss / v_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)

    # Evaluar a nivel de sujeto
    y_true_s, y_prob_s = evaluate_subject_level(model, val_ds, device)

    # AUC
    try:
        fpr, tpr, thresholds = roc_curve(y_true_s, y_prob_s)
        auc_score = auc(fpr, tpr)

        # Umbral óptimo (Youden)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        metrics_opt = compute_metrics(y_true_s, y_prob_s, threshold=best_threshold)
    except:
        auc_score = 0.5
        metrics_opt = compute_metrics(y_true_s, y_prob_s)
        best_threshold = 0.5

    return {
        "auc": auc_score,
        "acc_default": compute_metrics(y_true_s, y_prob_s)["Accuracy"],
        "acc_youden": metrics_opt["Accuracy"],
        "sens_youden": metrics_opt["Sensibilidad"],
        "spec_youden": metrics_opt["Especificidad"],
        "f1_youden": metrics_opt["F1-Score"],
        "mcc_youden": metrics_opt["MCC"],
        "threshold": best_threshold,
        "model": model,
        "y_true": y_true_s,
        "y_prob": y_prob_s
    }


def run_kfold_search(
    df: pl.DataFrame,
    window_len: int,
    hop_len: int,
    waveforms_processed: list[np.ndarray],
    model_config: dict,
    hyperparam_grid: dict,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Ejecuta K-Fold estratificado con búsqueda de hiperparámetros para cada modelo.

    Para cada combinación (lr, batch_size) de cada modelo, entrena N folds,
    calcula AUC medio y selecciona la mejor combinación.

    Returns
    -------
    dict
        Resultados por modelo con métricas medias, std y detalle por fold.
    """
    all_labels = df["label"].to_numpy()
    all_indices = np.arange(len(waveforms_processed))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    final_results = {}

    for model_name in model_config:
        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        config = model_config[model_name]
        grid = hyperparam_grid[model_name]

        best_combo_auc = -1
        best_combo = None
        best_combo_folds = None

        for lr in grid["lr"]:
            for bs in grid["batch_size"]:
                fold_metrics = []

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
                    X_tr, y_tr, s_tr = build_windowed_dataset(
                        train_idx, all_labels[train_idx], waveforms_processed, window_len, hop_len
                    )
                    X_vl, y_vl, s_vl = build_windowed_dataset(
                        val_idx, all_labels[val_idx], waveforms_processed, window_len, hop_len
                    )

                    if len(X_tr) == 0 or len(X_vl) == 0:
                        continue

                    metrics = train_one_fold(
                        model_class=config["class"],
                        X_train_f=X_tr, y_train_f=y_tr, subj_train_f=s_tr,
                        X_val_f=X_vl, y_val_f=y_vl, subj_val_f=s_vl,
                        lr=lr, batch_size=bs,
                        patience=config["patience"],
                        epochs=config["epochs"],
                    )
                    fold_metrics.append(metrics)

                if len(fold_metrics) == 0:
                    continue

                mean_auc = np.mean([m["auc"] for m in fold_metrics])
                print(f"  lr={lr:.0e} bs={bs}  →  AUC={mean_auc:.3f}")

                if mean_auc > best_combo_auc:
                    best_combo_auc = mean_auc
                    best_combo = {"lr": lr, "batch_size": bs}
                    best_combo_folds = fold_metrics

        if best_combo_folds is None:
            print(f"  ⚠️ No se pudo entrenar {model_name}")
            continue

        aucs  = [m["auc"] for m in best_combo_folds]
        accs  = [m["acc_youden"] for m in best_combo_folds]
        sens  = [m["sens_youden"] for m in best_combo_folds]
        specs = [m["spec_youden"] for m in best_combo_folds]
        f1s   = [m["f1_youden"] for m in best_combo_folds]
        mccs  = [m["mcc_youden"] for m in best_combo_folds]

        final_results[model_name] = {
            "best_lr": best_combo["lr"],
            "best_bs": best_combo["batch_size"],
            "auc_mean": np.mean(aucs),  "auc_std": np.std(aucs),
            "acc_mean": np.mean(accs),  "acc_std": np.std(accs),
            "sens_mean": np.mean(sens), "sens_std": np.std(sens),
            "spec_mean": np.mean(specs),"spec_std": np.std(specs),
            "f1_mean": np.mean(f1s),    "f1_std": np.std(f1s),
            "mcc_mean": np.mean(mccs),  "mcc_std": np.std(mccs),
            "fold_details": best_combo_folds,
        }

        print(f"\n  ✅ Mejor: lr={best_combo['lr']:.0e} bs={best_combo['batch_size']}")
        print(f"     AUC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"     Acc:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    return final_results
