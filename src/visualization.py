import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def plot_before_after_preprocessing(
    idx: int,
    df: pl.DataFrame,
    waveforms_orig: list[np.ndarray],
    waveforms_proc: list[np.ndarray],
    target_sr: int = 16_000,
) -> None:
    wf_orig = waveforms_orig[idx]
    wf_proc = waveforms_proc[idx]
    t_orig = np.arange(len(wf_orig)) / target_sr
    t_proc = np.arange(len(wf_proc)) / target_sr

    label_name = df["label_name"][idx]
    filename = df["filename"][idx]
    color = "#2196F3" if label_name == "HD" else "#F44336"

    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

    # --------------------------------------------------------------------------
    # SENAL ORIGINAL
    # --------------------------------------------------------------------------
    axes[0].plot(t_orig, wf_orig, color=color, linewidth=0.3, alpha=0.7)
    axes[0].set_title(f"ORIGINAL — {filename}  |  "
                      f"Duración: {len(wf_orig)/target_sr:.3f}s  |  "
                      f"Pico: {np.max(np.abs(wf_orig)):.4f}", fontsize=11)
    axes[0].set_ylabel("Amplitud")
    axes[0].set_ylim([-1, 1])
    axes[0].grid(True, alpha=0.3)

    # --------------------------------------------------------------------------
    # SENAL PROCESADA
    # --------------------------------------------------------------------------
    axes[1].plot(t_proc, wf_proc, color=color, linewidth=0.3, alpha=0.9)
    axes[1].set_title(f"PROCESADA (trimmed + normalized)  |  "
                      f"Duración: {len(wf_proc)/target_sr:.3f}s  |  "
                      f"Pico: {np.max(np.abs(wf_proc)):.4f}", fontsize=11)
    axes[1].set_ylabel("Amplitud")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylim([-1, 1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_duration_distribution(
    durations: list[float],
    stats: dict[str, float],
) -> None:
    """Histograma de duraciones con líneas de referencia en P95 y máximo."""
    arr = np.array(durations)

    plt.figure(figsize=(12, 4))
    plt.hist(arr, bins=25, color="#607D8B", edgecolor="white", alpha=0.8)
    plt.axvline(stats["p95"], color="#F44336", linestyle="--", linewidth=2,
                label=f"P95: {stats['p95']:.2f}s")
    plt.axvline(stats["max"], color="#FF9800", linestyle="--", linewidth=2,
                label=f"Máx: {stats['max']:.2f}s")
    plt.xlabel("Duración (s)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Duraciones — Señales Procesadas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(eval_results: dict) -> None:
    """Curvas ROC comparativas de todas las arquitecturas evaluadas."""
    colors = {
        "CNN1D": "#2196F3", "CNN1D_Light": "#4CAF50",
        "SincNet": "#FF9800", "Wav2Vec2": "#F44336",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for name, m in eval_results.items():
        color = colors.get(name, "#607D8B")
        ax.plot(m["fpr"], m["tpr"], color=color, linewidth=2,
                label=f"{name} (AUC = {m['AUC']:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Aleatorio (AUC = 0.500)")
    ax.set_xlabel("1 - Especificidad (Tasa de Falsos Positivos)")
    ax.set_ylabel("Sensibilidad (Tasa de Verdaderos Positivos)")
    ax.set_title("Curvas ROC — Comparación de Arquitecturas", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(eval_results: dict) -> None:
    """Matrices de confusión lado a lado para todas las arquitecturas."""
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, eval_results.items()):
        cm = np.array([[m["TN"], m["FP"]],
                       [m["FN"], m["TP"]]])

        ax.imshow(cm, cmap="Blues", aspect="equal")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["HD", "PD"])
        ax.set_yticklabels(["HD", "PD"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{name}\nAcc: {m['Accuracy']:.3f}", fontweight="bold")

    plt.suptitle("Matrices de Confusión — Conjunto de Test",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_top_confusion_matrices(
    sorted_results: list[tuple[str, dict]],
    top_n: int = 6,
) -> None:
    """Matrices de confusión del mejor fold para los top N modelos."""
    top_models = sorted_results[:top_n]
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, (name, r) in enumerate(top_models):
        best_fold = max(r["fold_details"], key=lambda f: f["auc"])
        y_pred = (best_fold["y_prob"] >= best_fold["threshold"]).astype(int)
        cm = confusion_matrix(best_fold["y_true"], y_pred)

        ax = axes[idx]
        ax.imshow(cm, cmap="Blues", interpolation="nearest")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["HD", "PD"])
        ax.set_yticklabels(["HD", "PD"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{name}\nAUC={r['auc_mean']:.3f} | Acc={r['acc_mean']:.3f}",
                     fontweight="bold", fontsize=11)

    # Ocultar ejes sobrantes si top_n < n_rows * n_cols
    for idx in range(len(top_models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Matrices de Confusión — Mejor Fold (Umbral de Youden)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_kfold_roc_curves(
    sorted_results: list[tuple[str, dict]],
    n_folds: int = 5,
) -> None:
    """Curvas ROC promediadas sobre K-Folds con banda ±1 std."""
    color_map = {
        "CNN1D": "#2196F3", "CNN1D_Light": "#4CAF50", "SincNet": "#FF9800",
        "WaveNet": "#9C27B0", "Wav2Vec2": "#F44336", "TemporalTransformer": "#00BCD4",
        "HuBERT": "#795548", "BiLSTM-CNN": "#E91E63", "CNN-Attention": "#3F51B5",
        "ECAPA-TDNN": "#009688", "RES1D": "#FF5722",
    }

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for name, r in sorted_results:
        tprs = []
        for fold in r["fold_details"]:
            fpr, tpr, _ = roc_curve(fold["y_true"], fold["y_prob"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        color = color_map.get(name, "#607D8B")

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                label=f"{name} (AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f})")
        ax.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1),
                        color=color, alpha=0.1)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Aleatorio (AUC=0.500)")
    ax.set_xlabel("1 - Especificidad (Tasa de Falsos Positivos)", fontsize=12)
    ax.set_ylabel("Sensibilidad (Tasa de Verdaderos Positivos)", fontsize=12)
    ax.set_title(f"Curvas ROC — {n_folds}-Fold CV + Evaluación por Sujeto",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()
