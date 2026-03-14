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
