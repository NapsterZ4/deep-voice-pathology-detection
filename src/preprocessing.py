import os
import polars as pl
import glob
import numpy as np
import soundfile as sf
import librosa

CLASS_MAP = {"PD": 1, "HC": 0}

def load_metadata(base_path: str = "data") -> pl.DataFrame:
    records = []
    for class_name, label in CLASS_MAP.items():
        pattern = os.path.join(base_path, class_name, "*.wav")
        for fp in sorted(glob.glob(pattern)):
            records.append({
                "filepath":   fp,
                "filename":   os.path.basename(fp),
                "label":      label,
                "label_name": class_name,
            })

    df = pl.DataFrame(records)
    return df


def load_waveforms(
    df: pl.DataFrame,
    target_sr: int = 16_000,
) -> tuple[list[np.ndarray], pl.DataFrame]:
    waveforms = []
    durations = []

    for filepath in df["filepath"].to_list():
        wf, sr_orig = sf.read(filepath)

        # ----------------------------------------------------------------------
        # CONVERTIR A MONO SI EL AUDIO ESTA EN STEREO
        # ----------------------------------------------------------------------
        if wf.ndim > 1:
            wf = wf.mean(axis=1)

        # ----------------------------------------------------------------------
        # REMUESTREAR A 16 kHz SI ES NECESARIO
        # ----------------------------------------------------------------------
        if sr_orig != target_sr:
            wf = librosa.resample(wf, orig_sr=sr_orig, target_sr=target_sr)

        waveforms.append(wf)
        durations.append(len(wf) / target_sr)

    df = df.with_columns(pl.Series("duration_s", durations))

    return waveforms, df


def preprocess_waveform(waveform, target_peak=0.95, top_db=20, eps=1e-8):
    # --------------------------------------------------------------------------
    # NORMALIZACION PEAK
    # --------------------------------------------------------------------------
    peak = np.max(np.abs(waveform))
    if peak > eps:
        wf_normalized = target_peak * (waveform / peak)
    else:
        wf_normalized = waveform

    # --------------------------------------------------------------------------
    # RECORTE DE SILENCIOS
    # --------------------------------------------------------------------------
    wf_trimmed, _ = librosa.effects.trim(wf_normalized, top_db=top_db)

    return wf_trimmed

def compute_duration_percentiles(
    durations: list[float],
) -> tuple[dict[str, float], dict[str, str]]:
    """Calcula percentiles clave de las duraciones y los imprime."""
    arr = np.array(durations)
    keys = {"p50": 50, "p75": 75, "p90": 90, "p95": 95}
    labels = {
        "p50": "Percentil 50 (mediana)", "p75": "Percentil 75",
        "p90": "Percentil 90", "p95": "Percentil 95", "max": "Máximo"
    }
    stats = {k: np.percentile(arr, v) for k, v in keys.items()}
    stats["max"] = np.max(arr)

    return stats, labels
