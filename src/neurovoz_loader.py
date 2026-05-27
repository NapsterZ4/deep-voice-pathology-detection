"""
Carga del corpus Neurovoz en el mismo formato que load_metadata para el
corpus original (devuelve un polars DataFrame con columnas: filepath,
filename, label, label_name).

Esto permite reutilizar todo el pipeline existente (load_waveforms,
preprocess_waveform, segment_waveform, etc.) sin cambios.

Implementacion 100% polars (sin pandas, sin pyarrow).
"""
import re
import polars as pl


_TASK_RE = re.compile(r"^[A-Z]+_(.+)_\d+\.wav$")


def _extract_task(filename: str):
    m = _TASK_RE.match(filename)
    return m.group(1) if m else None


def load_neurovoz_metadata(
    meta_hc_path: str,
    meta_pd_path: str,
    audio_dir: str,
    task: str = "PATAKA",
) -> pl.DataFrame:
    """
    Carga los CSV de metadatos de Neurovoz, filtra por tarea, y devuelve
    un polars DataFrame compatible con el pipeline original.

    Notas
    -----
    - Los CSV de HC y PD tienen schemas ligeramente distintos
      (HC trae "Fiber/VocalFolds", PD trae "Vocal folds analysis").
      Solo necesitamos "Audio" y "Group", asi que seleccionamos esas
      dos columnas antes de concatenar para evitar conflictos de schema.

    Parameters
    ----------
    meta_hc_path : ruta al CSV de controles sanos
    meta_pd_path : ruta al CSV de pacientes PD
    audio_dir    : carpeta donde estan los .wav
    task         : tarea a filtrar (PATAKA, A1, E1, etc.)

    Returns
    -------
    pl.DataFrame con columnas:
        filepath   : ruta absoluta al .wav
        filename   : nombre del archivo
        label      : 0=HC, 1=PD
        label_name : "HC" o "PD"
    """
    # Leer y seleccionar solo las columnas que necesitamos para evitar
    # conflictos de schema entre los dos archivos
    hc = pl.read_csv(meta_hc_path).select(["Audio", "Group"])
    pd_ = pl.read_csv(meta_pd_path).select(["Audio", "Group"])
    df = pl.concat([hc, pd_], how="vertical")

    # Extraer filename del path del CSV original
    df = df.with_columns([
        pl.col("Audio").str.split("/").list.last().alias("filename"),
    ])

    # Extraer task del filename (ej: "HC_PATAKA_0034.wav" -> "PATAKA")
    df = df.with_columns([
        pl.col("filename")
          .map_elements(_extract_task, return_dtype=pl.Utf8)
          .alias("task"),
    ])

    # Filtrar por tarea
    df = df.filter(pl.col("task") == task)

    # Construir columnas finales
    df = df.with_columns([
        (pl.col("Group") == "PD").cast(pl.Int64).alias("label"),
        pl.col("Group").alias("label_name"),
        (pl.lit(audio_dir.rstrip("/") + "/") + pl.col("filename")).alias("filepath"),
    ])

    return df.select(["filepath", "filename", "label", "label_name"])
