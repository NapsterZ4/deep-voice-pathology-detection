import os
import polars as pl
import glob

CLASS_MAP = {"PD": 1, "HC": 0}


# ------------------------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------------------------
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
