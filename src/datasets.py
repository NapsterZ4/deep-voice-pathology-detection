import torch
from torch.utils.data import Dataset
import numpy as np
from .preprocessing import segment_waveform

class ParkinsonWaveformDataset(Dataset):
    def __init__(self, waveforms, labels, indices):
        self.waveforms = [waveforms[i] for i in indices]
        self.labels = [labels[i] for i in indices]

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # ----------------------------------------------
        # FORMA DE ONDA: (L,) → (1, L) PARA EL CANAL DE AUDIO
        # ----------------------------------------------
        x = torch.tensor(self.waveforms[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def build_windowed_dataset(indices, labels, waveforms, window_len, hop_len):
    X_list, y_list, subj_list = [], [], []

    for idx, label in zip(indices, labels):
        windows = segment_waveform(
            waveforms[idx],
            window_len=window_len,
            hop_len=hop_len
        )
        for w in windows:
            X_list.append(w)
            y_list.append(label)
            subj_list.append(idx)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(subj_list, dtype=np.int64),
    )


class WindowedDataset(Dataset):
    """
    Dataset de ventanas de audio para PyTorch.
    Cada muestra es una ventana de longitud fija con su etiqueta.
    Opcionalmente almacena el ID del sujeto para evaluación posterior.
    """

    def __init__(self, X, y, subject_ids=None):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, W)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
