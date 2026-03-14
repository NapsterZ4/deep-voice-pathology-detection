import torch
from torch.utils.data import Dataset

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
