"""
Referencias
-----------
- Wodzinski et al. (2019): ResNet + espectrograma, PC-GITA, >90% acc
- Hireš et al. (2022): Ensemble Xception, AUC=0.89 en vocal /a/
- Madusanka & Lee (2024): AST, 91.67% acc en PC-GITA
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tv_transforms
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
from .config import logger


# ============================================================
# 1. GENERACIÓN DE MEL-ESPECTROGRAMAS
# ============================================================

def create_mel_transform(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> tuple[T.MelSpectrogram, T.AmplitudeToDB]:
    """
    Crea las transformaciones para convertir audio a mel-espectrograma en dB.

    Parameters
    ----------
    sample_rate : int
        Frecuencia de muestreo del audio.
    n_fft : int
        Tamaño de la FFT. Controla la resolución frecuencial:
        Δf = sample_rate / n_fft ≈ 15.6 Hz para n_fft=1024 a 16kHz.
    hop_length : int
        Paso entre ventanas STFT. hop_length=512 da 50% overlap.
    n_mels : int
        Número de bandas mel. 128 es estándar para capturar
        la estructura armónica de la voz humana (F0 ≈ 80-300 Hz,
        formantes hasta ~4 kHz).

    Returns
    -------
    mel_transform : T.MelSpectrogram
    amplitude_to_db : T.AmplitudeToDB
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    return mel_transform, amplitude_to_db


def waveform_to_mel_spectrogram(
    waveform: np.ndarray,
    mel_transform: T.MelSpectrogram,
    amplitude_to_db: T.AmplitudeToDB,
) -> np.ndarray:
    """
    Convierte una forma de onda cruda en mel-espectrograma (dB).

    Parameters
    ----------
    waveform : np.ndarray, shape (L,)
        Señal de audio normalizada.

    Returns
    -------
    np.ndarray, shape (n_mels, T)
        Mel-espectrograma en escala logarítmica (dB).
        T = floor(L / hop_length) + 1.
    """
    x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # (1, L)
    mel = mel_transform(x)         # (1, n_mels, T)
    mel_db = amplitude_to_db(mel)  # (1, n_mels, T)
    return mel_db.squeeze(0).numpy()  # (n_mels, T)


def generate_mel_spectrograms(
    waveforms: list[np.ndarray],
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> list[np.ndarray]:
    """
    Genera mel-espectrogramas para una lista de formas de onda.

    Returns
    -------
    list[np.ndarray]
        Lista de mel-espectrogramas, cada uno (n_mels, T_i).
        T_i varía según la duración del audio original.
    """
    mel_transform, amplitude_to_db = create_mel_transform(
        sample_rate, n_fft, hop_length, n_mels,
    )

    spectrograms = []
    for wf in waveforms:
        mel = waveform_to_mel_spectrogram(wf, mel_transform, amplitude_to_db)
        spectrograms.append(mel)

    return spectrograms


# ============================================================
# 2. EXTRACCIÓN DE FEATURES CON ResNet18 (TRANSFER LEARNING)
# ============================================================

class ResNet18SpectralExtractor:
    """
    Extrae features de un mel-espectrograma usando ResNet18 preentrenado
    en ImageNet como backbone congelado.
    """

    def __init__(self, device: str = "mps"):
        self.device = device

        # Cargar ResNet18 con pesos ImageNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Quitar capa FC final → output es (batch, 512, 1, 1) tras avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        # Normalización estándar de ImageNet
        self.normalize = tv_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def extract(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Extrae vector de features de un mel-espectrograma.

        Parameters
        ----------
        mel_db : np.ndarray, shape (n_mels, T)

        Returns
        -------
        np.ndarray, shape (512,)
        """
        # 1. Normalizar a [0, 1]
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max - mel_min > 0:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_db)

        # 2. Resize a 224×224 (requerido por ImageNet models)
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        mel_resized = nn.functional.interpolate(
            mel_tensor.unsqueeze(0), size=(224, 224),
            mode="bilinear", align_corners=False,
        ).squeeze(0)  # (1, 224, 224)

        # 3. Replicar a 3 canales (RGB)
        mel_rgb = mel_resized.repeat(3, 1, 1)  # (3, 224, 224)

        # 4. Normalización ImageNet
        mel_rgb = self.normalize(mel_rgb)

        # 5. Forward pass (congelado)
        with torch.no_grad():
            mel_rgb = mel_rgb.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
            features = self.backbone(mel_rgb)  # (1, 512, 1, 1)
            features = features.flatten().cpu().numpy()  # (512,)

        return features


def extract_spectral_features(
    waveforms: list[np.ndarray],
    device: str = "mps",
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Pipeline completo: waveforms → mel-spectrograms → ResNet18 features.

    """
    logger.info("Generando mel-espectrogramas...")
    mel_specs = generate_mel_spectrograms(
        waveforms, sample_rate, n_fft, hop_length, n_mels,
    )
    logger.info(f"  {len(mel_specs)} espectrogramas generados")
    logger.info(f"  Shape ejemplo: {mel_specs[0].shape}")

    logger.info("Extrayendo features con ResNet18 (ImageNet)...")
    extractor = ResNet18SpectralExtractor(device=device)

    features = []
    for i, mel in enumerate(mel_specs):
        feat = extractor.extract(mel)
        features.append(feat)
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(mel_specs)}")

    features = np.array(features)
    logger.info(f"  Features espectrales: {features.shape}")

    return features


class MelSpectrogramDataset(Dataset):
    """
    Dataset que convierte mel-espectrogramas a tensores RGB de 224×224
    listos para ResNet18.
    """
    def __init__(self, mel_specs, labels, indices):
        self.mel_specs = [mel_specs[i] for i in indices]
        self.labels = [labels[i] for i in indices]

        # Normalización ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mel_specs[idx]

        # Normalizar a [0, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max - mel_min > 0:
            mel_norm = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel)

        # Tensor → resize → 3 canales → normalización ImageNet
        t = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        t = nn.functional.interpolate(
            t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)  # (1, 224, 224)
        t = t.repeat(3, 1, 1)  # (3, 224, 224)
        t = (t - self.mean) / self.std

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return t, y
