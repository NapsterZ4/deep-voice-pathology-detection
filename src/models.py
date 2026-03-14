import torch.nn as nn
import torch
import numpy as np
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
import math

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # ----------------------------------------------
            # CAPTURA DE PATRONES LOCALES FINOS
            # ----------------------------------------------
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # ----------------------------------------------
            # PATRONES INTERMEDIOS
            # ----------------------------------------------
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # ----------------------------------------------
            # PATRONES ABSTRACTOS Y FINALES
            # ----------------------------------------------
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # ----------------------------------------------
            # PROMEDIO GLOBAL
            # ----------------------------------------------
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)     # (batch, 128, 1)
        x = x.squeeze(-1)        # (batch, 128)
        x = self.classifier(x)   # (batch, 1)
        return x.squeeze(-1)     # (batch,)


class Cnn1dLight(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(8),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)


class SincConv1d(nn.Module):
    def __init__(self, out_channels=80, kernel_size=251, sr=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sr = sr

        # ----------------------------------------------
        # FRECUENCIA MAXIMA REPRESENTABLE (Nyquist)
        # ----------------------------------------------
        f_nyquist = sr / 2.0

        # ----------------------------------------------
        # INICIALIZACION FILTROS DISTRIBUIDOS UNIFORMES EN ESCALA MEL
        # ----------------------------------------------
        low_hz = 30.0
        high_hz = f_nyquist - (low_hz + 50)

        # ----------------------------------------------
        # ESCALA MEL PARA DISTRIBUCION CONCEPTUAL DE FILTROS
        # ----------------------------------------------
        mel_low = 2595.0 * np.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * np.log10(1.0 + high_hz / 700.0)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # ----------------------------------------------
        # FRECUENCIAS DE CORTE BAJA Y ANCHO DE BANDA
        # ----------------------------------------------
        self.low_hz_ = nn.Parameter(torch.tensor(hz_points[:-1], dtype=torch.float32))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz_points), dtype=torch.float32))

        # ----------------------------------------------
        # VENTANA HAMMING (FIJA)
        # ----------------------------------------------
        n = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        self.register_buffer("n_", n)
        self.register_buffer("window_", torch.hamming_window(kernel_size))

    def sinc(self, x):
        """Función sinc: sin(pi*x) / (pi*x), con sinc(0) = 1"""
        x = x + 1e-6  # Evitar división por cero
        return torch.sin(x) / x

    def forward(self, x):
        # ----------------------------------------------
        # ASEGURAR FRECUENCIAS VALIDAS
        # ----------------------------------------------
        low = torch.abs(self.low_hz_) + 1.0
        high = torch.clamp(low + torch.abs(self.band_hz_) + 2.0, max=self.sr / 2.0)

        # ----------------------------------------------
        # NORMALIZAR FRECUENCIAS
        # ----------------------------------------------
        f1 = low / self.sr
        f2 = high / self.sr

        # ----------------------------------------------
        # CONSTRUIR FILTROS PASA BANDA: sinc(2*f2*n) - sinc(2*f1*n)
        # ----------------------------------------------
        f1 = f1.unsqueeze(1)  # (out_channels, 1)
        f2 = f2.unsqueeze(1)
        n = self.n_.unsqueeze(0)  # (1, kernel_size)

        bp = 2 * f2 * self.sinc(2 * np.pi * f2 * n) - \
             2 * f1 * self.sinc(2 * np.pi * f1 * n)

        # ----------------------------------------------
        # APLICAR VENTANA DE HAMMING
        # ----------------------------------------------
        bp = bp * self.window_.unsqueeze(0)

        # ----------------------------------------------
        # RESHAPE PARA Conv1d: (out_channels, 1, kernel_size)
        # ----------------------------------------------
        filters = bp.unsqueeze(1)

        return torch.nn.functional.conv1d(x, filters, padding=self.kernel_size // 2)

class SincNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # ----------------------------------------------
            # CAPA SINC: APRENDE BANDAS DE FRECUENCIAS
            # ----------------------------------------------
            SincConv1d(out_channels=80, kernel_size=251, sr=16000),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(8),

            nn.Conv1d(80, 60, kernel_size=5, padding=2),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(8),

            nn.Conv1d(60, 60, kernel_size=5, padding=2),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(60, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, unfreeze_last_n=2):
        super().__init__()

        # ----------------------------------------------
        # CARGAR MODELO PREENTRENADO
        # ----------------------------------------------
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # ----------------------------------------------
        # CONGELAR TODO
        # ----------------------------------------------
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # ----------------------------------------------
        # DESCONGELAR ULTIMAS N CAPAS DEL TRANSFORMER
        # ----------------------------------------------
        for layer in self.wav2vec.encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ----------------------------------------------
        # CLASIFICADOR
        # ----------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 1)
        )

    def forward(self, x):
        # x: (batch, 1, L) → wav2vec espera (batch, L)
        x = x.squeeze(1)

        # ----------------------------------------------
        # EXTRAER REPRESENTACIONES
        # ----------------------------------------------
        outputs = self.wav2vec(x)
        hidden = outputs.last_hidden_state  # (batch, seq_len, 768)

        # ----------------------------------------------
        # PROMEDIO TEMPORAL → VECTOR FIJO
        # ----------------------------------------------
        pooled = hidden.mean(dim=1)  # (batch, 768)

        out = self.classifier(pooled)
        return out.squeeze(-1)

class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # Mantener longitud

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1)  # 1x1 conv
        )

    def forward(self, x):
        return x + self.block(x)  # Conexión residual


class WaveNetLike(nn.Module):
    def __init__(self, channels=32, kernel_size=3, dropout=0.3):
        super().__init__()

        # ----------------------------------------------
        # CAPA DE ENTRADA: adapta 1 canal a 'channels'
        # ----------------------------------------------
        self.input_conv = nn.Conv1d(1, channels, kernel_size=1)

        # ----------------------------------------------
        # BLOQUES CON DILATACIONES CRECIENTES EXPONENCIALMENTE
        # ----------------------------------------------
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.Sequential(*[
            WaveNetBlock(channels, kernel_size, d, dropout)
            for d in dilations
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x.squeeze(-1)

# ==============================================================
# POSITIONAL ENCODING (Vaswani et al., 2017)
# ==============================================================
class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal.

    Genera un tensor de forma (1, max_len, d_model) con senos y cosenos
    a diferentes frecuencias. Se suma a los embeddings de entrada para
    inyectar información sobre la posición temporal de cada token.

    No tiene parámetros entrenables.
    """

    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------
        # CONSTRUIR TABLA DE POSICIONES
        # ------------------------------------------------
        pe = torch.zeros(max_len, d_model)                         # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # dimensiones pares
        pe[:, 1::2] = torch.cos(position * div_term)   # dimensiones impares

        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)                  # no es parámetro entrenable

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================
# TRANSFORMER TEMPORAL LIGHTWEIGHT
# ==============================================================
class TemporalTransformer(nn.Module):
    """
    Transformer diseñado para clasificación de audio crudo.

    Flujo:
        (batch, 1, 120000)
              │
              ▼
        ┌─────────────────────┐
        │  Conv1D Front-End   │  3 capas con stride → reduce a ~468 tokens
        │  (extracción local) │
        └─────────────────────┘
              │
              ▼
        ┌─────────────────────┐
        │ Positional Encoding │  Inyecta orden temporal (sinusoidal)
        └─────────────────────┘
              │
              ▼
        ┌─────────────────────┐
        │ Transformer Encoder │  2 capas, 4 cabezas, d_model=64
        │ (atención global)   │
        └─────────────────────┘
              │
              ▼
        ┌─────────────────────┐
        │ Global Avg Pooling  │  468 tokens → 1 vector de dim 64
        └─────────────────────┘
              │
              ▼
        ┌─────────────────────┐
        │    Clasificador     │  Linear(64 → 1)
        └─────────────────────┘
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()

        # ------------------------------------------------
        # FRONT-END CONVOLUCIONAL
        # Objetivo: comprimir la señal temporal preservando
        # patrones locales. Cada capa reduce la longitud
        # con stride, similar a como Wav2Vec2 usa su CNN
        # encoder, pero mucho más liviano.
        #
        # Reducciones:
        #   Capa 1: 120,000 / 4 = 30,000
        #   Capa 2:  30,000 / 4 =  7,500
        #   Capa 3:   7,500 / 4 =  1,875
        #   MaxPool:  1,875 / 4 =    468
        # ------------------------------------------------
        self.conv_frontend = nn.Sequential(
            # Capa 1: captura micro-patrones (kernel=7 ≈ 0.4ms a 16kHz)
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Capa 2: patrones de nivel medio
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Capa 3: proyección a d_model
            nn.Conv1d(64, d_model, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            # Reducción final para tener ~468 tokens
            nn.MaxPool1d(kernel_size=4)
        )

        # ------------------------------------------------
        # POSITIONAL ENCODING
        # ------------------------------------------------
        self.pos_encoder = PositionalEncoding(d_model, max_len=2000, dropout=dropout)

        # ------------------------------------------------
        # TRANSFORMER ENCODER
        # Usamos 2 capas con 4 cabezas.
        # dim_feedforward = 128 (2x d_model, conservador)
        # batch_first=True para mantener (batch, seq, feat)
        # ------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ------------------------------------------------
        # CLASIFICADOR
        # ------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (batch, 1, L) donde L = 120,000

        # --- Front-end convolucional ---
        x = self.conv_frontend(x)       # (batch, d_model, T') donde T' ≈ 468

        # --- Transponer para Transformer: (batch, T', d_model) ---
        x = x.transpose(1, 2)           # (batch, 468, 64)

        # --- Positional Encoding ---
        x = self.pos_encoder(x)          # (batch, 468, 64)

        # --- Transformer Encoder ---
        x = self.transformer(x)          # (batch, 468, 64)

        # --- Global Average Pooling temporal ---
        x = x.mean(dim=1)               # (batch, 64)

        # --- Clasificación ---
        x = self.classifier(x)          # (batch, 1)
        return x.squeeze(-1)            # (batch,)


class HuBERTClassifier(nn.Module):
    """
    Fine-tuning de HuBERT BASE para clasificación binaria.

    Arquitectura idéntica a Wav2Vec2Classifier para garantizar
    que la comparación entre ambos foundation models sea justa:
    misma estrategia de congelamiento, mismo clasificador,
    mismos hiperparámetros de entrenamiento.
    """

    def __init__(self, unfreeze_last_n=2):
        super().__init__()

        # ----------------------------------------------
        # CARGAR MODELO PREENTRENADO
        # HuBERT BASE: 12 capas Transformer, dim=768
        # Preentrenado en 960h de LibriSpeech
        # ----------------------------------------------
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        # ----------------------------------------------
        # CONGELAR TODO
        # ----------------------------------------------
        for param in self.hubert.parameters():
            param.requires_grad = False

        # ----------------------------------------------
        # DESCONGELAR ÚLTIMAS N CAPAS DEL TRANSFORMER
        # Misma estrategia que Wav2Vec2: últimas 2 capas
        # ----------------------------------------------
        for layer in self.hubert.encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ----------------------------------------------
        # CLASIFICADOR
        # Dropout + Linear, idéntico a Wav2Vec2Classifier
        # ----------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 1)
        )

    def forward(self, x):
        # x: (batch, 1, L) → HuBERT espera (batch, L)
        x = x.squeeze(1)

        # ----------------------------------------------
        # EXTRAER REPRESENTACIONES
        # ----------------------------------------------
        outputs = self.hubert(x)
        hidden = outputs.last_hidden_state  # (batch, seq_len, 768)

        # ----------------------------------------------
        # PROMEDIO TEMPORAL → VECTOR FIJO
        # ----------------------------------------------
        pooled = hidden.mean(dim=1)  # (batch, 768)

        out = self.classifier(pooled)
        return out.squeeze(-1)


class BiLSTM_CNN(nn.Module):
    """
    Arquitectura híbrida CNN-BiLSTM para clasificación binaria
    de formas de onda crudas.

    Pipeline:
        1. CNN 1D extrae características locales de la señal cruda.
        2. BiLSTM modela dependencias temporales bidireccionales.
        3. Clasificador lineal produce la predicción binaria.

    Entrada: (batch, 1, seq_len)
    Salida:  (batch,) → logits
    """

    def __init__(
        self,
        cnn_channels: list[int] = [32, 64, 128],
        cnn_kernels: list[int] = [7, 5, 3],
        pool_size: int = 4,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Etapa 1: CNN Feature Extractor ──────────────────────
        cnn_blocks = []
        in_ch = 1
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── Etapa 2: BiLSTM ────────────────────────────────────
        # La entrada al LSTM es (batch, T', cnn_channels[-1])
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],   # 128 features de la CNN
            hidden_size=lstm_hidden,        # 64 por dirección
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Etapa 3: Clasificador ──────────────────────────────
        # BiLSTM produce 2 * lstm_hidden (forward + backward)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, 1, seq_len)
        Returns:
            logits: Tensor de forma (batch,)
        """
        # ── CNN: extraer features locales ───────────────────────
        # x: (batch, 1, seq_len) → (batch, 128, T')
        features = self.cnn(x)

        # ── Permutar para LSTM: (batch, T', 128) ───────────────
        features = features.permute(0, 2, 1)

        # ── BiLSTM: modelar contexto secuencial ────────────────
        # lstm_out: (batch, T', 2*lstm_hidden)
        lstm_out, (h_n, _) = self.lstm(features)

        # h_n: (num_layers*2, batch, lstm_hidden)
        # Tomamos el último layer de ambas direcciones
        # h_n[-2] → última capa forward, h_n[-1] → última capa backward
        h_forward = h_n[-2]   # (batch, lstm_hidden)
        h_backward = h_n[-1]  # (batch, lstm_hidden)
        h_bi = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*lstm_hidden)

        # ── Clasificación ───────────────────────────────────────
        logits = self.classifier(h_bi)  # (batch, 1)
        return logits.squeeze(-1)       # (batch,)

class CNNAttention(nn.Module):
    """
    CNN + Multi-Head Self-Attention + Attention Pooling.

    Pipeline:
        1. CNN 1D extrae features locales.
        2. Self-Attention modela relaciones globales entre posiciones.
        3. Attention Pooling agrega la secuencia en un vector fijo.
        4. Clasificador lineal produce la predicción.

    Entrada: (batch, 1, seq_len)
    Salida:  (batch,) → logits
    """

    def __init__(
        self,
        cnn_channels=(32, 64, 128),
        cnn_kernels=(7, 5, 3),
        pool_size=4,
        n_heads=4,
        dropout=0.3,
    ):
        super().__init__()

        # ── CNN Feature Extractor ────────────────────────────
        cnn_blocks = []
        in_ch = 1
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        d_model = cnn_channels[-1]  # 128

        # ── Multi-Head Self-Attention ────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # ── Attention Pooling ────────────────────────────────
        # Query aprendible que agrega la secuencia
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))

        # ── Clasificador ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # ── CNN ──────────────────────────────────────────────
        features = self.cnn(x)                      # (B, 128, T')
        features = features.permute(0, 2, 1)        # (B, T', 128)

        # ── Self-Attention + Residual + LayerNorm ────────────
        attn_out, _ = self.attention(features, features, features)
        features = self.layer_norm(features + attn_out)  # (B, T', 128)

        # ── Attention Pooling ────────────────────────────────
        # Expandir query aprendible al batch
        query = self.pool_query.expand(x.size(0), -1, -1)  # (B, 1, 128)
        pooled, _ = self.attention(query, features, features)
        pooled = pooled.squeeze(1)                   # (B, 128)

        # ── Clasificación ────────────────────────────────────
        logits = self.classifier(pooled)
        return logits.squeeze(-1)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: recalibra canales por importancia.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.se(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class SEResBlock(nn.Module):
    """
    Bloque residual con SE-block para ECAPA-TDNN.
    Conv1d dilatada + BatchNorm + ReLU + SE + residual.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        return x + self.se(self.block(x))


class AttentiveStatPooling(nn.Module):
    """
    Attentive Statistical Pooling.
    Calcula media y desviación estándar ponderadas por atención.
    """
    def __init__(self, channels, attention_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T)
        alpha = torch.softmax(self.attention(x), dim=-1)  # (B, C, T)
        mu = (alpha * x).sum(dim=-1)                       # (B, C)
        sigma = torch.sqrt(
            (alpha * x ** 2).sum(dim=-1) - mu ** 2 + 1e-8
        )
        return torch.cat([mu, sigma], dim=1)               # (B, 2C)


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN simplificado para clasificación de voz patológica.

    Entrada: (batch, 1, seq_len)
    Salida:  (batch,) → logits
    """
    def __init__(self, channels=128, dropout=0.3):
        super().__init__()

        # ── Stem ─────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ── SE-Res Blocks con dilatación creciente ───────────
        self.block1 = SEResBlock(channels, kernel_size=3, dilation=1)
        self.block2 = SEResBlock(channels, kernel_size=3, dilation=2)
        self.block3 = SEResBlock(channels, kernel_size=3, dilation=3)

        # ── Fusión multi-escala ──────────────────────────────
        # Concatena salidas de los 3 bloques → 3*channels
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ── Attentive Statistical Pooling ────────────────────
        self.asp = AttentiveStatPooling(channels)

        # ── Clasificador ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),  # 256 → 128
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        # ── Stem ─────────────────────────────────────────────
        x = self.stem(x)            # (B, 128, T)

        # ── SE-Res Blocks ────────────────────────────────────
        out1 = self.block1(x)       # (B, 128, T)
        out2 = self.block2(out1)    # (B, 128, T)
        out3 = self.block3(out2)    # (B, 128, T)

        # ── Fusión multi-escala ──────────────────────────────
        multi = torch.cat([out1, out2, out3], dim=1)  # (B, 384, T)
        fused = self.fusion(multi)                     # (B, 128, T)

        # ── Pooling estadístico atentivo ─────────────────────
        stats = self.asp(fused)     # (B, 256)

        # ── Clasificación ────────────────────────────────────
        logits = self.classifier(stats)
        return logits.squeeze(-1)


class ResBlock1D(nn.Module):
    """
    Bloque residual 1D: dos convoluciones + skip connection.
    Si la dimensión cambia, se agrega proyección en el atajo.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        # Proyección en el atajo si cambian dimensiones
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class Res1D(nn.Module):
    """
    Red Residual 1D para clasificación binaria de audio crudo.

    Entrada: (batch, 1, seq_len)
    Salida:  (batch,) → logits
    """
    def __init__(self, dropout=0.3):
        super().__init__()

        # ── Stem ─────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # ── Bloques residuales ───────────────────────────────
        self.layer1 = nn.Sequential(
            ResBlock1D(64, 64),
            ResBlock1D(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(64, 128, downsample=True),
            ResBlock1D(128, 128),
        )

        # ── Pooling + Clasificador ───────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)           # (B, 64, T/4)
        x = self.layer1(x)         # (B, 64, T/4)
        x = self.layer2(x)         # (B, 128, T/8)
        x = self.pool(x)           # (B, 128, 1)
        x = x.squeeze(-1)          # (B, 128)
        logits = self.classifier(x)
        return logits.squeeze(-1)

class Wav2Vec2LargeClassifier(nn.Module):
    """
    Fine-tuning de Wav2Vec2-LARGE para clasificación binaria.

    Arquitectura:
        1. Feature Encoder (congelado): 7 bloques Conv1D → 512-dim
        2. Context Network: 24 capas Transformer, d=1024
           - Capas [0..21] congeladas
           - Capas [22..23] descongeladas (fine-tuning)
        3. Mean Pooling temporal: (batch, T, 1024) → (batch, 1024)
        4. Clasificador: Linear(1024, 1)

    Input:  (batch, 1, L) — forma de onda cruda, canal mono
    Output: (batch,) — logits para clasificación binaria
    """

    def __init__(self, unfreeze_last_n=2):
        super().__init__()

        # ── 1. Cargar backbone preentrenado ──────────────
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large"
        )

        # ── 2. Congelar TODO ─────────────────────────────
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # ── 3. Descongelar últimas N capas Transformer ───
        total_layers = len(self.wav2vec.encoder.layers)  # 24
        for layer in self.wav2vec.encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ── 4. Clasificador ──────────────────────────────
        # LARGE produce embeddings de 1024-dim (vs 768 en BASE)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        # x: (batch, 1, L) → wav2vec espera (batch, L)
        x = x.squeeze(1)

        # Extraer representaciones contextuales
        outputs = self.wav2vec(x)
        hidden = outputs.last_hidden_state  # (batch, T, 1024)

        # Mean pooling temporal → vector fijo
        pooled = hidden.mean(dim=1)  # (batch, 1024)

        # Clasificación
        out = self.classifier(pooled)
        return out.squeeze(-1)  # (batch,)

class WavLMClassifier(nn.Module):
    """
    Fine-tuning de WavLM BASE para clasificación binaria.

    Arquitectura idéntica a Wav2Vec2Classifier y HuBERTClassifier,
    permitiendo una comparación directa entre los tres backbones.

    Input:  (batch, 1, L) — forma de onda cruda
    Output: (batch,) — logits
    """

    def __init__(self, unfreeze_last_n=2):
        super().__init__()

        # ── 1. Backbone preentrenado ─────────────────────
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

        # ── 2. Congelar todo ─────────────────────────────
        for param in self.wavlm.parameters():
            param.requires_grad = False

        # ── 3. Descongelar últimas N capas ───────────────
        for layer in self.wavlm.encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ── 4. Clasificador ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 1)
        )

    def forward(self, x):
        x = x.squeeze(1)                        # (batch, L)
        outputs = self.wavlm(x)
        hidden = outputs.last_hidden_state       # (batch, T, 768)
        pooled = hidden.mean(dim=1)              # (batch, 768)
        out = self.classifier(pooled)
        return out.squeeze(-1)                   # (batch,)
