"""
Arquitecturas de Time Series Classification (TSC) del paper de Marta Rey-Paredes:
Rey-Paredes, M., Perez, C. J., & Mateos-Caballero, A. (2025).
"Time Series Classification of Raw Voice Waveforms for Parkinson's Disease
Detection Using Generative Adversarial Network-Driven Data Augmentation".
IEEE Open Journal of the Computer Society, 6, 72-84.

Se implementan las dos arquitecturas que NO tienen equivalente directo en
src/models.py:
    - InceptionTime (Fawaz et al., DAMI 2020)
    - CDIL-CNN (Cheng et al., 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_conv1d(x, conv, kernel_size, dilation=1):
    """
    Aplica un Conv1d preservando la longitud de la entrada,
    incluso con kernels pares (donde padding=k//2 falla).
    Padding total = (kernel_size - 1) * dilation, repartido
    asimetricamente (mas a la izquierda si es impar).
    """
    total = (kernel_size - 1) * dilation
    left = total // 2
    right = total - left
    x = F.pad(x, (left, right))
    return conv(x)


# ============================================================================
# INCEPTION TIME (Fawaz et al., DAMI 2020)
# ============================================================================

class InceptionModule(nn.Module):
    def __init__(self, in_ch, bottleneck=32, n_filters=32, kernel_sizes=(10, 20, 40)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.use_bottleneck = (in_ch > 1)

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
            bn_in = bottleneck
        else:
            bn_in = in_ch

        # Sin padding interno: lo aplicamos via F.pad para kernels pares
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_in, n_filters, k, padding=0, bias=False)
            for k in kernel_sizes
        ])

        # Maxpool con stride=1 + padding=1 (preserva longitud con kernel=3)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, n_filters, 1, bias=False)

        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x):
        x_b = self.bottleneck(x) if self.use_bottleneck else x

        outs = [
            _same_pad_conv1d(x_b, conv, k)
            for conv, k in zip(self.convs, self.kernel_sizes)
        ]
        outs.append(self.conv_pool(self.maxpool(x)))

        # Defensa: alinear al minimo comun por si hay desfases de 1 timestep
        min_len = min(o.shape[-1] for o in outs)
        outs = [o[..., :min_len] for o in outs]

        return F.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_ch=1, n_blocks=6, n_filters=32, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(n_blocks):
            self.blocks.append(InceptionModule(ch, n_filters=n_filters))
            ch = n_filters * 4

        self.res_in = nn.Conv1d(in_ch, ch, 1)
        self.res_mid = nn.Conv1d(ch, ch, 1)
        self.bn_res_in = nn.BatchNorm1d(ch)
        self.bn_res_mid = nn.BatchNorm1d(ch)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch, 1),
        )

    def forward(self, x):
        x_in = x
        x_mid = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:
                x_mid = x
                res = self.bn_res_in(self.res_in(x_in))
                min_len = min(x.shape[-1], res.shape[-1])
                x = F.relu(x[..., :min_len] + res[..., :min_len])
            if i == 5:
                res = self.bn_res_mid(self.res_mid(x_mid))
                min_len = min(x.shape[-1], res.shape[-1])
                x = F.relu(x[..., :min_len] + res[..., :min_len])

        x = self.gap(x).squeeze(-1)
        logits = self.classifier(x)
        return logits.squeeze(-1)


# ============================================================================
# CDIL-CNN (Cheng et al., 2023)
# ============================================================================

class CircularDilatedConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        total = (self.kernel_size - 1) * self.dilation
        left = total // 2
        right = total - left
        x = F.pad(x, (left, right), mode="circular")
        return self.conv(x)


class CDILBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.conv = CircularDilatedConv1d(c_in, c_out, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(c_out)
        self.shortcut = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.shortcut(x))


class CDIL_CNN(nn.Module):
    def __init__(self, in_ch=1, hidden=32, n_layers=8, kernel_size=3, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for l in range(n_layers):
            self.blocks.append(
                CDILBlock(ch, hidden, kernel_size, dilation=2 ** l)
            )
            ch = hidden

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        logits = self.classifier(x)
        return logits.squeeze(-1)
