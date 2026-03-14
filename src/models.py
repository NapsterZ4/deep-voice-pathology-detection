import torch.nn as nn

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
