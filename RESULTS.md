## DBFNet Regularizado — Resultados con 80% CV / 20% Test

### Hiperparametros optimos (trial #29, score = 0.9057)

| Parametro       | Valor    |
|-----------------|----------|
| lr              | 9.95e-4  |
| weight_decay    | 6.81e-4  |
| dropout         | 0.25     |
| label_smoothing | 0.1      |
| batch_size      | 32       |
| dim_proj        | 128      |
| dim_hidden      | 32       |

### Cross-validation (80 sujetos, 5-fold)

| Fold | AUC   | Acc   | Sens  | Spec  |
|------|-------|-------|-------|-------|
| 1    | 0.953 | 0.938 | 0.875 | 1.000 |
| 2    | 0.891 | 0.875 | 0.875 | 0.875 |
| 3    | 0.969 | 0.938 | 1.000 | 0.875 |
| 4    | 0.906 | 0.875 | 0.750 | 1.000 |
| 5    | 0.812 | 0.750 | 0.500 | 1.000 |

| Metrica       | Media | +/- std |
|---------------|-------|---------|
| AUC           | 0.906 | 0.055   |
| Accuracy      | 0.875 | 0.068   |
| Sensibilidad  | 0.800 | 0.170   |
| Especificidad | 0.950 | 0.061   |

- **CV Pearson (AUC): 6.1%**

![](images/img.png)

### Resultados en test (20 sujetos, ensemble de 5 seeds)

| Metrica       | Valor     |
|---------------|-----------|
| AUC           | 0.800     |
| **Accuracy**  | **0.750** |
| Sensibilidad  | 0.600     |
| Especificidad | 0.900     |
| Umbral        | 0.910     |

## DBFNet Regularizado — Resultados con 100% CV

### Hiperparametros optimos (trial #42, score = 0.8670)

| Parametro       | Valor    |
|-----------------|----------|
| lr              | 1.28e-4  |
| weight_decay    | 1.58e-3  |
| dropout         | 0.45     |
| label_smoothing | 0.05     |
| batch_size      | 32       |
| dim_proj        | 128      |
| dim_hidden      | 128      |

### Cross-validation (100 sujetos, 5-fold)

| Fold | AUC   | Acc   | Sens  | Spec  |
|------|-------|-------|-------|-------|
| 1    | 0.940 | 0.900 | 0.900 | 0.900 |
| 2    | 0.810 | 0.850 | 0.700 | 1.000 |
| 3    | 0.790 | 0.800 | 0.700 | 0.900 |
| 4    | 0.980 | 0.950 | 1.000 | 0.900 |
| 5    | 0.990 | 0.950 | 0.900 | 1.000 |

| Metrica       | Media | +/- std |
|---------------|-------|---------|
| AUC           | 0.902 | 0.085   |
| Accuracy      | 0.890 | 0.058   |
| Sensibilidad  | 0.840 | 0.120   |
| Especificidad | 0.940 | 0.049   |

- **CV Pearson (AUC): 9.4%**

![](images/img_1.png)

## Conclusiones del experimento de Data Augmentation

#### Resumen de resultados (Test Holdout)

| Métrica | Baseline | Gaussian Noise (SNR 25 dB) | VTLP (α ∈ [0.95, 1.05]) |
|---------|----------|---------------------------|-------------------------|
| AUC | 0.880 | 0.810 | 0.800 |
| Accuracy | 0.750 | 0.700 | 0.600 |
| Sensibilidad | 0.800 | 0.800 | 0.600 |
| Especificidad | 1.000 | 1.000 | 1.000 |

#### H1 — Confirmada
El modelo baseline presenta buen rendimiento en CV interna pero muestra signos de sobreajuste al evaluarse en test independiente por sujeto, consistente con la escasez de datos.

#### H2 — No confirmada
Ninguna de las dos técnicas de data augmentation mejora el rendimiento del baseline en test independiente:

- **Ruido gaussiano (SNR 25 dB):** Los embeddings wav2vec2, preentrenados con miles de horas de audio, ya son robustos a variaciones de ruido ambiental. Añadir ruido al waveform genera embeddings casi idénticos a los originales → no aporta diversidad en el espacio de representación donde opera el clasificador.

- **VTLP (α ∈ [0.95, 1.05]):** Aunque genera embeddings espectralmente distintos (altera formantes), el rango conservador de perturbación necesario para preservar la coherencia fisiológica de la DDK limita la diversidad introducida. Un rango más agresivo arriesgaría deformar las propiedades motor-articulatorias que constituyen el biomarcador clínico.

#### Interpretación clínica

Este resultado es consistente con la advertencia de que **no todo augmentation útil en audio general es útil en DDK-PD** (H2 condicionada). En diadococinesia, el espacio de variación fisiológicamente plausible es estrecho: la tasa de repetición silábica, la regularidad temporal y las transiciones articulatorias son precisamente los rasgos que definen la tarea y diferencian PD de controles. Cualquier perturbación debe preservarlos, lo cual limita inherentemente la diversidad que puede aportar el augmentation.

Con embeddings congelados de modelos preentrenados, la representación ya captura variabilidad acústica de bajo nivel. El cuello de botella no es la cantidad de datos en el espacio de entrada, sino la cantidad de **sujetos independientes** en el espacio clínico.

**Resultados usando esta configuración con un sujeto de Augment con VTLP + ruido gaussiano:**
![](images/img_2.png)

**Resultados usando esta configuración con un sujeto de Augment con solamente ruido gaussiano:**
![](images/img_3.png)

**Resultados usando esta configuración con dos sujetos de Augment con solamente ruido gaussiano:**
![](images/img_4.png)

**Resultados usando esta configuración con un sujetos de Augment con solamente Manifold Mixup:**
![](images/img_5.png)

En la tendencia observamos, que agregar mas sujetos de augmentation, baja el rendimiento del modelo.

## Optimización de hipeparámetros con Manifold MixUp:
Mejor trial #72
- Score (AUC - 0.5*std): 0.9150
- AUC: 0.934 ± 0.039
- Parámetros:
  - lr: 0.0007839314160797212 
  - weight_decay: 0.0002939827898456091 
  - dropout: 0.30000000000000004 
  - label_smoothing: 0.0 
  - batch_size: 8 
  - dim_proj: 128 
  - dim_hidden: 64 
  - mixup_alpha: 0.8

Observamos una mejoría con MixUp usando optimización de hiperparámetros.
![img.png](img_6.png)

### Conclusiones del experimento de Data Augmentation
#### Técnicas evaluadas

Se probaron 4 estrategias de data augmentation, todas con validación subject-independent y augmentation exclusivamente en entrenamiento:

**1. Gaussian Noise en waveform (SNR 25/20 dB):** Sin mejora.

**2. VTLP en waveform (α ∈ [0.95, 1.05]):** Sin mejora

**3. Feature Noise Injection en embeddings (σ=0.26):** Sin mejora en test (AUC idéntico al baseline). Genera variantes demasiado similares en el espacio de representación.

**4. Manifold Mixup intra-clase en embeddings (α=1.0):** Única técnica con un buen impacto por ahora

#### Resultados con test holdout (TEST_SIZE=0.20, ensemble 5 seeds)

| Método | AUC | Accuracy | Sensibilidad | Especificidad |
|--------|-----|----------|-------------|---------------|
| Baseline | 0.850 | 0.750 | 0.800 | 1.000 |
| Gaussian Noise Opt | 0.750 | 0.550 | 0.700 | 1.000 |
| Feature Noise Injection | 0.860 | 0.700 | 0.800 | 0.800 |
| **Manifold Mixup Opt** | **0.850** | **0.700** | **0.800** | **1.000** |

#### Estabilidad con holdout (media ± std por seed)

| Métrica | Baseline | Manifold Mixup |
|---------|----------|----------------|
| AUC | 0.858±0.016 | 0.850±0.030 |
| Accuracy | 0.720±0.040 | 0.730±0.024 |
| Sensibilidad | 0.780±0.040 | **0.800±0.000** |
| Especificidad | 0.740±0.080 | **0.840±0.049** |

#### CV de Pearson con holdout (5 seeds)

| Métrica | Baseline | Manifold Mixup |
|---------|----------|----------------|
| AUC | 1.86% | 3.57% |
| Accuracy | 5.56% | **3.36%** |
| Sensibilidad | 5.13% | **0.00%** |
| Especificidad | 10.81% | **5.83%** |

#### Resultados CV puro (TEST_SIZE=0, 100% datos en CV)

| Método | AUC | Accuracy | Sensibilidad | Especificidad |
|--------|-----|----------|-------------|---------------|
| Baseline (NB15, sin aug) | 0.902±0.085 | 0.890±0.058 | 0.840±0.120 | 0.940±0.049 |
| **Mixup Opt (α=1.0)** | **0.904±0.096** | **0.830±0.103** | **0.760±0.102** | **0.960±0.049** |
