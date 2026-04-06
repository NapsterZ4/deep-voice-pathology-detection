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