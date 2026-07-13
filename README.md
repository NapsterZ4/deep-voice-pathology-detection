# Detección de la enfermedad de Parkinson a partir de la voz

Trabajo Fin de Máster — Máster en Inteligencia Artificial, Universidad Politécnica de Madrid.

Detección de la enfermedad de Parkinson (EP) a partir de la tarea diadococinética "pa-ta-ka" mediante **GDBFNet** (*Gated Dual-Branch Fusion Network*), una arquitectura de fusión de doble rama con compuerta, regularizada con **ICMM** (*Intra-Class Manifold Mixing*), validada sobre dos corpus independientes (PC-GITA y NeuroVoz).

## Reproducibilidad y fuente de cifras

> **`results/` y `09_resultados_unificado.ipynb` son la fuente única de cifras y figuras de la memoria.** Los notebooks `00`–`08` documentan el recorrido experimental, pero **no deben usarse para regenerar los números de la memoria**. El notebook `09` recalcula ambos corpus con un método coherente (GDBFNet + ICMM, umbral de Youden), **congela** las predicciones en `results/` y, en ejecuciones posteriores, dibuja las figuras desde esos resultados guardados sin reentrenar.

## Estructura del repositorio

    ├── notebooks/          # secuencia experimental (00–09)
    ├── src/                # código compartido
    ├── data/               # corpus (no versionado)
    ├── results/            # resultados congelados: .npz, .json y modelos .pt (fuente única)
    ├── images/             # figuras finales de la memoria (fondo blanco)
    └── DEPRECATED_*        # versiones previas, solo histórico

## Notebooks (secuencia experimental)

| Notebook | Descripción |
|---|---|
| `00_requisitos.ipynb` | Preparación del entorno y verificación de dependencias, rutas y acceso a los datos. Punto de partida. |
| `01_lineas_base_1.ipynb` | Líneas base en el dominio temporal: arquitecturas entrenadas desde cero sobre la forma de onda, como punto de referencia. Producen `baseline_results.json`. |
| `01_lineas_base_2.ipynb` | Segundo bloque de líneas base. <!-- CONFIRMAR: ¿son las de clasificación de series temporales tomadas de la literatura? Ajustar. --> |
| `02_autosupervisado_capas_wav2vec.ipynb` | Ajuste fino parcial de wav2vec 2.0 y análisis por capas; muestra que las capas iniciales de bajo nivel son las más discriminativas. |
| `03_rama_espectral.ipynb` | Rama espectral: espectrograma log-Mel procesado con ResNet18 congelada; evaluación de la representación espectral aislada. |
| `04_fusion_gdbfnet.ipynb` | Arquitectura de fusión GDBFNet (compuerta que combina rama temporal y espectral) y ablación de la fusión. |
| `05_aumento_icmm_1.ipynb` | Estrategias de aumento y regularización: cribado de ruido gaussiano, ruido sobre *embeddings* e ICMM. |
| `05_aumento_icmm_2.ipynb` | Continuación del estudio de ICMM (número de interpolaciones, comparación de codificadores auto-supervisados). |
| `06_neurovoz_gdbfnet.ipynb` | Validación externa en NeuroVoz con GDBFNet + ICMM, bajo el mismo protocolo que PC-GITA. |
| `07_cross_corpus.ipynb` | Generalización entre corpus: transferencia directa en ambos sentidos y entrenamiento combinado. Guarda `cross_corpus_metrics.json`. |
| `08_validacion_anidada.ipynb` | Validación cruzada anidada para acotar el optimismo del protocolo (no anidado vs anidado). |
| `09_resultados_unificado.ipynb` | **Notebook maestro.** Recalcula de forma coherente ambos corpus (GDBFNet + ICMM, umbral de Youden), congela los resultados en `results/` y genera todas las figuras finales. Fuente única de cifras y figuras. |
| `10_lineas_base_figuras_pc_gita.ipynb` | Entrenamiento de las 11 líneas base sobre PC-GITA y figuras del capítulo de resultados. Produce `baseline_results.json`. |
| `11_lineas_base_neurovoz.ipynb` | Entrenamiento de las 11 líneas base sobre NeuroVoz (mismo esquema 5-fold que PC-GITA) y figura de sensibilidad/especificidad. Produce `baseline_results_neurovoz.json` y `images/res_baselines_sens_spec_nv.png`. |

### Notebooks obsoletos

| Notebook | Motivo |
|---|---|
| `DEPRECATED_experiments_neurovoz_dual_branch_1.ipynb` | Versión previa de NeuroVoz (concatenación sin compuerta). Superada por `06`. |
| `DEPRECATED_neurovoz_dual_branch_resnet18.ipynb` | Versión previa de NeuroVoz. Superada por `06`. |

## Módulos de `src/`

| Módulo | Descripción |
|---|---|
| `__init__.py` | Marca `src` como paquete importable. |
| `config.py` | Parámetros globales: rutas, semilla, *sample rate* e hiperparámetros por defecto. |
| `preprocessing.py` | Carga y preprocesamiento de audio: remuestreo a 16 kHz, normalización de pico, recorte de silencios y ventaneo. |
| `datasets.py` | Datasets y particiones a nivel de sujeto (evita fuga de información entre entrenamiento y prueba). |
| `neurovoz_loader.py` | Carga y parseo del corpus NeuroVoz (metadatos, filtrado por tarea, etiquetas). |
| `embeddings.py` | Extracción de *embeddings* de modelos auto-supervisados (wav2vec 2.0), por capas. |
| `spectral.py` | Espectrograma log-Mel y extracción de características espectrales (ResNet18). |
| `models.py` | Arquitecturas, incluida `DualBranchFusionNet` (GDBFNet, fusión de doble rama con compuerta). |
| `models_marta.py` | <!-- CONFIRMAR: describir qué contiene. Si no se usa en la versión final, considerar mover a DEPRECATED. --> |
| `training.py` | Bucle de entrenamiento, *early stopping* y funciones de entrenamiento por fold. |
| `evaluation.py` | Cálculo de métricas (AUC y métricas de umbral con índice de Youden), *bootstrap* e intervalos de confianza. |
| `visualization.py` | Funciones de figuras (ROC, matriz de confusión, preprocesamiento, etc.). |

## Artefactos en `results/`

| Fichero | Contenido |
|---|---|
| `resultados_completos_{corpus}.npz` | Predicciones OOF congeladas (`y_true`, `y_prob`, `alphas`) y métricas por fold. Base determinista de las figuras. |
| `resultados_gdbfnet_{corpus}.json` | Métricas robustas legibles (con y sin ICMM, con IC 95 %), α, umbral e hiperparámetros. |
| `resultados_gdbfnet_todos.json` | Resumen conjunto de ambos corpus. |
| `modelo_gdbfnet_{corpus}.pt` | Modelo final entrenado sobre el 100 % del corpus (pesos + configuración). |
| `baseline_results.json` | Métricas de las líneas base (PC-GITA). |
| `baseline_results_neurovoz.json` | Métricas de las líneas base (NeuroVoz). |
| `cross_corpus_metrics.json` | Métricas de los experimentos de generalización entre corpus. |

## Uso

    # instalar dependencias (uv)
    uv sync

    # ejecutar el notebook maestro:
    jupyter lab notebooks/09_resultados_unificado.ipynb

La primera ejecución de `09` entrena y congela los resultados en `results/`; las siguientes reutilizan esos ficheros y regeneran las figuras sin reentrenar.