"""
src — Módulos reutilizables para detección de Parkinson desde audio.

Módulos:
    preprocessing: normalización, recorte, padding, windowing
    datasets:      clases Dataset de PyTorch
    models:        arquitecturas de clasificación (from scratch + preentrenadas)
    training:      loops de entrenamiento, K-Fold
    evaluation:    métricas, evaluación por sujeto, umbral de Youden
    embeddings:    extracción de embeddings congelados

Uso típico en un notebook:
    from src.preprocessing import preprocess_waveform, segment_waveform, build_windowed_dataset
    from src.datasets import WindowedDataset
    from src.models import CNN1D, Wav2Vec2Classifier, HuBERTClassifier
    from src.training import train_and_evaluate, train_one_fold
    from src.evaluation import compute_metrics, evaluate_subject_level
    from src.embeddings import extract_embeddings
"""