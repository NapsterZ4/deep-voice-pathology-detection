import torch
import numpy as np
from transformers import Wav2Vec2Model
from .preprocessing import segment_waveform
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import itertools
import polars as pl

def extract_embeddings(df_metadata, waveforms, subject_indices, model_name, device="mps"):
    print(f"  Cargando {model_name}...")

    if "hubert" in model_name.lower():
        from transformers import HubertModel
        model = HubertModel.from_pretrained(model_name)
    elif "wavlm" in model_name.lower():
        from transformers import WavLMModel
        model = WavLMModel.from_pretrained(model_name)
    else:
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    all_labels = df_metadata["label"].to_numpy()
    embeddings = []
    labels_out = []

    with torch.no_grad():
        for idx in subject_indices:
            x = torch.tensor(waveforms[idx], dtype=torch.float32).unsqueeze(0).to(device)
            hidden = model(x).last_hidden_state
            pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(pooled)
            labels_out.append(all_labels[idx])

    return np.array(embeddings), np.array(labels_out)


def extract_windowed_embeddings(
        waveforms,
        subject_indices,
        labels,
        model_name,
        window_len,
        hop_len,
        device="mps"
):
    """
    Extrae embeddings de Wav2Vec2/HuBERT por VENTANA (no por señal completa).

    Pipeline por cada sujeto:
        1. Segmentar la señal en ventanas de longitud fija
        2. Cada ventana → modelo congelado → mean pooling → embedding (D,)
        3. Asociar cada embedding con su etiqueta y sujeto

    Returns:
        embeddings: np.array (N_ventanas, D)
        labels:     np.array (N_ventanas,)
        subjects:   np.array (N_ventanas,) — ID del sujeto original
    """
    print(f"  Cargando {model_name}...")
    model = Wav2Vec2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    emb_list, lab_list, subj_list = [], [], []
    skipped = 0

    with torch.no_grad():
        for idx in subject_indices:
            wf = waveforms[idx]
            windows = segment_waveform(wf, window_len, hop_len)

            if len(windows) == 0:
                skipped += 1
                continue

            for w in windows:
                x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
                outputs = model(x)
                hidden = outputs.last_hidden_state     # (1, T, D)
                pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)

                emb_list.append(pooled)
                lab_list.append(labels[idx])
                subj_list.append(idx)

    print(f"  Sujetos procesados: {len(subject_indices) - skipped}/{len(subject_indices)}")
    print(f"  Ventanas extraídas: {len(emb_list)}")

    return np.array(emb_list), np.array(lab_list), np.array(subj_list)

def run_embedding_svm_search(
    df: pl.DataFrame,
    waveforms_processed: list[np.ndarray],
    embedding_models: dict[str, str],
    pca_components: list[int],
    svm_c: list[float],
    svm_kernels: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    K-Fold con embeddings congelados + PCA + SVM.

    Extrae embeddings una sola vez por modelo, luego busca la mejor
    combinación (PCA, C, kernel) evaluando a nivel de sujeto.

    Returns
    -------
    dict
        Resultados por modelo con métricas medias, std y detalle por fold.
    """
    all_labels = df["label"].to_numpy()
    all_indices = np.arange(len(waveforms_processed))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {}

    for emb_name, hf_model_name in embedding_models.items():
        print(f"\n{'='*70}")
        print(f"  {emb_name} → Embeddings por ventana + PCA + SVM")
        print(f"{'='*70}")

        all_emb, all_emb_labels, all_emb_subjects = extract_windowed_embeddings(
            waveforms_processed, all_indices, all_labels, hf_model_name
        )
        print(f"  Shape total: {all_emb.shape}")

        best_combo_auc = -1
        best_combo = None
        best_combo_folds = None

        for n_comp, C, kernel in itertools.product(pca_components, svm_c, svm_kernels):
            fold_metrics = []

            for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(
                skf.split(all_indices, all_labels)
            ):
                train_subj_set = set(train_subj_idx)
                val_subj_set = set(val_subj_idx)

                train_mask = np.array([s in train_subj_set for s in all_emb_subjects])
                val_mask = np.array([s in val_subj_set for s in all_emb_subjects])

                X_train_emb = all_emb[train_mask]
                y_train_emb = all_emb_labels[train_mask]
                subj_train_emb = all_emb_subjects[train_mask]

                X_val_emb = all_emb[val_mask]
                y_val_emb = all_emb_labels[val_mask]
                subj_val_emb = all_emb_subjects[val_mask]

                if len(X_train_emb) == 0 or len(X_val_emb) == 0:
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_emb)
                X_val_scaled = scaler.transform(X_val_emb)

                n_comp_actual = min(n_comp, X_train_scaled.shape[0], X_train_scaled.shape[1])
                pca = PCA(n_components=n_comp_actual)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_val_pca = pca.transform(X_val_scaled)

                svm = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
                svm.fit(X_train_pca, y_train_emb)

                y_prob_windows = svm.predict_proba(X_val_pca)[:, 1]

                unique_subjects = np.unique(subj_val_emb)
                y_true_subj = []
                y_prob_subj = []

                for subj_id in unique_subjects:
                    subj_mask = subj_val_emb == subj_id
                    y_true_subj.append(y_val_emb[subj_mask][0])
                    y_prob_subj.append(y_prob_windows[subj_mask].mean())

                y_true_subj = np.array(y_true_subj)
                y_prob_subj = np.array(y_prob_subj)

                try:
                    fpr, tpr, thresholds = roc_curve(y_true_subj, y_prob_subj)
                    fold_auc = auc(fpr, tpr)
                    best_thr = thresholds[np.argmax(tpr - fpr)]
                except:
                    fold_auc = 0.5
                    best_thr = 0.5

                y_pred_subj = (y_prob_subj >= best_thr).astype(int)

                fold_metrics.append({
                    "auc": fold_auc,
                    "acc": np.mean(y_pred_subj == y_true_subj),
                    "sens": np.mean(y_pred_subj[y_true_subj == 1] == 1) if np.sum(y_true_subj == 1) > 0 else 0,
                    "spec": np.mean(y_pred_subj[y_true_subj == 0] == 0) if np.sum(y_true_subj == 0) > 0 else 0,
                    "threshold": best_thr,
                    "y_true": y_true_subj,
                    "y_prob": y_prob_subj,
                })

            if len(fold_metrics) == 0:
                continue

            mean_auc = np.mean([m["auc"] for m in fold_metrics])

            if mean_auc > 0.75:
                print(f"  PCA={n_comp_actual:<3} C={C:<5} kernel={kernel:<7} → AUC={mean_auc:.3f}")

            if mean_auc > best_combo_auc:
                best_combo_auc = mean_auc
                best_combo = {"pca": n_comp_actual, "C": C, "kernel": kernel}
                best_combo_folds = fold_metrics

        if best_combo_folds is None:
            print(f"  ⚠️ No se pudo evaluar {emb_name}")
            continue

        aucs  = [m["auc"] for m in best_combo_folds]
        accs  = [m["acc"] for m in best_combo_folds]
        senss = [m["sens"] for m in best_combo_folds]
        specs = [m["spec"] for m in best_combo_folds]

        results[emb_name] = {
            "best_pca": best_combo["pca"],
            "best_C": best_combo["C"],
            "best_kernel": best_combo["kernel"],
            "auc_mean": np.mean(aucs),  "auc_std": np.std(aucs),
            "acc_mean": np.mean(accs),  "acc_std": np.std(accs),
            "sens_mean": np.mean(senss),"sens_std": np.std(senss),
            "spec_mean": np.mean(specs),"spec_std": np.std(specs),
            "fold_metrics": best_combo_folds,
        }

    return results

def extract_fusion_embeddings(
    df: pl.DataFrame,
    waveforms_processed: list[np.ndarray],
    model_names: list[str],
    device: str = "mps",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae embeddings de múltiples modelos preentrenados y los concatena.

    Parameters
    ----------
    df : pl.DataFrame
        Metadata con columna "label".
    waveforms_processed : list[np.ndarray]
        Señales preprocesadas.
    model_names : list[str]
        Nombres HuggingFace de los modelos a fusionar.
    device : str
        Dispositivo de cómputo.

    Returns
    -------
    emb_fusion : np.ndarray
        Embeddings concatenados (n_sujetos, sum(dims)).
    labels : np.ndarray
        Etiquetas por sujeto.
    """
    all_indices = np.arange(len(waveforms_processed))
    all_embeddings = []
    labels = None

    print("Extrayendo embeddings para fusión...")

    for model_name in model_names:
        emb, lab = extract_embeddings(df, waveforms_processed, all_indices, model_name, device)
        print(f"  {model_name}: {emb.shape}")
        all_embeddings.append(emb)
        labels = lab

    emb_fusion = np.concatenate(all_embeddings, axis=1)
    print(f"  Fusión: {emb_fusion.shape}")

    return emb_fusion, labels

def run_fusion_svm_search(
    fusion_configs: dict[str, np.ndarray],
    labels: np.ndarray,
    pca_components: list[int],
    svm_c: list[float],
    svm_kernels: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    K-Fold con embeddings fusionados (concatenados) + PCA + SVM.

    Evalúa cada configuración de fusión con grid search sobre
    (PCA, C, kernel) a nivel de sujeto.

    Parameters
    ----------
    fusion_configs : dict[str, np.ndarray]
        Nombre de fusión → matriz de embeddings (n_sujetos, dim).
    labels : np.ndarray
        Etiquetas por sujeto.
    pca_components, svm_c, svm_kernels
        Grids de búsqueda.

    Returns
    -------
    dict
        Resultados por configuración de fusión.
    """
    all_indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fusion_results = {}

    for fusion_name, emb_data in fusion_configs.items():
        print(f"\n{'='*60}")
        print(f"  {fusion_name} ({emb_data.shape[1]}-dim) → PCA + SVM")
        print(f"{'='*60}")

        best_combo_auc = -1
        best_combo = None
        best_combo_folds = None

        for n_comp, C, kernel in itertools.product(pca_components, svm_c, svm_kernels):
            fold_metrics = []

            for train_idx, val_idx in skf.split(all_indices, labels):
                X_tr = emb_data[train_idx]
                y_tr = labels[train_idx]
                X_vl = emb_data[val_idx]
                y_vl = labels[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_vl_s = scaler.transform(X_vl)

                n_actual = min(n_comp, X_tr_s.shape[0], X_tr_s.shape[1])
                pca = PCA(n_components=n_actual)
                X_tr_pca = pca.fit_transform(X_tr_s)
                X_vl_pca = pca.transform(X_vl_s)

                svm = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
                svm.fit(X_tr_pca, y_tr)
                y_prob = svm.predict_proba(X_vl_pca)[:, 1]

                try:
                    fpr, tpr, thresholds = roc_curve(y_vl, y_prob)
                    fold_auc = auc(fpr, tpr)
                    best_thr = thresholds[np.argmax(tpr - fpr)]
                except:
                    fold_auc = 0.5
                    best_thr = 0.5

                y_pred = (y_prob >= best_thr).astype(int)
                fold_metrics.append({
                    "auc": fold_auc,
                    "acc": np.mean(y_pred == y_vl),
                    "sens": np.mean(y_pred[y_vl == 1] == 1) if np.sum(y_vl == 1) > 0 else 0,
                    "spec": np.mean(y_pred[y_vl == 0] == 0) if np.sum(y_vl == 0) > 0 else 0,
                })

            mean_auc = np.mean([m["auc"] for m in fold_metrics])

            if mean_auc > 0.75:
                print(f"  PCA={n_actual:<3} C={C:<5} kernel={kernel:<7} → AUC={mean_auc:.3f}")

            if mean_auc > best_combo_auc:
                best_combo_auc = mean_auc
                best_combo = {"pca": n_actual, "C": C, "kernel": kernel}
                best_combo_folds = fold_metrics

        if best_combo_folds is None:
            print(f"  ⚠️ No se pudo evaluar {fusion_name}")
            continue

        aucs  = [m["auc"] for m in best_combo_folds]
        accs  = [m["acc"] for m in best_combo_folds]
        senss = [m["sens"] for m in best_combo_folds]
        specs = [m["spec"] for m in best_combo_folds]

        fusion_results[fusion_name] = {
            "best_pca": best_combo["pca"],
            "best_C": best_combo["C"],
            "best_kernel": best_combo["kernel"],
            "auc_mean": np.mean(aucs),  "auc_std": np.std(aucs),
            "acc_mean": np.mean(accs),  "acc_std": np.std(accs),
            "sens_mean": np.mean(senss),"sens_std": np.std(senss),
            "spec_mean": np.mean(specs),"spec_std": np.std(specs),
        }

    return fusion_results
