# -*- coding: utf-8 -*-
"""
train_raw_lstm_window.py

Casuística 2a — Raw + CNN por canal + LSTM con ventana temporal.

Pipeline:

  WINDOW chunks consecutivos  (contexto de 60 s)
      │
      ▼  para cada chunk (22, T):
         CNN aplicada a cada canal por separado (pesos compartidos)
         → 22 vectores de CNN_OUT dims
         → flatten → vector de 22*CNN_OUT dims  (representación del chunk)
      │
      ▼  secuencia (B, WINDOW, 22*CNN_OUT)
  LSTM(input=22*CNN_OUT, hidden=HIDDEN, layers=N_LAYERS)
      │
      ▼  último hidden state (B, HIDDEN)
  head(LayerNorm + Dropout + Linear)
      │
      ▼  logit  ->  ictal (1) / interictal (0)  del ÚLTIMO chunk

La CNN se aplica POR CANAL con pesos compartidos: la misma red procesa
cada uno de los 22 canales de cada chunk. Aprende patrones temporales
locales (formas de onda, oscilaciones) dentro de cada canal antes de
comprimir. Los 22 vectores resultantes se concatenan (flatten) para
formar la representación del chunk completo, preservando la identidad
de cada canal. Ese vector es UN PASO de la secuencia que recibe la
LSTM — la secuencia son los WINDOW chunks en el tiempo, no los canales.

La etiqueta que se predice es la del ÚLTIMO chunk de la ventana.
Los chunks anteriores son el contexto temporal de la LSTM.

Normalización: z-score por canal sobre TODA la grabación.
Etiquetas: ictal (1) vs interictal (0).
Los chunks preictal y postictal se DESCARTAN (no son ni clase positiva ni
negativa), igual que antes se descartaban ictal/postictal en la tarea
preictal-vs-interictal. Así el interictal sigue siendo "lejos de crisis".
Validación: K-fold a nivel de PACIENTE (group k-fold), SEED=123.

Métricas (todas calculadas sobre la ÚLTIMA época, época 30):
  - Agregado entre folds: media ± std de la última época a través de los
    folds (recall, precision, f1, specificity, acc).
  - A NIVEL DE PACIENTE: en cada fold se acumula la matriz de confusión
    por paciente de los pacientes en VALIDACIÓN, y se calculan recall /
    precision / f1 / specificity por paciente. Como el k-fold es por
    paciente (leave-subjects-out), cada paciente se valida exactamente una
    vez, así que al final se obtiene un set de métricas por paciente
    sacado del fold en el que le tocó testear. Se añade además la media ±
    std a través de los pacientes (ignorando NaN).

Balanceo de clases:
  - TRAIN: submuestreo del interictal a 1:1 (INTERICTAL_RATIO=1).
  - VALIDACIÓN: SIN balancear (VAL_INTERICTAL_RATIO=None). Se conserva
    todo el interictal para que recall / precision / specificity reflejen
    la distribución real de clases y los resultados sean más fiables.

Checkpoint: se guarda el modelo de la ÚLTIMA época (no el de mejor F1).
"""

from __future__ import annotations

import random
import time
import csv
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import mne

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT    = Path(r"C:\Users\mcasesf\Desktop\TFG-SeizurePrediction\chb-mit-scalp-eeg-database-1.0.0")
SEIZURES_CSV = Path(r"C:\Users\mcasesf\Desktop\TFG-SeizurePrediction\seizures_manifest.csv")

# --- Split / esquema de partición k-fold ---
# Esta casuística usa group k-fold a NIVEL DE PACIENTE (leave-subjects-out).
# SPLIT se añade al nombre del run de W&B y al de los ficheros/carpeta de
# salida, para no colisionar con runs ni ficheros de configuraciones
# anteriores (p.ej. val balanceada) ni de otros esquemas de partición.
SPLIT = "chunk"

CKPT_DIR  = Path(f"checkpoints_raw_plain_lstm-withCNN_casuistica2_ictal_{SPLIT}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def last_path_for_fold(fold_idx: int) -> Path:
    """Checkpoint de la ÚLTIMA época de cada fold (namespaced por fold)."""
    return CKPT_DIR / f"last_fold{fold_idx}.pt"

# Métricas por época de TODOS los folds (incluye columna 'fold')
OUT_METRICS_CSV = Path(f"eval_raw_lstm_window_per_epoch_casuistica2_ictal_kfold_{SPLIT}.csv")
# Agregado POR ÉPOCA a través de folds: media ± std en cada época (curva)
OUT_PER_EPOCH_AGG_CSV = Path(f"eval_raw_lstm_window_per_epoch_agg_casuistica2_ictal_kfold_{SPLIT}.csv")
# Resumen: última época de cada fold + media ± std a través de folds
OUT_SUMMARY_CSV = Path(f"eval_raw_lstm_window_kfold_summary_casuistica2_ictal_{SPLIT}.csv")
# Métricas a NIVEL DE PACIENTE (última época): una fila por paciente +
# media ± std a través de pacientes. Cada paciente sale del fold en el que
# estuvo en validación (k-fold por paciente -> cada paciente se valida 1 vez).
OUT_PER_PATIENT_CSV = Path(f"eval_raw_lstm_window_per_patient_casuistica2_ictal_kfold_{SPLIT}.csv")

# --- Chunking ---
CHUNK_SEC  = 2.0
STRIDE_SEC = 2.0
C          = 22          # número de canales

# --- Ventana temporal ---
WINDOW     = 30          # chunks por secuencia (30 * 2s = 60 s de historia)
STRIDE_WIN = 1           # desplazamiento de la ventana ICTAL en chunks
                         # (stride 1 -> se capturan todas las ventanas cuyo
                         # último chunk es ictal; la clase positiva es escasa
                         # y conviene muestrearla densamente)
STRIDE_WIN_INTERICTAL = WINDOW   # = 30 -> desplazamiento de la ventana
                         # INTERICTAL. Al ser igual a WINDOW, las ventanas
                         # interictales NO se solapan (cada una empieza donde
                         # acaba la anterior: 30 chunks = 60 s más tarde). Esto
                         # reduce ~30x el nº de ventanas interictales tanto en
                         # train como en val/test.

# --- Etiquetado de crisis ---
SPH_MIN       = 0
SOP_MIN       = 5
POSTICTAL_MIN = 5

# --- Submuestreo del interictal ---
INTERICTAL_RATIO     = 1     # 1 interictal por ictal en TRAIN (balanceado 1:1)
VAL_INTERICTAL_RATIO = None  # VALIDACIÓN SIN balancear: se conserva todo el
                             # interictal para que recall/precision/specificity
                             # reflejen la distribución real (resultados fiables)

# --- CNN por canal ---
# Aplicada a cada canal de cada chunk por separado (pesos compartidos).
# Salida: 22 vectores de CNN_OUT dims -> flatten -> 22*CNN_OUT dims por chunk.
CNN_CHANNELS = (16, 32, 64)   # filtros de las 3 capas conv
CNN_KERNELS  = (7,  5,  3)    # kernel size de cada capa
CNN_POOLS    = (4,  2,  1)    # MaxPool tras cada capa (1 = sin pool)
CNN_OUT      = 64             # dims por canal tras la CNN
CNN_DROPOUT  = 0.1
# dim del vector por chunk que entra a la LSTM = C * CNN_OUT = 22 * 64 = 1408

# --- Arquitectura LSTM ---
HIDDEN   = 256
N_LAYERS = 2
DROPOUT  = 0.3

# --- Entrenamiento ---
BATCH_SIZE   = 128      # la LSTM ve vectores de C*CNN_OUT=1408 por paso
                         # (en vez de C*T=11264 sin CNN), así cabe más batch
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-3
# Nota: sin LR scheduler. Con 30 épocas y Adam no aporta, y un
# ReduceLROnPlateau sobre train casi nunca se dispararía (la loss de train
# baja de forma monótona). LR constante = LR durante todo el entrenamiento.

# --- Verbosidad ---
PRINT_EVERY_N_BATCHES = 10   # progreso dentro de cada epoch (None = silencio)

# --- Split / K-Fold ---
# Validación cruzada a NIVEL DE PACIENTE (group k-fold): los pacientes se
# barajan con SEED y se reparten en N_FOLDS grupos. Cada grupo es la
# validación exactamente una vez; el resto de pacientes son train. Así
# ningún paciente aparece a la vez en train y val de un mismo fold.
N_FOLDS              = 5     # nº de folds (4-5 recomendado con ~24 pacientes)
SEED                 = 123   # baraja de pacientes + init del modelo (igual en todos los folds)
VAL_SEED             = 124   # seed distinto para submuestreo val (no colisiona con train)

# (VAL_PATIENT_FRACTION ya no se usa: la fracción de val la fija 1/N_FOLDS)

PRELOAD_EDF = True

# --- W&B ---
WANDB_PROJECT  = "TFG-seizure-embeddings"
WANDB_RUN_NAME = f"raw_cnn_lstm_window_casuistica2_ictal_kfold_{SPLIT}_intstride30"

CANONICAL_22 = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8",
]


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# EDF HELPERS
# ============================================================
def clean_channel(name):
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    if name in {"", "-"}: return ""
    if name.startswith("--") or name.startswith("."): return ""
    return re.sub(r"-(\d+)$", "", name)


def build_first_index_map(ch_names):
    mp = {}
    for i, ch in enumerate(ch_names):
        c = clean_channel(ch)
        if c and c not in mp:
            mp[c] = i
    return mp


def load_edf_as_X22(edf_path, preload=True):
    """
    Devuelve X (22, T) float32 normalizado por canal sobre TODA la
    grabación (per_recording_channel_zscore), idéntico al resto del estudio.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
    mp  = build_first_index_map(raw.ch_names)
    missing = [ch for ch in CANONICAL_22 if ch not in mp]
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    picks = [mp[ch] for ch in CANONICAL_22]
    X  = raw.get_data(picks=picks).astype(np.float64)
    sf = float(raw.info["sfreq"])
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1,  keepdims=True)
    X    = (X - mean) / (std + 1e-8)
    return X.astype(np.float32), sf


def chunkify(X, sf):
    """Trocea X (C, T_total) en chunks (N, C, tchunk)."""
    Cc, T  = X.shape
    tchunk = int(round(CHUNK_SEC * sf))
    stride = int(round(STRIDE_SEC * sf))
    if T < tchunk:
        return np.zeros((0, Cc, tchunk), dtype=np.float32)
    n   = 1 + (T - tchunk) // stride
    out = np.empty((n, Cc, tchunk), dtype=np.float32)
    for i in range(n):
        a = i * stride
        out[i] = X[:, a:a+tchunk]
    return out


def infer_tchunk(data_root):
    """Longitud de chunk en muestras, leyendo el primer EDF disponible."""
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.startswith("chb"):
            for edf in sorted(p.glob("*.edf")):
                try:
                    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
                    sf  = float(raw.info["sfreq"])
                    return int(round(CHUNK_SEC * sf)), sf
                except Exception:
                    continue
    raise RuntimeError("No EDF found.")


# ============================================================
# SEIZURE MANIFEST + ETIQUETADO
# ============================================================
def load_seizures_manifest(csv_path):
    df = pd.read_csv(csv_path)
    required = {"patient","edf","seizure_idx","start_sec","end_sec"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    df = df.copy()
    df["patient"]   = df["patient"].astype(str)
    df["edf"]       = df["edf"].astype(str)
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"]   = df["end_sec"].astype(float)
    return df


def intervals_overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def label_single_chunk(t0, t1, seizure_rows):
    """Etiqueta de fase de un chunk [t0, t1): ictal / postictal / preictal / interictal."""
    sph = SPH_MIN*60.0; sop = SOP_MIN*60.0; pos = POSTICTAL_MIN*60.0
    for _, row in seizure_rows.iterrows():
        if intervals_overlap(t0, t1, float(row["start_sec"]), float(row["end_sec"])):
            return "ictal"
    for _, row in seizure_rows.iterrows():
        s1 = float(row["end_sec"])
        if intervals_overlap(t0, t1, s1, s1 + pos):
            return "postictal"
    for _, row in seizure_rows.iterrows():
        onset = float(row["start_sec"])
        if intervals_overlap(t0, t1, onset-(sph+sop), onset-sph):
            return "preictal"
    return "interictal"


def label_array_for_chunks(n_chunks, seizure_rows):
    """
    Devuelve (labels_bin, labels_keep), uno por chunk:
      labels_bin  : 1 si ictal, 0 en cualquier otro caso.
      labels_keep : True solo para ictal e interictal (los chunks
                    válidos para clasificación). preictal/postictal -> False.
    """
    t_start = np.arange(n_chunks, dtype=np.float32) * STRIDE_SEC
    t_end   = t_start + CHUNK_SEC
    bins, keeps = [], []
    for a, b in zip(t_start, t_end):
        name = label_single_chunk(float(a), float(b), seizure_rows)
        bins.append(1 if name == "ictal" else 0)
        keeps.append(name in {"ictal", "interictal"})
    return np.array(bins, dtype=np.int64), np.array(keeps, dtype=bool)


# ============================================================
# DATASET  —  chunks raw sueltos (ictal / interictal)
# ============================================================
def build_full_pool(all_edfs, seizures_df):
    """
    Construye el POOL COMPLETO de chunks válidos (split por CHUNK).

    Carga TODOS los EDF una sola vez y trocea/ventana con
    RawWindowDataset(interictal_ratio=None): se conservan todos los chunks
    ictales (stride 1) y todos los interictales sin solape (stride WINDOW),
    SIN submuestrear. El mapeo paciente->idx (idx_to_patient) es GLOBAL (todos
    los pacientes), de modo que los índices de paciente son consistentes en
    todos los folds. El k-fold reparte luego estos chunks por índices.
    """
    return RawWindowDataset(all_edfs, seizures_df,
                            interictal_ratio=None, seed=SEED)


def chunk_kfold_splits(full_ds, n_folds, seed):
    """
    K-fold por CHUNK (mismo estilo que el antiguo split por paciente, pero la
    unidad de partición es el CHUNK, no el paciente).

    Baraja TODOS los chunks válidos del pool con `seed` y los reparte en
    `n_folds` grupos en round-robin (fold[i % n_folds]), SIN tener en cuenta el
    paciente ni el recording de origen. Para cada fold k ese grupo es la
    VALIDACIÓN (~20% de los chunks, 80/20) y el resto es TRAIN. Es el esquema
    más optimista: chunks ictales casi idénticos (stride 1, comparten 29 de 30
    ventanas) pueden caer a la vez en train y val -> fuga temporal en la clase
    positiva (cota superior optimista de rendimiento).

    Yields, por cada fold:
        (fold_idx, train_idx, val_idx)
    con fold_idx en 1..n_folds y train_idx/val_idx listas de índices sobre
    full_ds.samples.
    """
    n = len(full_ds.samples)
    if n < n_folds:
        raise RuntimeError(
            f"Hay {n} chunks válidos pero pediste {n_folds} folds.")

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)

    # reparto round-robin -> folds equilibrados en nº de chunks
    folds = [[] for _ in range(n_folds)]
    for i, s in enumerate(order):
        folds[i % n_folds].append(s)

    print(f"K-fold por CHUNK: {n} chunks válidos -> {n_folds} folds "
          f"(seed={seed})")
    for k, grp in enumerate(folds, 1):
        print(f"  fold {k}: val chunks = {len(grp)} "
              f"({100*len(grp)/max(1,n):.1f}%)")

    for k in range(n_folds):
        val_idx   = folds[k]
        train_idx = [s for j, g in enumerate(folds) if j != k for s in g]
        yield k + 1, train_idx, val_idx


class RawWindowDataset(Dataset):
    """
    Construye secuencias de WINDOW chunks consecutivos.

    Cada sample: (X_seq, label, patient_idx)
      X_seq       : (WINDOW, C, T)  los WINDOW chunks con sus canales intactos
                                    La CNN los procesa canal por canal en el forward
      label       : etiqueta del ÚLTIMO chunk de la ventana
                    0 interictal / 1 ictal
      patient_idx : índice entero del paciente al que pertenece la ventana
                    (se mapea con self.idx_to_patient). Permite calcular
                    métricas por paciente en validación.

    La ventana se desliza con DOS strides distintos según la clase del
    último chunk de la ventana (que es el que determina su etiqueta):
      - ventanas ICTALES: stride STRIDE_WIN (=1), muestreo denso.
      - ventanas INTERICTALES: stride STRIDE_WIN_INTERICTAL (=WINDOW=30),
        de forma que las ventanas interictales NO se solapan.
    Una secuencia se incluye solo si el último chunk es ictal o interictal
    (no preictal/postictal). Los chunks anteriores de la ventana pueden ser de
    cualquier fase — son el contexto temporal de la LSTM.

    Submuestreo: si interictal_ratio no es None, se conservan todos los
    samples ictales y se submuestrean los interictales al ratio dado
    (se usa en TRAIN con SEED para balancear 1:1). Si interictal_ratio es
    None NO se submuestrea nada: se conserva todo el interictal (uso en
    VALIDACIÓN, para no balancear y obtener métricas fiables).
    """
    def __init__(self, edf_files, seizures_df, interictal_ratio=None, seed=123):
        # cada elemento es (seq, patient_name) para arrastrar el paciente
        ict_samples: list[tuple[np.ndarray, str]] = []
        int_samples: list[tuple[np.ndarray, str]] = []

        for edf in edf_files:
            try:
                X, sf  = load_edf_as_X22(edf, preload=PRELOAD_EDF)
                chunks = chunkify(X, sf)                   # (N, C, T)
                N, Cc, T = chunks.shape
                if N < WINDOW:
                    continue

                patient = edf.parent.name
                rows = seizures_df[
                    (seizures_df["patient"] == patient) &
                    (seizures_df["edf"]     == edf.name)
                ].copy()
                labels_bin, labels_keep = label_array_for_chunks(N, rows)

                # Deslizar la ventana — guardamos (C, T) por chunk, sin aplanar.
                # La etiqueta de cada ventana es la de su ÚLTIMO chunk.
                # Se usan DOS strides distintos según la clase del último chunk:
                #
                #   ICTAL      -> stride STRIDE_WIN (=1): muestreo denso, se
                #                 capturan todas las ventanas cuyo último chunk
                #                 es ictal (clase positiva, escasa).
                #   INTERICTAL -> stride STRIDE_WIN_INTERICTAL (=WINDOW=30):
                #                 ventanas interictales SIN solape (cada una
                #                 60 s después de la anterior). Reduce ~30x el
                #                 nº de ventanas interictales.
                #
                # Son dos pasadas independientes: la pasada ictal solo añade
                # ventanas ictales y la interictal solo añade interictales, así
                # que ninguna ventana se cuenta dos veces.

                # --- pasada ICTAL (stride 1) ---
                for end in range(WINDOW, N + 1, STRIDE_WIN):
                    last = end - 1            # índice del último chunk
                    # labels_bin[last]==1 implica ictal (que siempre es 'keep')
                    if labels_bin[last] == 1:
                        seq = chunks[end - WINDOW:end]   # (WINDOW, C, T)
                        ict_samples.append((seq, patient))

                # --- pasada INTERICTAL (stride 30, sin solape) ---
                for end in range(WINDOW, N + 1, STRIDE_WIN_INTERICTAL):
                    last = end - 1
                    # interictal = keep (ictal/interictal) y NO ictal.
                    # El check de keep excluye preictal/postictal.
                    if labels_keep[last] and labels_bin[last] == 0:
                        seq = chunks[end - WINDOW:end]   # (WINDOW, C, T)
                        int_samples.append((seq, patient))

            except Exception as e:
                print(f"  [skip] {edf.name}: {type(e).__name__}: {e}")

        n_ict_total = len(ict_samples)
        n_int_total = len(int_samples)
        print(f"  cargadas: ictal={n_ict_total}  interictal={n_int_total}  "
              f"ratio_natural={n_int_total / max(1, n_ict_total):.1f}x")

        # submuestreo del interictal (solo si interictal_ratio no es None;
        # en validación se pasa None -> NO se submuestrea, val sin balancear)
        if interictal_ratio is not None and n_ict_total > 0:
            n_int_keep = min(n_int_total, n_ict_total * int(interictal_ratio))
            if n_int_keep < n_int_total:
                rng = random.Random(seed)
                idxs = list(range(n_int_total))
                rng.shuffle(idxs)
                int_samples = [int_samples[i] for i in idxs[:n_int_keep]]
                print(f"  submuestreo interictal: {n_int_total} -> {n_int_keep}  "
                      f"(ratio={interictal_ratio}x  seed={seed})")
        else:
            print(f"  SIN submuestreo de interictal (val sin balancear)")

        # --- mapeo paciente -> índice (sobre todos los pacientes de los EDF) ---
        all_pats = sorted({edf.parent.name for edf in edf_files})
        self.idx_to_patient: list[str] = all_pats
        self.patient_to_idx: dict[str, int] = {p: i for i, p in enumerate(all_pats)}

        # samples como (seq, label, patient_idx)
        self.samples: list[tuple[np.ndarray, int, int]] = (
            [(s, 1, self.patient_to_idx[p]) for (s, p) in ict_samples] +
            [(s, 0, self.patient_to_idx[p]) for (s, p) in int_samples]
        )
        self.n_ict = len(ict_samples)
        self.n_int = len(int_samples)

        # conteos por paciente (para reportar nº ictal / interictal en val)
        self.patient_n_ict: Counter = Counter(p for (_, p) in ict_samples)
        self.patient_n_int: Counter = Counter(p for (_, p) in int_samples)
        print(f"  dataset final: ictal={self.n_ict}  interictal={self.n_int}  "
              f"total={len(self.samples)}  "
              f"ratio={self.n_int / max(1, self.n_ict):.1f}x")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, pidx = self.samples[idx]
        # seq: (WINDOW, C, T) — la CNN necesita los canales separados
        X = torch.from_numpy(np.ascontiguousarray(seq)).float()
        y = torch.tensor(label, dtype=torch.float32)
        p = torch.tensor(pidx, dtype=torch.long)
        return X, y, p



# ============================================================
# DATASET-VISTA  —  subconjunto de chunks por índices (split por CHUNK)
# ============================================================
class SamplesDataset(Dataset):
    """
    Dataset-vista sobre una lista de samples ya construidos
    (seq, label, patient_idx), compartiendo el mapeo GLOBAL paciente->idx.

    Permite partir el pool de chunks por índices (split por CHUNK) sin recargar
    EDFs. Expone la misma interfaz que RawWindowDataset en lo que usa el resto
    del pipeline: __getitem__ devuelve (X, y, patient_idx); n_ict / n_int;
    idx_to_patient / patient_to_idx; patient_n_ict / patient_n_int.
    """
    def __init__(self, samples, idx_to_patient):
        self.samples = samples
        self.idx_to_patient: list[str] = idx_to_patient
        self.patient_to_idx: dict[str, int] = {
            p: i for i, p in enumerate(idx_to_patient)}
        self.n_ict = sum(1 for (_, y, _) in samples if y == 1)
        self.n_int = sum(1 for (_, y, _) in samples if y == 0)
        self.patient_n_ict: Counter = Counter(
            idx_to_patient[p] for (_, y, p) in samples if y == 1)
        self.patient_n_int: Counter = Counter(
            idx_to_patient[p] for (_, y, p) in samples if y == 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, pidx = self.samples[idx]
        X = torch.from_numpy(np.ascontiguousarray(seq)).float()
        y = torch.tensor(label, dtype=torch.float32)
        p = torch.tensor(pidx, dtype=torch.long)
        return X, y, p


def make_train_val_datasets(full_ds, train_idx, val_idx,
                            train_interictal_ratio, val_interictal_ratio,
                            seed, fold_idx):
    """
    Construye los datasets de train y val de un fold del split por CHUNK a
    partir de los índices del pool.

    - VAL: todos los chunks de `val_idx` (sin rebalancear) -> se conserva todo
      el interictal (métricas fiables con desbalance real). val_interictal_ratio
      se mantiene por simetría con la API anterior; con None (por defecto) el
      val no se submuestrea.
    - TRAIN: si train_interictal_ratio no es None, se submuestrea el interictal
      al ratio dado (1:1 con SEED) sobre los chunks de `train_idx`; el ictal se
      conserva íntegro. El balanceo se aplica SOLO al train.
    """
    idx_to_patient = full_ds.idx_to_patient
    samples = full_ds.samples

    # --- VAL ---
    val_samples = [samples[i] for i in val_idx]
    if val_interictal_ratio is not None:
        v_ict = [samples[i] for i in val_idx if samples[i][1] == 1]
        v_int = [samples[i] for i in val_idx if samples[i][1] == 0]
        n_keep = min(len(v_int), len(v_ict) * int(val_interictal_ratio))
        if len(v_ict) > 0 and n_keep < len(v_int):
            rng = random.Random(seed + 1000 + fold_idx)
            order = list(range(len(v_int)))
            rng.shuffle(order)
            v_int = [v_int[i] for i in order[:n_keep]]
        val_samples = v_ict + v_int

    # --- TRAIN (submuestreo del interictal SOLO aquí) ---
    t_ict = [samples[i] for i in train_idx if samples[i][1] == 1]
    t_int = [samples[i] for i in train_idx if samples[i][1] == 0]
    n_ict, n_int = len(t_ict), len(t_int)
    if train_interictal_ratio is not None and n_ict > 0:
        n_keep = min(n_int, n_ict * int(train_interictal_ratio))
        if n_keep < n_int:
            rng = random.Random(seed)
            order = list(range(n_int))
            rng.shuffle(order)
            t_int = [t_int[i] for i in order[:n_keep]]
            print(f"  [fold {fold_idx}] submuestreo interictal (train): "
                  f"{n_int} -> {n_keep}  "
                  f"(ratio={train_interictal_ratio}x  seed={seed})")
    train_samples = t_ict + t_int

    train_ds = SamplesDataset(train_samples, idx_to_patient)
    val_ds   = SamplesDataset(val_samples, idx_to_patient)
    print(f"  [fold {fold_idx}] train chunks: ictal={train_ds.n_ict} "
          f"interictal={train_ds.n_int}  |  val chunks: ictal={val_ds.n_ict} "
          f"interictal={val_ds.n_int}")
    return train_ds, val_ds


# ============================================================
# MODELO  —  CNN por canal + flatten + LSTM sobre ventana temporal
# ============================================================
class ChannelCNN(nn.Module):
    """
    CNN 1D pequeña aplicada POR CANAL (pesos compartidos entre canales).

    Entrada: (B*, 1, T)   un canal de un chunk
    Salida:  (B*, CNN_OUT) vector representativo del canal

    Aprende patrones temporales locales dentro de cada canal
    (formas de onda, oscilaciones) antes de comprimir a CNN_OUT dims.
    """
    def __init__(self, channels=CNN_CHANNELS, kernels=CNN_KERNELS,
                 pools=CNN_POOLS, out_dim=CNN_OUT, dropout=CNN_DROPOUT):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernels
        p1, p2, p3 = pools

        def block(in_ch, out_ch, k, p):
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            if p > 1:
                layers.append(nn.MaxPool1d(p))
            return nn.Sequential(*layers)

        self.conv1 = block(1,  c1, k1, p1)
        self.conv2 = block(c1, c2, k2, p2)
        self.conv3 = block(c2, c3, k3, p3)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.proj  = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, out_dim),
        )

    def forward(self, x):
        # x: (B*, 1, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)        # (B*, c3, 1)
        return self.proj(x)     # (B*, CNN_OUT)


class RawCNNWindowLSTM(nn.Module):
    """
    Pipeline completo:

      X_seq (B, WINDOW, C, T)
        ↓  para cada chunk de la secuencia:
           reshape -> (B*WINDOW*C, 1, T)
           CNN por canal (pesos compartidos) -> (B*WINDOW*C, CNN_OUT)
           reshape -> (B*WINDOW, C, CNN_OUT)
           flatten canales -> (B*WINDOW, C*CNN_OUT)   <- representación del chunk
        ↓
        reshape -> (B, WINDOW, C*CNN_OUT)             <- secuencia para la LSTM
        LSTM(input=C*CNN_OUT, hidden=HIDDEN)
        ↓
        último hidden state (B, HIDDEN)
        head(LayerNorm + Dropout + Linear)
        ↓
        logit (B,)   del ÚLTIMO chunk de la ventana
    """
    def __init__(self, cnn_out=CNN_OUT, hidden_size=HIDDEN,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.cnn  = ChannelCNN(out_dim=cnn_out)
        lstm_input = C * cnn_out           # 22 * 64 = 1408
        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, X_seq):
        """
        X_seq: (B, WINDOW, C, T)
        return logit: (B,)
        """
        B, W, Cc, T = X_seq.shape

        # aplicar CNN a todos los canales de todos los chunks a la vez
        x = X_seq.reshape(B * W * Cc, 1, T)       # (B*WINDOW*C, 1, T)
        f = self.cnn(x)                            # (B*WINDOW*C, CNN_OUT)
        f = f.reshape(B * W, Cc, -1)               # (B*WINDOW, C, CNN_OUT)
        f = f.reshape(B * W, Cc * f.shape[-1])     # (B*WINDOW, C*CNN_OUT)
        f = f.reshape(B, W, -1)                    # (B, WINDOW, C*CNN_OUT)

        _, (hn, _) = self.lstm(f)                  # hn: (layers, B, HIDDEN)
        h = hn[-1]                                 # (B, HIDDEN)
        return self.head(h).squeeze(-1)            # (B,)


# ============================================================
# MÉTRICAS  —  idénticas a lstm_baseline.py
# ============================================================
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """logits: (B,) sin sigmoid;  labels: (B,) 0/1 float."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    acc         = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "acc": float(acc), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "specificity": float(specificity),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def metrics_from_counts(tp, tn, fp, fn) -> dict:
    """
    Métricas a partir de conteos de confusión (enteros). Devuelve NaN cuando
    el denominador es 0 (p.ej. precision si el modelo no predice positivos
    para ese paciente, o recall si el paciente no tiene ventanas ictales).
    """
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    tot = tp + tn + fp + fn
    acc         = (tp + tn) / tot           if tot > 0          else float("nan")
    recall      = tp / (tp + fn)            if (tp + fn) > 0    else float("nan")
    precision   = tp / (tp + fp)            if (tp + fp) > 0    else float("nan")
    specificity = tn / (tn + fp)            if (tn + fp) > 0    else float("nan")
    if (np.isnan(recall) or np.isnan(precision)
            or (precision + recall) == 0):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "acc": acc, "precision": precision, "recall": recall,
        "f1": f1, "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, device, optimizer, criterion,
                    epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    n_batches = len(loader)
    header_tag = (f"[Epoch {epoch:03d}/{total_epochs}] "
                  if epoch is not None and total_epochs is not None else "")
    print(f"\n{header_tag}TRAIN — {n_batches} batches", flush=True)

    t_epoch_start = time.time()
    t_block_start = time.time()
    running_tp = running_tn = running_fp = running_fn = 0

    for b_idx, (X, labels, _pid) in enumerate(loader, 1):
        X      = X.to(device)
        labels = labels.to(device)

        logits = model(X)
        loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        # contadores rápidos para mostrar f1 acumulado de los últimos N batches
        with torch.no_grad():
            preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
            running_tp += int(((preds == 1) & (labels == 1)).sum())
            running_tn += int(((preds == 0) & (labels == 0)).sum())
            running_fp += int(((preds == 1) & (labels == 0)).sum())
            running_fn += int(((preds == 0) & (labels == 1)).sum())

        if (PRINT_EVERY_N_BATCHES is not None
                and b_idx % PRINT_EVERY_N_BATCHES == 0):
            now = time.time()
            block_secs = now - t_block_start
            sec_per_batch = block_secs / PRINT_EVERY_N_BATCHES
            elapsed = now - t_epoch_start
            eta = sec_per_batch * (n_batches - b_idx)
            avg_loss = total_loss / b_idx

            prec = running_tp / max(1, running_tp + running_fp)
            rec  = running_tp / max(1, running_tp + running_fn)
            f1   = 2 * prec * rec / max(1e-8, prec + rec)

            print(
                f"  train batch {b_idx:>4d}/{n_batches}"
                f"  loss={loss.item():.4f}"
                f"  avg_loss={avg_loss:.4f}"
                f"  f1_run={f1:.3f}"
                f"  recall_run={rec:.3f}"
                f"  {sec_per_batch:.2f}s/b"
                f"  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )
            t_block_start = now

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(1, n_batches)

    epoch_secs = time.time() - t_epoch_start
    print(
        f"  train epoch done  "
        f"loss={metrics['loss']:.4f}  f1={metrics['f1']:.4f}  "
        f"recall={metrics['recall']:.4f}  precision={metrics['precision']:.4f}  "
        f"acc={metrics['acc']:.4f}  ({epoch_secs:.1f}s)",
        flush=True,
    )
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion,
                   epoch=None, total_epochs=None,
                   collect_patients=False, n_patients=0):
    """
    Evalúa una época sobre `loader`.

    Si collect_patients=True, acumula además la matriz de confusión POR
    PACIENTE (usando el índice de paciente que devuelve el dataset) y la
    deja en metrics["per_patient_counts"] = {"tp","tn","fp","fn"} (arrays
    de tamaño n_patients). Se usa solo en la ÚLTIMA época para sacar las
    métricas por paciente sin recorrer el val dos veces.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    all_pids = [] if collect_patients else None

    n_batches = len(loader)
    header_tag = (f"[Epoch {epoch:03d}/{total_epochs}] "
                  if epoch is not None and total_epochs is not None else "")
    print(f"{header_tag}VAL   — {n_batches} batches", flush=True)

    t_epoch_start = time.time()
    t_block_start = time.time()

    for b_idx, (X, labels, pids) in enumerate(loader, 1):
        X      = X.to(device)
        labels = labels.to(device)

        logits = model(X)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
        if collect_patients:
            all_pids.append(pids.detach().cpu())

        if (PRINT_EVERY_N_BATCHES is not None
                and b_idx % PRINT_EVERY_N_BATCHES == 0):
            now = time.time()
            sec_per_batch = (now - t_block_start) / PRINT_EVERY_N_BATCHES
            elapsed = now - t_epoch_start
            eta = sec_per_batch * (n_batches - b_idx)
            avg_loss = total_loss / b_idx
            print(
                f"  val   batch {b_idx:>4d}/{n_batches}"
                f"  loss={loss.item():.4f}"
                f"  avg_loss={avg_loss:.4f}"
                f"  {sec_per_batch:.2f}s/b"
                f"  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )
            t_block_start = now

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(1, n_batches)

    # --- confusión POR PACIENTE (solo si se pidió; típicamente última época) ---
    if collect_patients:
        pids  = torch.cat(all_pids).numpy()
        probs = torch.sigmoid(all_logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        lab   = all_labels.cpu().numpy().astype(np.int64)
        tp = np.zeros(n_patients, dtype=np.int64)
        tn = np.zeros(n_patients, dtype=np.int64)
        fp = np.zeros(n_patients, dtype=np.int64)
        fn = np.zeros(n_patients, dtype=np.int64)
        for pidx in range(n_patients):
            m = (pids == pidx)
            if not m.any():
                continue
            pr = preds[m]; lb = lab[m]
            tp[pidx] = int(((pr == 1) & (lb == 1)).sum())
            tn[pidx] = int(((pr == 0) & (lb == 0)).sum())
            fp[pidx] = int(((pr == 1) & (lb == 0)).sum())
            fn[pidx] = int(((pr == 0) & (lb == 1)).sum())
        metrics["per_patient_counts"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    epoch_secs = time.time() - t_epoch_start
    print(
        f"  val   epoch done  "
        f"loss={metrics['loss']:.4f}  f1={metrics['f1']:.4f}  "
        f"recall={metrics['recall']:.4f}  precision={metrics['precision']:.4f}  "
        f"acc={metrics['acc']:.4f}  ({epoch_secs:.1f}s)",
        flush=True,
    )
    return metrics


# ============================================================
# ENTRENAMIENTO DE UN FOLD
# ============================================================
def run_one_fold(fold_idx, n_folds, train_ds, val_ds, tchunk, sf0, device):
    """
    Entrena un fold completo (EPOCHS épocas) y devuelve un dict con:
        - 'last'               : métricas de val de la ÚLTIMA época (lo agregado)
        - 'per_patient_counts' : matriz de confusión por paciente (última época)
                                 de los chunks de val de ESTE fold, alineada al
                                 índice GLOBAL de paciente. main la acumula a
                                 través de folds -> 1 fila por paciente.
        - 'csv_rows'           : filas por época (col 'fold') para el CSV global
        - 'val_history' / 'train_history' : métricas por época
        - 'n_int'/'n_ict'      : conteos de train

    El checkpoint guardado es el de la ÚLTIMA época (no el de mejor F1).
    train_ds / val_ds ya vienen construidos (split por CHUNK): el train con el
    interictal submuestreado 1:1 y el val sin rebalancear.
    """
    # Mismo init de modelo y orden de baraja en todos los folds
    seed_everything(SEED)

    tag = f"FOLD {fold_idx}/{n_folds}"
    # En el split por CHUNK todos los pacientes aparecen tanto en train como en
    # val; los derivamos de los datasets solo para logging / config de W&B.
    train_pats = sorted({train_ds.idx_to_patient[p]
                         for (_, _, p) in train_ds.samples})
    val_pats   = sorted({val_ds.idx_to_patient[p]
                         for (_, _, p) in val_ds.samples})
    print("\n" + "#" * 64)
    print(f"# {tag}")
    print(f"#   pacientes en val   ({len(val_pats)})")
    print(f"#   pacientes en train ({len(train_pats)})")
    print(f"#   chunks: train={len(train_ds)}  val={len(val_ds)}")
    print("#" * 64)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"[{tag}] Dataset vacío (train={len(train_ds)} val={len(val_ds)}): "
            f"revisa etiquetas / rutas / reparto de chunks.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # --- pos_weight = n_interictal / n_ictal ---
    n0, n1 = train_ds.n_int, train_ds.n_ict
    print(f"[{tag}] Train sequences  ->  interictal: {n0}  ictal: {n1}  "
          f"ratio: {n0/max(1,n1):.1f}x")
    pos_weight = torch.tensor([n0 / max(1, n1)], dtype=torch.float32,
                              device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Modelo (init idéntico en todos los folds por seed_everything(SEED)) ---
    model = RawCNNWindowLSTM(
        cnn_out     = CNN_OUT,
        hidden_size = HIDDEN,
        num_layers  = N_LAYERS,
        dropout     = DROPOUT,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_cnn   = sum(p.numel() for p in model.cnn.parameters()  if p.requires_grad)
    n_lstm  = sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
    n_head  = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    lstm_input = C * CNN_OUT
    print(f"\n[{tag}] Modelo RawCNNWindowLSTM")
    print(f"  CNN por canal   : {CNN_CHANNELS} filtros, kernels {CNN_KERNELS} -> {CNN_OUT} dims/canal")
    print(f"  repr. por chunk : {C} canales × {CNN_OUT} dims = {lstm_input} dims (flatten)")
    print(f"  secuencia LSTM  : {WINDOW} pasos (un chunk por paso)")
    print(f"  parámetros      : {n_train:,}  "
          f"(CNN={n_cnn:,}  LSTM={n_lstm:,}  head={n_head:,})")

    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    # Sin LR scheduler: LR constante = LR durante las EPOCHS épocas.

    # --- W&B: un run por fold, agrupados por WANDB_RUN_NAME ---
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}_fold{fold_idx}of{n_folds}",
        group=WANDB_RUN_NAME,
        job_type="fold",
        reinit=True,
        config={
            "casuistica":    "raw_cnn_lstm_window",
            "task":          "ictal_vs_interictal",
            "sequence":      "window_chunks",
            "cv":            "chunk_kfold",
            "split":         SPLIT,
            "n_folds":       n_folds,
            "fold":          fold_idx,
            "val_patients":  val_pats,
            "train_patients": train_pats,
            "C": C, "tchunk": tchunk, "sfreq": sf0,
            "window":        WINDOW,
            "stride_win":    STRIDE_WIN,
            "stride_win_interictal": STRIDE_WIN_INTERICTAL,
            "cnn_channels":  list(CNN_CHANNELS),
            "cnn_kernels":   list(CNN_KERNELS),
            "cnn_pools":     list(CNN_POOLS),
            "cnn_out":       CNN_OUT,
            "cnn_dropout":   CNN_DROPOUT,
            "lstm_input":    C * CNN_OUT,
            "HIDDEN": HIDDEN, "N_LAYERS": N_LAYERS, "DROPOUT": DROPOUT,
            "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
            "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "chunk_sec": CHUNK_SEC, "stride_sec": STRIDE_SEC,
            "sph_min": SPH_MIN, "sop_min": SOP_MIN,
            "postictal_min": POSTICTAL_MIN,
            "interictal_ratio": INTERICTAL_RATIO,
            "val_interictal_ratio": VAL_INTERICTAL_RATIO,
            "val_seed": VAL_SEED,
            "n_params_trainable": n_train,
            "n_params_cnn":  n_cnn,
            "n_params_lstm": n_lstm,
            "n_params_head": n_head,
            "pos_weight": float(pos_weight.item()),
            "train_seq_ictal":      train_ds.n_ict,
            "train_seq_interictal": train_ds.n_int,
            "val_seq_ictal":        val_ds.n_ict,
            "val_seq_interictal":   val_ds.n_int,
        },
    )
    wandb.watch(model, log="gradients", log_freq=200)

    # --- Bucle de entrenamiento ---
    last_val_m  = None
    last_pp_counts = None           # confusión por paciente de la última época
    n_val_patients = len(val_ds.idx_to_patient)
    last_path   = last_path_for_fold(fold_idx)
    t0 = time.time()
    csv_rows = []
    val_history   = []   # métricas de val por época (para agregar entre folds)
    train_history = []   # idem train

    for epoch in range(1, EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, device,
                                  optimizer, criterion,
                                  epoch=epoch, total_epochs=EPOCHS)
        # En la ÚLTIMA época pedimos la confusión por paciente (un solo pase)
        is_last = (epoch == EPOCHS)
        val_m   = eval_one_epoch(model, val_loader, device, criterion,
                                 epoch=epoch, total_epochs=EPOCHS,
                                 collect_patients=is_last,
                                 n_patients=n_val_patients)
        # sacamos las arrays por paciente del dict de métricas escalares
        if is_last:
            last_pp_counts = val_m.pop("per_patient_counts", None)

        last_val_m = val_m
        val_history.append(val_m)
        train_history.append(train_m)

        elapsed = time.time() - t0
        print(
            f"[{tag}][Epoch {epoch:03d}/{EPOCHS}] "
            f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
            f"val_f1={val_m['f1']:.4f}  val_recall={val_m['recall']:.4f}  "
            f"val_acc={val_m['acc']:.4f}  ({elapsed:.1f}s)"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss":        train_m["loss"],
            "train/acc":         train_m["acc"],
            "train/f1":          train_m["f1"],
            "train/recall":      train_m["recall"],
            "train/precision":   train_m["precision"],
            "train/specificity": train_m["specificity"],
            "train/tp":          train_m["tp"],
            "train/tn":          train_m["tn"],
            "train/fp":          train_m["fp"],
            "train/fn":          train_m["fn"],
            "val/loss":          val_m["loss"],
            "val/acc":           val_m["acc"],
            "val/f1":            val_m["f1"],
            "val/recall":        val_m["recall"],
            "val/precision":     val_m["precision"],
            "val/specificity":   val_m["specificity"],
            "val/tp":            val_m["tp"],
            "val/tn":            val_m["tn"],
            "val/fp":            val_m["fp"],
            "val/fn":            val_m["fn"],
        }, step=epoch)

        csv_rows.append({"fold": fold_idx, "epoch": epoch, "split": "train", **train_m})
        csv_rows.append({"fold": fold_idx, "epoch": epoch, "split": "val",   **val_m})

    # --- Guardar checkpoint de la ÚLTIMA época (no del mejor F1) ---
    torch.save({
        "fold":        fold_idx,
        "epoch":       EPOCHS,
        "model_state": model.state_dict(),
        "opt_state":   optimizer.state_dict(),
        "val_metrics": last_val_m,
        "config": {
            "tchunk": tchunk, "C": C,
            "WINDOW": WINDOW, "STRIDE_WIN": STRIDE_WIN,
            "STRIDE_WIN_INTERICTAL": STRIDE_WIN_INTERICTAL,
            "CNN_CHANNELS": list(CNN_CHANNELS),
            "CNN_KERNELS":  list(CNN_KERNELS),
            "CNN_POOLS":    list(CNN_POOLS),
            "CNN_OUT":      CNN_OUT,
            "HIDDEN": HIDDEN,
            "N_LAYERS": N_LAYERS, "DROPOUT": DROPOUT,
            "task": "ictal_vs_interictal",
            "casuistica": "raw_cnn_lstm_window",
            "cv": "chunk_kfold",
            "n_folds": n_folds,
            "checkpoint": "last_epoch",
        },
    }, last_path)
    try:
        artifact = wandb.Artifact(
            name=f"raw-cnn-lstm-casuistica2-ictal-{SPLIT}-fold{fold_idx}-last",
            type="model",
            metadata={"val_f1": float(last_val_m["f1"]),
                      "epoch": int(EPOCHS), "fold": int(fold_idx)},
        )
        artifact.add_file(str(last_path))
        run.log_artifact(artifact, aliases=["last"])
    except Exception as e:
        print(f"[wandb] no se pudo registrar el artifact: {e}")

    # --- Resumen del fold en el summary del run ---
    run.summary["last/val_recall"]      = float(last_val_m["recall"])
    run.summary["last/val_f1"]          = float(last_val_m["f1"])
    run.summary["last/val_precision"]   = float(last_val_m["precision"])
    run.summary["last/val_acc"]         = float(last_val_m["acc"])
    run.summary["last/val_specificity"] = float(last_val_m["specificity"])
    run.summary["last/val_loss"]        = float(last_val_m["loss"])

    run.finish()

    print(f"\n[{tag}] DONE.  last_val_recall={last_val_m['recall']:.4f}  "
          f"last_val_f1={last_val_m['f1']:.4f}")
    print(f"[{tag}] Checkpoint (última época): {last_path.resolve()}")

    # Confusión POR PACIENTE de la ÚLTIMA época sobre los chunks de val de
    # ESTE fold (arrays alineados al índice GLOBAL de paciente). En el split
    # por CHUNK cada paciente aparece en val en todos los folds con ~20% de sus
    # chunks; main acumula estos conteos a través de folds y calcula UNA fila
    # por paciente (cada chunk del paciente se valida exactamente una vez).
    return {
        "fold":               fold_idx,
        "last":               last_val_m,
        "per_patient_counts": last_pp_counts,
        "csv_rows":           csv_rows,
        "val_history":        val_history,
        "train_history":      train_history,
        "n_int":              n0,
        "n_ict":              n1,
    }


# ============================================================
# AGREGACIÓN K-FOLD
# ============================================================
def _mean_std(values):
    """media y std muestral (ddof=1); std=0 si hay <2 valores."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def _nan_mean_std(values):
    """
    media y std muestral (ddof=1) ignorando NaN. Devuelve (nan, nan) si todos
    son NaN. Se usa para la agregación a través de PACIENTES, donde algunas
    métricas (p.ej. precision) pueden quedar indefinidas para algún paciente.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


# ============================================================
# MAIN
# ============================================================
def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("=" * 60)
    print("CASUÍSTICA 2a:  raw -> CNN por canal -> LSTM (ventana)")
    print(f"            K-FOLD por CHUNK | {N_FOLDS} folds | ictal vs interictal")
    print("=" * 60)

    # --- tchunk (= longitud de chunk en muestras) ---
    tchunk, sf0 = infer_tchunk(DATA_ROOT)
    print(f"sfreq={sf0} Hz  ->  tchunk={tchunk} samples ({CHUNK_SEC}s)")

    # --- Manifest de crisis ---
    seizures_df = load_seizures_manifest(SEIZURES_CSV)

    # --- Bucle de folds ---
    print()
    per_fold = []
    all_csv_rows = []
    all_patient_rows = []     # métricas por paciente de todos los folds
    t_global = time.time()

    # --- Pool COMPLETO de chunks (split por CHUNK) ---
    # Se construye UNA sola vez sobre TODOS los EDF (sin submuestrear). El
    # k-fold reparte luego los chunks por índices, sin recargar EDFs.
    all_edfs = [e for p in sorted(DATA_ROOT.iterdir())
                if p.is_dir() and p.name.startswith("chb")
                for e in sorted(p.glob("*.edf"))]
    print(f"\nConstruyendo pool completo de chunks sobre {len(all_edfs)} EDFs...")
    full_ds = build_full_pool(all_edfs, seizures_df)
    n_patients = len(full_ds.idx_to_patient)

    # Acumuladores de confusión POR PACIENTE a través de folds (split por
    # CHUNK): sumando los N_FOLDS folds, cada chunk de cada paciente se valida
    # exactamente una vez -> matriz de confusión COMPLETA por paciente.
    pat_tp = np.zeros(n_patients, dtype=np.int64)
    pat_tn = np.zeros(n_patients, dtype=np.int64)
    pat_fp = np.zeros(n_patients, dtype=np.int64)
    pat_fn = np.zeros(n_patients, dtype=np.int64)

    for fold_idx, train_idx, val_idx in \
            chunk_kfold_splits(full_ds, N_FOLDS, SEED):
        train_ds, val_ds = make_train_val_datasets(
            full_ds, train_idx, val_idx,
            train_interictal_ratio=INTERICTAL_RATIO,
            val_interictal_ratio=VAL_INTERICTAL_RATIO,
            seed=SEED, fold_idx=fold_idx,
        )
        res = run_one_fold(
            fold_idx, N_FOLDS, train_ds, val_ds, tchunk, sf0, device,
        )
        per_fold.append(res)
        all_csv_rows.extend(res["csv_rows"])
        c = res["per_patient_counts"]
        if c is not None:
            pat_tp += c["tp"]; pat_tn += c["tn"]
            pat_fp += c["fp"]; pat_fn += c["fn"]

    # --- Agregación de la ÚLTIMA época a través de folds ---
    metric_keys = ["recall", "f1", "precision", "acc", "specificity"]
    last_vals = {k: [r["last"][k] for r in per_fold] for k in metric_keys}
    last_loss = [r["last"]["loss"] for r in per_fold]
    agg = {k: _mean_std(v) for k, v in last_vals.items()}
    loss_mean, loss_std = _mean_std(last_loss)

    # --- Agregación POR ÉPOCA a través de folds (curva media ± std) ---
    # Todos los folds corren EPOCHS épocas; alineamos por índice de época.
    n_ep = min(len(r["val_history"]) for r in per_fold)
    agg_keys = metric_keys + ["loss"]
    per_epoch_agg = []   # un dict por época con *_mean / *_std de val y train
    for e in range(n_ep):
        row = {"epoch": e + 1}
        for split, hist_key in (("val", "val_history"), ("train", "train_history")):
            for k in agg_keys:
                mean, std = _mean_std([r[hist_key][e][k] for r in per_fold])
                row[f"{split}_{k}_mean"] = mean
                row[f"{split}_{k}_std"]  = std
        per_epoch_agg.append(row)

    elapsed_total = time.time() - t_global
    print("\n" + "=" * 64)
    print(f"RESUMEN K-FOLD ({N_FOLDS} folds)  —  métricas de la ÚLTIMA época")
    print("=" * 64)
    print(f"{'fold':>5} | {'recall':>7} {'f1':>7} {'prec':>7} "
          f"{'acc':>7} {'spec':>7} {'loss':>7}")
    print("-" * 64)
    for r in per_fold:
        m = r["last"]
        print(f"{r['fold']:>5} | {m['recall']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['acc']:>7.4f} "
              f"{m['specificity']:>7.4f} {m['loss']:>7.4f}")
    print("-" * 64)
    print(f"{'mean':>5} | {agg['recall'][0]:>7.4f} {agg['f1'][0]:>7.4f} "
          f"{agg['precision'][0]:>7.4f} {agg['acc'][0]:>7.4f} "
          f"{agg['specificity'][0]:>7.4f} {loss_mean:>7.4f}")
    print(f"{'std':>5} | {agg['recall'][1]:>7.4f} {agg['f1'][1]:>7.4f} "
          f"{agg['precision'][1]:>7.4f} {agg['acc'][1]:>7.4f} "
          f"{agg['specificity'][1]:>7.4f} {loss_std:>7.4f}")
    print("=" * 64)
    print(f">>> RECALL (última época) = {agg['recall'][0]:.4f} "
          f"± {agg['recall'][1]:.4f}   (media ± std, n={N_FOLDS} folds)")
    print(f"    Tiempo total k-fold: {elapsed_total/60:.1f} min")

    # --- Construir UNA fila por paciente desde los acumuladores cross-fold ---
    # En el split por CHUNK cada paciente aparece en val en TODOS los folds con
    # ~20% de sus chunks; sumando los N_FOLDS folds cada chunk se valida una vez
    # -> matriz de confusión COMPLETA por paciente (una fila por paciente).
    for pidx, pat in enumerate(full_ds.idx_to_patient):
        tp, tn = int(pat_tp[pidx]), int(pat_tn[pidx])
        fp, fn = int(pat_fp[pidx]), int(pat_fn[pidx])
        if (tp + tn + fp + fn) == 0:
            continue
        m = metrics_from_counts(tp, tn, fp, fn)
        all_patient_rows.append({
            "patient": pat, "fold": "all",
            "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "specificity": m["specificity"], "acc": m["acc"],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_ictal": tp + fn, "n_interictal": tn + fp,
        })

    # --- Agregación A NIVEL DE PACIENTE (última época) ---
    patient_metric_keys = ["recall", "precision", "f1", "specificity", "acc"]
    all_patient_rows.sort(key=lambda r: (r["fold"], r["patient"]))
    pat_agg = {
        k: _nan_mean_std([r[k] for r in all_patient_rows])
        for k in patient_metric_keys
    }

    print("\n" + "=" * 78)
    print(f"MÉTRICAS A NIVEL DE PACIENTE (última época)  —  {len(all_patient_rows)} pacientes")
    print("=" * 78)
    print(f"{'patient':>8} {'fold':>4} | {'recall':>7} {'prec':>7} {'f1':>7} "
          f"{'spec':>7} {'acc':>7} | {'ict':>6} {'int':>7}")
    print("-" * 78)
    for r in all_patient_rows:
        def _fmt(x): return f"{x:>7.4f}" if not np.isnan(x) else f"{'nan':>7}"
        print(f"{r['patient']:>8} {r['fold']:>4} | {_fmt(r['recall'])} "
              f"{_fmt(r['precision'])} {_fmt(r['f1'])} {_fmt(r['specificity'])} "
              f"{_fmt(r['acc'])} | {r['n_ictal']:>6} {r['n_interictal']:>7}")
    print("-" * 78)
    print(f"{'mean':>8} {'':>4} | "
          f"{pat_agg['recall'][0]:>7.4f} {pat_agg['precision'][0]:>7.4f} "
          f"{pat_agg['f1'][0]:>7.4f} {pat_agg['specificity'][0]:>7.4f} "
          f"{pat_agg['acc'][0]:>7.4f}")
    print(f"{'std':>8} {'':>4} | "
          f"{pat_agg['recall'][1]:>7.4f} {pat_agg['precision'][1]:>7.4f} "
          f"{pat_agg['f1'][1]:>7.4f} {pat_agg['specificity'][1]:>7.4f} "
          f"{pat_agg['acc'][1]:>7.4f}")
    print("=" * 78)
    print("(media ± std a través de PACIENTES, ignorando NaN)")

    # --- CSV: métricas por época de todos los folds ---
    fields = ["fold", "epoch", "split", "loss", "acc", "f1", "recall",
              "precision", "specificity", "tp", "tn", "fp", "fn"]
    with open(OUT_METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nMétricas por época (todos los folds): {OUT_METRICS_CSV.resolve()}")

    # --- CSV: agregado POR ÉPOCA a través de folds (media ± std por época) ---
    agg_fields = ["epoch"]
    for split in ("val", "train"):
        for k in agg_keys:
            agg_fields += [f"{split}_{k}_mean", f"{split}_{k}_std"]
    with open(OUT_PER_EPOCH_AGG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(per_epoch_agg)
    print(f"Agregado por época (media ± std entre folds): "
          f"{OUT_PER_EPOCH_AGG_CSV.resolve()}")

    # --- CSV: resumen (última época por fold + mean/std) ---
    sum_fields = ["fold", "recall", "f1", "precision", "acc",
                  "specificity", "loss"]
    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        for r in per_fold:
            m = r["last"]
            writer.writerow({
                "fold": r["fold"], "recall": m["recall"], "f1": m["f1"],
                "precision": m["precision"], "acc": m["acc"],
                "specificity": m["specificity"], "loss": m["loss"],
            })
        writer.writerow({
            "fold": "mean", "recall": agg["recall"][0], "f1": agg["f1"][0],
            "precision": agg["precision"][0], "acc": agg["acc"][0],
            "specificity": agg["specificity"][0], "loss": loss_mean,
        })
        writer.writerow({
            "fold": "std", "recall": agg["recall"][1], "f1": agg["f1"][1],
            "precision": agg["precision"][1], "acc": agg["acc"][1],
            "specificity": agg["specificity"][1], "loss": loss_std,
        })
    print(f"Resumen k-fold: {OUT_SUMMARY_CSV.resolve()}")

    # --- CSV: métricas A NIVEL DE PACIENTE (una fila por paciente + mean/std) ---
    pp_fields = ["patient", "fold", "recall", "precision", "f1",
                 "specificity", "acc", "tp", "tn", "fp", "fn",
                 "n_ictal", "n_interictal"]
    with open(OUT_PER_PATIENT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pp_fields)
        writer.writeheader()
        for r in all_patient_rows:
            writer.writerow({k: r[k] for k in pp_fields})
        writer.writerow({
            "patient": "mean", "fold": "",
            "recall": pat_agg["recall"][0], "precision": pat_agg["precision"][0],
            "f1": pat_agg["f1"][0], "specificity": pat_agg["specificity"][0],
            "acc": pat_agg["acc"][0],
        })
        writer.writerow({
            "patient": "std", "fold": "",
            "recall": pat_agg["recall"][1], "precision": pat_agg["precision"][1],
            "f1": pat_agg["f1"][1], "specificity": pat_agg["specificity"][1],
            "acc": pat_agg["acc"][1],
        })
    print(f"Métricas por paciente (última época): {OUT_PER_PATIENT_CSV.resolve()}")

    # --- W&B: run de resumen con media ± std y tabla por fold ---
    try:
        summary_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}_summary",
            group=WANDB_RUN_NAME,
            job_type="summary",
            reinit=True,
            config={
                "casuistica": "raw_cnn_lstm_window",
                "task": "ictal_vs_interictal",
                "cv": "chunk_kfold",
                "split": SPLIT,
                "n_folds": N_FOLDS,
                "aggregated_from": "all_epochs+last_epoch",
                "EPOCHS": EPOCHS,
            },
        )
        table = wandb.Table(
            columns=["fold", "recall", "f1", "precision",
                     "acc", "specificity", "val_loss"]
        )
        for r in per_fold:
            m = r["last"]
            table.add_data(r["fold"], m["recall"], m["f1"], m["precision"],
                           m["acc"], m["specificity"], m["loss"])
        wandb.log({"kfold/last_epoch_per_fold": table})

        # Tabla A NIVEL DE PACIENTE (última época), todos los folds
        pat_table = wandb.Table(
            columns=["patient", "fold", "recall", "precision", "f1",
                     "specificity", "acc", "tp", "tn", "fp", "fn",
                     "n_ictal", "n_interictal"]
        )
        for r in all_patient_rows:
            pat_table.add_data(
                r["patient"], r["fold"], r["recall"], r["precision"],
                r["f1"], r["specificity"], r["acc"],
                r["tp"], r["tn"], r["fp"], r["fn"],
                r["n_ictal"], r["n_interictal"],
            )
        wandb.log({"per_patient/last_epoch": pat_table})
        for k in patient_metric_keys:
            summary_run.summary[f"per_patient/{k}_mean"] = pat_agg[k][0]
            summary_run.summary[f"per_patient/{k}_std"]  = pat_agg[k][1]
        summary_run.summary["per_patient/n_patients"] = len(all_patient_rows)

        # Curvas por época: media ± std a través de folds (step=epoch)
        for row in per_epoch_agg:
            ep = row["epoch"]
            log_row = {}
            for split in ("val", "train"):
                for k in agg_keys:
                    log_row[f"kfold_{split}/{k}_mean"] = row[f"{split}_{k}_mean"]
                    log_row[f"kfold_{split}/{k}_std"]  = row[f"{split}_{k}_std"]
            wandb.log(log_row, step=ep)

        for k in metric_keys:
            mean, std = agg[k]
            summary_run.summary[f"kfold/last_{k}_mean"] = mean
            summary_run.summary[f"kfold/last_{k}_std"]  = std
        summary_run.summary["kfold/last_loss_mean"] = loss_mean
        summary_run.summary["kfold/last_loss_std"]  = loss_std
        summary_run.summary["kfold/n_folds"]        = N_FOLDS
        # también como métrica logueada (para gráficos rápidos)
        agg_log = {f"kfold/last_{k}_mean": agg[k][0] for k in metric_keys}
        for k in metric_keys:
            agg_log[f"kfold/last_{k}_std"] = agg[k][1]
        wandb.log(agg_log, step=n_ep)

        # Subir los CSV a W&B como artifact de resultados
        try:
            csv_art = wandb.Artifact(
                name=f"{WANDB_RUN_NAME}_results_csv",
                type="results",
                metadata={"n_folds": N_FOLDS, "epochs": EPOCHS},
            )
            for p in (OUT_METRICS_CSV, OUT_PER_EPOCH_AGG_CSV, OUT_SUMMARY_CSV,
                      OUT_PER_PATIENT_CSV):
                if p.exists():
                    csv_art.add_file(str(p))
            summary_run.log_artifact(csv_art, aliases=["latest"])
            print("[wandb] CSV subidos como artifact "
                  f"'{WANDB_RUN_NAME}_results_csv'")
        except Exception as e:
            print(f"[wandb] no se pudieron subir los CSV: {e}")

        summary_run.finish()
    except Exception as e:
        print(f"[wandb] no se pudo registrar el run de resumen: {e}")

    print(f"\nDONE k-fold. RECALL última época = "
          f"{agg['recall'][0]:.4f} ± {agg['recall'][1]:.4f}")


if __name__ == "__main__":
    main()
