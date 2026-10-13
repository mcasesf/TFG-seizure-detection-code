# -*- coding: utf-8 -*-
"""
S3 — raw + Transformer  ·  protocolo k-fold by patient (leave-subjects-out)

Código del TFG "A Comparative Study of Feature Extractors for Epileptic
Seizure Detection" (Marc Cases Font, UAB / CVC). Las referencias a secciones,
ecuaciones y tablas son las del paper (docs/TFG_MarcCases.pdf).

Este script entrena y valida la estrategia S3 (Sec. VI-F) con el protocolo
k-fold by patient (Sec. VII-A.c) y produce:
  - la fila S3 de la Tabla VII (media ± std entre folds), y
  - las columnas S3 de la Tabla XIb (métricas por paciente).

Terminología (paper  <->  código)
  window : segmento de L = 2 s sin solape        <->  "chunk": CHUNK_SEC, tchunk,
           (22 canales × 512 muestras, 256 Hz)         chunkify()
  chunk  : N = 30 windows consecutivas (60 s);   <->  "ventana": WINDOW,
           su etiqueta es la de su última window       RawWindowDataset
  stride del chunk (ictal = 1, interictal = N)   <->  STRIDE_WIN,
                                                       STRIDE_WIN_INTERICTAL
  estrategia SN                                  <->  "casuisticaN" en los nombres
                                                       de salida y de W&B
  Los identificadores, los mensajes de log y los nombres de salida se
  mantienen tal y como se ejecutaron los experimentos; los comentarios usan
  la terminología del paper.

Pipeline compartido (Sec. VI-A, Fig. 1)

  chunk: N = 30 windows X_i ∈ R^(22×512)
      │
      ▼  g(·) aplicada a cada window   <- lo único que cambia entre S1–S4
         canales como tokens + CLS (dim 512) -> Linear(512 -> 128)     (Eq. 7)
         -> Transformer encoder (4 capas pre-norm, 4 heads, FFN 512,
            GELU, sin channel embedding) -> salida del CLS -> v_i ∈ R^128
      │
      ▼  secuencia [v_(i-29), ..., v_i]  (B, N, d_window)
  LSTM(input = d_window, hidden = 256, 2 capas, dropout 0.3)
      │
      ▼  último hidden state h_T (B, 256)
  head: LayerNorm -> Dropout(0.3) -> Linear(256 -> 1)                (Eq. 4)
      │
      ▼  logit -> p_i = P(la última window del chunk es ictal)        (Eq. 3)

Estrategia S3 — raw + Transformer (Sec. VI-F)
  Los 22 canales de cada window son los tokens (vectores de 512 muestras) de
  un Transformer encoder con token CLS; la self-attention mezcla los canales
  entre sí. Sin channel embedding: los canales se tratan como un conjunto sin
  orden. La representación de la window es la salida del CLS (en el código se
  le aplica además una LayerNorm final, norm_out). Transformer, LSTM y head se
  entrenan end-to-end, sin pre-entrenamiento.

Datos y etiquetado (Sec. III–V)
  - CHB-MIT, 22 canales bipolares (Tabla I); un EDF al que le falte alguno se
    descarta. z-score por canal sobre TODA la grabación (Eq. 2).
  - Estados: ictal, preictal (SPH = 0, SOP = 5 min), postictal (5 min) e
    interictal. Tarea binaria ictal (1) vs interictal (0): un chunk entra en
    el dataset solo si su última window es ictal o interictal; preictal y
    postictal solo aparecen como contexto dentro de un chunk.

Protocolo de validación: k-fold by patient / leave-subjects-out (Sec. VII-A.c)
  - Los 24 pacientes (carpetas chbXX) se barajan con SEED y se reparten en
    round-robin en K = 5 grupos; en cada fold un grupo es la validación y el
    resto el train. Los datasets de train y validación se construyen dentro
    de cada fold.
  - Es el protocolo más exigente: se evalúa sobre pacientes no vistos en
    entrenamiento.
  - Por paciente: cada paciente se valida en un único fold.

Balanceo de clases (Sec. VII-A)
  - TRAIN: los chunks interictales se submuestrean a 1:1 (con SEED).
  - VALIDACIÓN: sin rebalancear (todos los chunks ictales y todos los
    interictales sin solape), de modo que el interictal sigue siendo mayoría.

Entrenamiento y métricas (Tabla II, Sec. VII-C)
  - Adam (LR = 1e-3, weight decay = 1e-3), BCEWithLogitsLoss con
    pos_weight = n_interictal / n_ictal del train (≈ 1), batch 128, grad
    clipping 1.0, sin scheduler, 30 épocas. Mismo init (SEED) en todos los
    folds. Se guarda el checkpoint de la ÚLTIMA época.
  - Métricas de la ÚLTIMA época (30), no las del mejor checkpoint: recall,
    precision, specificity, F1 (y acc), agregadas como media ± std (ddof=1)
    entre los K = 5 folds.
  - Por paciente: las mismas métricas, resumidas como media ± std entre
    pacientes ignorando los valores indefinidos (NaN). Ver la nota de
    metrics_from_counts sobre F1.
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

# --- Protocolo de validación (Sec. VII-A del paper) ---
# Este script usa el k-fold BY PATIENT / leave-subjects-out (Sec. VII-A.c):
# la unidad de partición es el paciente (carpeta chbXX).
# SPLIT se añade al nombre del run de W&B y al de los ficheros/carpeta de
# salida, para no colisionar con los runs ni los ficheros de los otros
# protocolos.
SPLIT = "patient"

CKPT_DIR  = Path(f"checkpoints_raw_channel_transformer_casuistica3_ictal_{SPLIT}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def last_path_for_fold(fold_idx: int) -> Path:
    """Checkpoint de la ÚLTIMA época de cada fold (namespaced por fold)."""
    return CKPT_DIR / f"last_fold{fold_idx}.pt"

# Métricas por época de TODOS los folds (incluye columna 'fold')
OUT_METRICS_CSV = Path(f"eval_raw_channel_transformer_per_epoch_casuistica3_ictal_kfold_{SPLIT}.csv")
# Agregado POR ÉPOCA a través de folds: media ± std en cada época (curva)
OUT_PER_EPOCH_AGG_CSV = Path(f"eval_raw_channel_transformer_per_epoch_agg_casuistica3_ictal_kfold_{SPLIT}.csv")
# Resumen: última época de cada fold + media ± std a través de folds
# (fila de esta estrategia en la Tabla VII del paper)
OUT_SUMMARY_CSV = Path(f"eval_raw_channel_transformer_kfold_summary_casuistica3_ictal_{SPLIT}.csv")
# Métricas a NIVEL DE PACIENTE (última época): una fila por paciente +
# media ± std a través de pacientes (Tabla XI del paper). Cada paciente sale
# del único fold en el que estuvo en validación (by patient -> cada paciente
# se valida 1 vez).
OUT_PER_PATIENT_CSV = Path(f"eval_raw_channel_transformer_per_patient_casuistica3_ictal_kfold_{SPLIT}.csv")

# --- Windows (Sec. IV-A del paper) ---
# OJO terminología: en el código "chunk" (CHUNK_SEC, tchunk, chunkify) es la
# WINDOW del paper: segmento de L = 2 s sin solape (22 × 512 muestras a 256 Hz).
CHUNK_SEC  = 2.0         # L: duración de cada window (s)
STRIDE_SEC = 2.0         # stride entre windows (= L -> sin solape)
C          = 22          # número de canales (Tabla I)

# --- Chunks (Sec. IV-A del paper) ---
# OJO terminología: en el código "WINDOW"/"ventana" es el CHUNK del paper:
# N = 30 windows consecutivas (60 s de contexto). La etiqueta de un chunk es
# la de su última window.
WINDOW     = 30          # N: windows por chunk (30 × 2 s = 60 s de contexto)
STRIDE_WIN = 1           # stride (en windows) de los chunks ICTALES
                         # (stride 1 -> se capturan todos los chunks cuya
                         # última window es ictal; la clase positiva es escasa
                         # y conviene muestrearla densamente). Chunks ictales
                         # consecutivos comparten 29 de sus 30 windows.
STRIDE_WIN_INTERICTAL = WINDOW   # = N = 30 -> stride (en windows) de los
                         # chunks INTERICTALES. Al ser igual a N, los chunks
                         # interictales NO se solapan (cada uno empieza donde
                         # acaba el anterior: 30 windows = 60 s más tarde).
                         # Reduce ~30x el nº de chunks interictales tanto en
                         # train como en validación.

# --- Etiquetado de crisis (Sec. III del paper) ---
# Buffer preictal = [onset - (SPH + SOP), onset - SPH] (Eq. 1) y buffer
# postictal de POSTICTAL_MIN tras el final de la crisis; ambos se excluyen
# como objetivo de clasificación.
SPH_MIN       = 0
SOP_MIN       = 5
POSTICTAL_MIN = 5

# --- Balanceo de clases (Sec. VII-A del paper) ---
INTERICTAL_RATIO     = 1     # TRAIN: chunks interictales submuestreados a 1:1
VAL_INTERICTAL_RATIO = None  # VALIDACIÓN sin rebalancear: se conservan todos
                             # los chunks ictales y todos los interictales (sin
                             # solape), así que el interictal sigue siendo
                             # mayoría en validación

# --- Transformer inter-canal (Sec. VI-F del paper) ---
# Cada canal es un token de T muestras. Se prepende CLS.
# Proyección lineal T -> D_MODEL antes del encoder.
# Sin channel embedding: los canales se tratan como conjunto desordenado.
D_MODEL       = 128
N_HEADS       = 4
N_TF_LAYERS   = 4
DIM_FFN       = 512
TF_ACTIVATION = "gelu"
TF_DROPOUT    = 0.1
# d_window (vector por window que entra a la LSTM) = D_MODEL = 128

# --- Clasificador compartido: LSTM + head (Sec. VI-B, Tabla II) ---
HIDDEN   = 256
N_LAYERS = 2
DROPOUT  = 0.3       # entre capas de la LSTM y en el head

# --- Entrenamiento (Tabla II del paper) ---
BATCH_SIZE   = 128       # chunks por batch (igual en S1–S4)
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-3
# Nota: sin LR scheduler. Con 30 épocas y Adam no aporta, y un
# ReduceLROnPlateau sobre train casi nunca se dispararía (la loss de train
# baja de forma monótona). LR constante = LR durante todo el entrenamiento.

# --- Verbosidad ---
PRINT_EVERY_N_BATCHES = 10   # progreso dentro de cada epoch (None = silencio)

# --- K-fold (Sec. VII-A.c del paper: by patient / leave-subjects-out) ---
# Los pacientes (carpetas chbXX) se barajan con SEED y se reparten en
# round-robin en N_FOLDS grupos. Cada grupo es la validación exactamente una
# vez; el resto de pacientes son train, así que ningún paciente (carpeta
# chbXX) aparece a la vez en train y validación de un mismo fold.
N_FOLDS              = 5     # K = 5
SEED                 = 123   # baraja de pacientes + submuestreo de train +
                             # init del modelo (igual en todos los folds)
VAL_SEED             = 124   # solo actuaría si VAL_INTERICTAL_RATIO no fuera
                             # None (la validación no se submuestrea)

PRELOAD_EDF = True

# --- W&B ---
WANDB_PROJECT  = "TFG-seizure-embeddings"
WANDB_RUN_NAME = f"raw_channel_transformer_casuistica3_ictal_kfold_{SPLIT}_intstride30"

# 22 canales bipolares seleccionados (Tabla I del paper), en orden canónico
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
    Devuelve X (22, T) float32 con z-score por canal sobre TODA la grabación
    (Sec. V-b, Eq. 2), aplicado antes de trocear en windows. Si falta algún
    canal canónico se lanza ValueError y el EDF se descarta (Sec. V-a).
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
    """Trocea X (C, T_total) en windows de L = CHUNK_SEC s: (n_windows, C, tchunk)."""
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
    """Longitud de la window en muestras (L·fs = 512), leyendo el primer EDF disponible."""
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
    """Estado de una window [t0, t1) (Sec. III-B): ictal / postictal / preictal / interictal."""
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
    Devuelve (labels_bin, labels_keep), uno por window:
      labels_bin  : 1 si ictal, 0 en cualquier otro caso.
      labels_keep : True solo para ictal e interictal (las windows
                    válidas como objetivo de clasificación).
                    preictal/postictal -> False.
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
# DATASET  —  chunks de N windows (ictal / interictal)
# ============================================================
def patient_kfold_splits(root, n_folds, seed):
    """
    K-fold BY PATIENT / leave-subjects-out (Sec. VII-A.c del paper): la
    unidad de partición es el paciente (carpeta chbXX).

    Baraja los pacientes con `seed` y los reparte en `n_folds` grupos lo más
    equilibrados posible. Para cada fold k, ese grupo es la VALIDACIÓN y el
    resto de pacientes son TRAIN. Garantiza que ningún paciente esté a la vez
    en train y val de un mismo fold (sin fuga de datos entre pacientes).

    Yields, por cada fold:
        (fold_idx, train_edfs, val_edfs, train_patients, val_patients)
    con fold_idx en 1..n_folds.
    """
    pat_dirs = [p for p in sorted(root.iterdir())
                if p.is_dir() and p.name.startswith("chb")]
    if len(pat_dirs) < n_folds:
        raise RuntimeError(
            f"Hay {len(pat_dirs)} pacientes pero pediste {n_folds} folds.")

    rng = random.Random(seed)
    rng.shuffle(pat_dirs)

    # reparto round-robin -> grupos lo más balanceados posible en nº de pacientes
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(pat_dirs):
        folds[i % n_folds].append(d)

    print(f"K-fold por paciente: {len(pat_dirs)} pacientes -> {n_folds} folds "
          f"(seed={seed})")
    for k, grp in enumerate(folds, 1):
        print(f"  fold {k}: val patients ({len(grp)}) = {[d.name for d in grp]}")

    for k in range(n_folds):
        val_dirs   = folds[k]
        train_dirs = [d for j, g in enumerate(folds) if j != k for d in g]
        train_edfs = [e for d in train_dirs for e in sorted(d.glob("*.edf"))]
        val_edfs   = [e for d in val_dirs   for e in sorted(d.glob("*.edf"))]
        train_pats = [d.name for d in train_dirs]
        val_pats   = [d.name for d in val_dirs]
        yield k + 1, train_edfs, val_edfs, train_pats, val_pats


class RawWindowDataset(Dataset):
    """
    Construye los CHUNKS del paper: secuencias de N = WINDOW windows
    consecutivas de un mismo recording.

    Cada sample (= un chunk): (X_seq, label, patient_idx)
      X_seq       : (WINDOW, C, T)  las N windows del chunk con sus canales intactos
                                    El Transformer trata los C canales como
                                    tokens en el forward
      label       : etiqueta de la ÚLTIMA window del chunk
                    0 interictal / 1 ictal
      patient_idx : índice entero del paciente al que pertenece el chunk
                    (se mapea con self.idx_to_patient). Permite calcular
                    métricas por paciente en validación.

    El chunk se desliza con DOS strides distintos según la clase de su
    última window (que es la que determina su etiqueta):
      - chunks ICTALES: stride STRIDE_WIN (=1), muestreo denso.
      - chunks INTERICTALES: stride STRIDE_WIN_INTERICTAL (=N=30),
        de forma que los chunks interictales NO se solapan.
    Un chunk se incluye solo si su última window es ictal o interictal
    (no preictal/postictal). Las 29 windows anteriores pueden ser de
    cualquier fase — son el contexto temporal de la LSTM.

    Submuestreo: si interictal_ratio no es None, se conservan todos los
    chunks ictales y se submuestrean los interictales al ratio dado
    (TRAIN, con SEED, para balancear 1:1). Si interictal_ratio es None NO se
    submuestrea nada: se conserva todo el interictal (validación sin
    rebalancear, o el pool completo en by chunk / by recording).
    """
    def __init__(self, edf_files, seizures_df, interictal_ratio=None, seed=123):
        # cada elemento es (seq, patient_name) para arrastrar el paciente
        ict_samples: list[tuple[np.ndarray, str]] = []
        int_samples: list[tuple[np.ndarray, str]] = []

        for edf in edf_files:
            try:
                X, sf  = load_edf_as_X22(edf, preload=PRELOAD_EDF)
                chunks = chunkify(X, sf)                   # (N, C, T), N = nº de windows
                N, Cc, T = chunks.shape
                if N < WINDOW:
                    continue

                patient = edf.parent.name
                rows = seizures_df[
                    (seizures_df["patient"] == patient) &
                    (seizures_df["edf"]     == edf.name)
                ].copy()
                labels_bin, labels_keep = label_array_for_chunks(N, rows)

                # Construir los chunks: N windows consecutivas, guardando
                # (C, T) por window, sin aplanar. La etiqueta de cada chunk es
                # la de su ÚLTIMA window. Se usan DOS strides distintos según
                # la clase de esa última window:
                #
                #   ICTAL      -> stride STRIDE_WIN (=1): muestreo denso, se
                #                 capturan todos los chunks cuya última window
                #                 es ictal (clase positiva, escasa).
                #   INTERICTAL -> stride STRIDE_WIN_INTERICTAL (=N=30):
                #                 chunks interictales SIN solape (cada uno
                #                 60 s después del anterior). Reduce ~30x el
                #                 nº de chunks interictales.
                #
                # Son dos pasadas independientes: la pasada ictal solo añade
                # chunks ictales y la interictal solo añade interictales, así
                # que ningún chunk se cuenta dos veces.

                # --- pasada ICTAL (stride 1) ---
                for end in range(WINDOW, N + 1, STRIDE_WIN):
                    last = end - 1            # índice de la última window del chunk
                    # labels_bin[last]==1 implica ictal (que siempre es 'keep')
                    if labels_bin[last] == 1:
                        seq = chunks[end - WINDOW:end]   # (WINDOW, C, T) = un chunk
                        ict_samples.append((seq, patient))

                # --- pasada INTERICTAL (stride 30, sin solape) ---
                for end in range(WINDOW, N + 1, STRIDE_WIN_INTERICTAL):
                    last = end - 1
                    # interictal = keep (ictal/interictal) y NO ictal.
                    # El check de keep excluye preictal/postictal.
                    if labels_keep[last] and labels_bin[last] == 0:
                        seq = chunks[end - WINDOW:end]   # (WINDOW, C, T) = un chunk
                        int_samples.append((seq, patient))

            except Exception as e:
                print(f"  [skip] {edf.name}: {type(e).__name__}: {e}")

        n_ict_total = len(ict_samples)
        n_int_total = len(int_samples)
        print(f"  cargadas: ictal={n_ict_total}  interictal={n_int_total}  "
              f"ratio_natural={n_int_total / max(1, n_ict_total):.1f}x")

        # submuestreo del interictal (solo si interictal_ratio no es None;
        # con None -> NO se submuestrea: validación y pool completo)
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
        # seq: (WINDOW, C, T) = un chunk de N windows; el Transformer trata los
        # C canales de cada window como tokens en el forward
        X = torch.from_numpy(np.ascontiguousarray(seq)).float()
        y = torch.tensor(label, dtype=torch.float32)
        p = torch.tensor(pidx, dtype=torch.long)
        return X, y, p


# ============================================================
# MODELO  —  S3 (raw + Transformer): Transformer inter-canal + LSTM compartida
# ============================================================
class ChannelTransformer(nn.Module):
    """
    Transformer encoder que mezcla los C canales de una window vía
    self-attention. Los canales son los tokens; el tiempo es la dim
    de cada token.

    Entrada: (B*, C, T)  una window con sus C canales
    Salida:  (B*, D_MODEL) representación de la window v_i (token CLS)

    Pipeline:
      (B*, C, T)
        ↓  prepend CLS (aprendible, dim T)        -> (B*, C+1, T)
        ↓  Linear(T -> D_MODEL) por token         -> (B*, C+1, D_MODEL)
        ↓  Transformer encoder pre-norm           -> (B*, C+1, D_MODEL)
        ↓  tomar token CLS (posición 0)           -> (B*, D_MODEL)
        ↓  LayerNorm final (norm_out)             -> (B*, D_MODEL)

    Sin channel embedding: los canales se tratan como un conjunto sin
    orden. La identidad del canal no se codifica porque el orden de la
    lista canónica no representa topología real del montaje bipolar.

    Nota: el paper (Sec. VI-F) define v_i = h_CLS; el código aplica además
    una LayerNorm final (norm_out) sobre la salida del CLS.
    """
    def __init__(self, tchunk, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_TF_LAYERS, dim_ffn=DIM_FFN,
                 dropout=TF_DROPOUT, activation=TF_ACTIVATION):
        super().__init__()
        self.tchunk  = int(tchunk)
        self.d_model = d_model

        # CLS token aprendible con la dim del canal raw (T)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.tchunk))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Proyección lineal de T -> D_MODEL, compartida por todos los tokens
        self.token_proj = nn.Linear(self.tchunk, d_model)

        # Transformer encoder pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_ffn,
            dropout         = dropout,
            activation      = activation,
            batch_first     = True,
            norm_first      = True,   # pre-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm final aplicado al token CLS
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B*, C, T)  -> una window, canales como tokens
        return: (B*, D_MODEL)  representación de la window vía CLS
        """
        Bx = x.shape[0]
        cls = self.cls_token.expand(Bx, -1, -1)        # (B*, 1, T)
        tokens = torch.cat([cls, x], dim=1)             # (B*, C+1, T)
        tokens = self.token_proj(tokens)                # (B*, C+1, D_MODEL)
        tokens = self.encoder(tokens)                   # (B*, C+1, D_MODEL)
        cls_out = tokens[:, 0]                          # (B*, D_MODEL)
        return self.norm_out(cls_out)


class RawChannelTransformerWindowLSTM(nn.Module):
    """
    Pipeline completo:

      X_seq (B, WINDOW, C, T)
        ↓  para cada window del chunk  (g(·)):
           reshape -> (B*WINDOW, C, T)
           ChannelTransformer (CLS + proj + encoder) -> (B*WINDOW, D_MODEL)  <- v_i
        ↓
        reshape -> (B, WINDOW, D_MODEL)               <- secuencia para la LSTM
        LSTM(input=D_MODEL, hidden=HIDDEN)
        ↓
        último hidden state (B, HIDDEN)
        head(LayerNorm + Dropout + Linear)
        ↓
        logit (B,)   de la ÚLTIMA window del chunk
    """
    def __init__(self, tchunk, d_model=D_MODEL, n_heads=N_HEADS,
                 n_tf_layers=N_TF_LAYERS, dim_ffn=DIM_FFN,
                 tf_dropout=TF_DROPOUT,
                 hidden_size=HIDDEN, num_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.channel_tf = ChannelTransformer(
            tchunk     = tchunk,
            d_model    = d_model,
            n_heads    = n_heads,
            n_layers   = n_tf_layers,
            dim_ffn    = dim_ffn,
            dropout    = tf_dropout,
            activation = TF_ACTIVATION,
        )
        self.lstm = nn.LSTM(
            input_size  = d_model,
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

        # g(·): aplicar el Transformer inter-canal a todas las windows a la vez
        x = X_seq.reshape(B * W, Cc, T)            # (B*WINDOW, C, T)
        f = self.channel_tf(x)                     # (B*WINDOW, D_MODEL)
        f = f.reshape(B, W, -1)                    # (B, WINDOW, D_MODEL)

        _, (hn, _) = self.lstm(f)                  # hn: (layers, B, HIDDEN)
        h = hn[-1]                                 # (B, HIDDEN)
        return self.head(h).squeeze(-1)            # (B,)


# ============================================================
# MÉTRICAS  (Eq. 8 del paper)  —  idénticas en S1–S4
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
    para ese paciente, o recall si el paciente no tiene chunks ictales).

    Nota para reproducir las tablas por paciente del paper (IX–XI): aquí F1
    sale NaN siempre que TP = 0, y _nan_mean_std lo excluye de la media. En
    las tablas del paper, los pacientes con P = R = 0 aparecen con F1 = 0 y
    cuentan en la media ± std entre pacientes.
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
def run_one_fold(fold_idx, n_folds, train_edfs, val_edfs,
                 train_pats, val_pats, seizures_df, tchunk, sf0, device):
    """
    Entrena un fold completo (EPOCHS épocas) y devuelve un dict con:
        - 'last'        : métricas de val de la ÚLTIMA época (lo que se agrega)
        - 'per_patient' : lista de métricas por paciente (última época) de los
                          pacientes que estaban en VALIDACIÓN en este fold
        - 'csv_rows'    : filas por época (con columna 'fold') para el CSV global
        - 'val_history' / 'train_history' : métricas por época
        - 'n_int'/'n_ict' : conteos de train

    El checkpoint guardado es el de la ÚLTIMA época (no el de mejor F1).
    """
    # Mismo init de modelo y orden de submuestreo/baraja en todos los folds
    seed_everything(SEED)

    tag = f"FOLD {fold_idx}/{n_folds}"
    print("\n" + "#" * 64)
    print(f"# {tag}")
    print(f"#   val patients   ({len(val_pats)}): {val_pats}")
    print(f"#   train patients ({len(train_pats)}): {train_pats}")
    print(f"#   EDFs: train={len(train_edfs)}  val={len(val_edfs)}")
    print("#" * 64)

    # --- Datasets ---
    print(f"\n[{tag}] Construyendo dataset de train...")
    train_ds = RawWindowDataset(train_edfs, seizures_df,
                                interictal_ratio=INTERICTAL_RATIO,
                                seed=SEED)
    print(f"[{tag}] Construyendo dataset de val...")
    val_ds   = RawWindowDataset(val_edfs, seizures_df,
                                interictal_ratio=VAL_INTERICTAL_RATIO,
                                seed=VAL_SEED)
    print(f"[{tag}] Train sequences: {len(train_ds)}   Val sequences: {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"[{tag}] Dataset vacío (train={len(train_ds)} val={len(val_ds)}): "
            f"revisa etiquetas / rutas / reparto de pacientes.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # --- pos_weight = n_interictal / n_ictal (≈ 1: el train ya está balanceado 1:1) ---
    n0, n1 = train_ds.n_int, train_ds.n_ict
    print(f"[{tag}] Train sequences  ->  interictal: {n0}  ictal: {n1}  "
          f"ratio: {n0/max(1,n1):.1f}x")
    pos_weight = torch.tensor([n0 / max(1, n1)], dtype=torch.float32,
                              device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Modelo (init idéntico en todos los folds por seed_everything(SEED)) ---
    model = RawChannelTransformerWindowLSTM(
        tchunk       = tchunk,
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_tf_layers  = N_TF_LAYERS,
        dim_ffn      = DIM_FFN,
        tf_dropout   = TF_DROPOUT,
        hidden_size  = HIDDEN,
        num_layers   = N_LAYERS,
        dropout      = DROPOUT,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tf    = sum(p.numel() for p in model.channel_tf.parameters() if p.requires_grad)
    n_lstm  = sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
    n_head  = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"\n[{tag}] Modelo RawChannelTransformerWindowLSTM")
    print(f"  Transformer inter-canal: tokens=canales+CLS ({C+1} tokens)")
    print(f"                          d_model={D_MODEL}  heads={N_HEADS}  "
          f"layers={N_TF_LAYERS}  ffn={DIM_FFN}  dropout={TF_DROPOUT}")
    print(f"  repr. por chunk        : token CLS -> {D_MODEL} dims")
    print(f"  secuencia LSTM         : {WINDOW} pasos (un chunk por paso)")
    print(f"  parámetros             : {n_train:,}  "
          f"(TF={n_tf:,}  LSTM={n_lstm:,}  head={n_head:,})")

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
            "casuistica":    "raw_channel_transformer_lstm_window",
            "task":          "ictal_vs_interictal",
            "sequence":      "window_chunks",
            "cv":            "patient_group_kfold",
            "split":         SPLIT,
            "n_folds":       n_folds,
            "fold":          fold_idx,
            "val_patients":  val_pats,
            "train_patients": train_pats,
            "C": C, "tchunk": tchunk, "sfreq": sf0,
            "window":        WINDOW,
            "stride_win":    STRIDE_WIN,
            "stride_win_interictal": STRIDE_WIN_INTERICTAL,
            "d_model":       D_MODEL,
            "n_heads":       N_HEADS,
            "n_tf_layers":   N_TF_LAYERS,
            "dim_ffn":       DIM_FFN,
            "tf_dropout":    TF_DROPOUT,
            "tf_activation": TF_ACTIVATION,
            "tf_norm_first": True,
            "channel_embedding": "none",
            "n_tokens":      C + 1,
            "lstm_input":    D_MODEL,
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
            "n_params_tf":   n_tf,
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
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS,
            "N_TF_LAYERS": N_TF_LAYERS, "DIM_FFN": DIM_FFN,
            "TF_DROPOUT": TF_DROPOUT,
            "HIDDEN": HIDDEN,
            "N_LAYERS": N_LAYERS, "DROPOUT": DROPOUT,
            "task": "ictal_vs_interictal",
            "casuistica": "raw_channel_transformer_lstm_window",
            "cv": "patient_group_kfold",
            "n_folds": n_folds,
            "checkpoint": "last_epoch",
        },
    }, last_path)
    try:
        artifact = wandb.Artifact(
            name=f"raw-channel-transformer-casuistica3-ictal-{SPLIT}-fold{fold_idx}-last",
            type="model",
            metadata={"val_f1": float(last_val_m["f1"]),
                      "epoch": int(EPOCHS), "fold": int(fold_idx)},
        )
        artifact.add_file(str(last_path))
        run.log_artifact(artifact, aliases=["last"])
    except Exception as e:
        print(f"[wandb] no se pudo registrar el artifact: {e}")

    # --- Métricas POR PACIENTE de la última época (pacientes en validación) ---
    per_patient_rows = []
    if last_pp_counts is not None:
        tp_a = last_pp_counts["tp"]; tn_a = last_pp_counts["tn"]
        fp_a = last_pp_counts["fp"]; fn_a = last_pp_counts["fn"]
        for pidx, pat in enumerate(val_ds.idx_to_patient):
            tp, tn = int(tp_a[pidx]), int(tn_a[pidx])
            fp, fn = int(fp_a[pidx]), int(fn_a[pidx])
            if (tp + tn + fp + fn) == 0:
                continue   # paciente sin muestras válidas en val
            m = metrics_from_counts(tp, tn, fp, fn)
            per_patient_rows.append({
                "patient":      pat,
                "fold":         fold_idx,
                "recall":       m["recall"],
                "precision":    m["precision"],
                "f1":           m["f1"],
                "specificity":  m["specificity"],
                "acc":          m["acc"],
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "n_ictal":       tp + fn,    # positivos reales del paciente
                "n_interictal":  tn + fp,    # negativos reales del paciente
            })

    # --- Resumen del fold en el summary del run ---
    run.summary["last/val_recall"]      = float(last_val_m["recall"])
    run.summary["last/val_f1"]          = float(last_val_m["f1"])
    run.summary["last/val_precision"]   = float(last_val_m["precision"])
    run.summary["last/val_acc"]         = float(last_val_m["acc"])
    run.summary["last/val_specificity"] = float(last_val_m["specificity"])
    run.summary["last/val_loss"]        = float(last_val_m["loss"])

    # --- W&B: métricas POR PACIENTE de este fold (última época) ---
    if per_patient_rows:
        try:
            pat_table = wandb.Table(
                columns=["patient", "recall", "precision", "f1",
                         "specificity", "acc", "tp", "tn", "fp", "fn",
                         "n_ictal", "n_interictal"]
            )
            for r in per_patient_rows:
                pat_table.add_data(
                    r["patient"], r["recall"], r["precision"], r["f1"],
                    r["specificity"], r["acc"], r["tp"], r["tn"], r["fp"],
                    r["fn"], r["n_ictal"], r["n_interictal"],
                )
            wandb.log({"per_patient/last_epoch": pat_table})
            for r in per_patient_rows:
                p = r["patient"]
                run.summary[f"per_patient/{p}/recall"]      = r["recall"]
                run.summary[f"per_patient/{p}/precision"]   = r["precision"]
                run.summary[f"per_patient/{p}/f1"]          = r["f1"]
                run.summary[f"per_patient/{p}/specificity"] = r["specificity"]
                run.summary[f"per_patient/{p}/acc"]         = r["acc"]
        except Exception as e:
            print(f"[wandb] no se pudo registrar la tabla por paciente: {e}")

    run.finish()

    print(f"\n[{tag}] DONE.  last_val_recall={last_val_m['recall']:.4f}  "
          f"last_val_f1={last_val_m['f1']:.4f}")
    print(f"[{tag}] Checkpoint (última época): {last_path.resolve()}")
    if per_patient_rows:
        print(f"[{tag}] Métricas por paciente (última época):")
        for r in per_patient_rows:
            print(f"    {r['patient']:>8}  recall={r['recall']:.4f}  "
                  f"prec={r['precision']:.4f}  f1={r['f1']:.4f}  "
                  f"spec={r['specificity']:.4f}  "
                  f"(ict={r['n_ictal']} int={r['n_interictal']})")

    return {
        "fold":          fold_idx,
        "last":          last_val_m,
        "per_patient":   per_patient_rows,
        "csv_rows":      csv_rows,
        "val_history":   val_history,
        "train_history": train_history,
        "n_int":         n0,
        "n_ict":         n1,
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
    print("CASUÍSTICA 3:  raw -> Transformer inter-canal -> LSTM (ventana)")
    print(f"            K-FOLD por paciente | {N_FOLDS} folds | ictal vs interictal")
    print("=" * 60)

    # --- tchunk: muestras por window (L·fs = 512) ---
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

    for fold_idx, train_edfs, val_edfs, train_pats, val_pats in \
            patient_kfold_splits(DATA_ROOT, N_FOLDS, SEED):
        res = run_one_fold(
            fold_idx, N_FOLDS, train_edfs, val_edfs,
            train_pats, val_pats, seizures_df, tchunk, sf0, device,
        )
        per_fold.append(res)
        all_csv_rows.extend(res["csv_rows"])
        all_patient_rows.extend(res["per_patient"])

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

    # --- Agregación A NIVEL DE PACIENTE (última época) ---
    # Cada paciente aparece una sola vez (k-fold por paciente -> validado 1 vez).
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
                "casuistica": "raw_channel_transformer_lstm_window",
                "task": "ictal_vs_interictal",
                "cv": "patient_group_kfold",
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
