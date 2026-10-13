# Data

The experiments need two inputs: the CHB-MIT EEG recordings and a CSV manifest of the annotated seizures. Neither is distributed with this repository. Download them as described below, then point `DATA_ROOT` and `SEIZURES_CSV` (the `CONFIG` block at the top of every script) to their location.

## 1. CHB-MIT Scalp EEG Database (v1.0.0)

The database is available on PhysioNet: <https://physionet.org/content/chbmit/1.0.0/>

Either download the ZIP file, which unpacks to `chb-mit-scalp-eeg-database-1.0.0/`, or mirror the database with `wget`:

```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

`DATA_ROOT` must be the folder that directly contains the case folders:

```
chb-mit-scalp-eeg-database-1.0.0/
├── chb01/
│   ├── chb01_01.edf
│   ├── ...
│   └── chb01-summary.txt
├── chb02/
├── ...
└── chb24/
```

The scripts take every sub-folder whose name starts with `chb` as a patient and every `*.edf` file inside it as a recording.

**Channel selection.** The scripts keep the 22 bipolar channels of Table I of the paper, in this canonical order:

```
FP1-F7  F7-T7   T7-P7   P7-O1   FP1-F3  F3-C3   C3-P3   P3-O1
FP2-F4  F4-C4   C4-P4   P4-O2   FP2-F8  F8-T8   T8-P8   P8-O2
FZ-CZ   CZ-PZ   P7-T7   T7-FT9  FT9-FT10 FT10-T8
```

Channel names in the EDF are normalised before matching:
- extra whitespace is removed;
- a numeric suffix is stripped, so `T8-P8-0` becomes `T8-P8`;
- when the same name appears more than once, the first occurrence is used.

A recording that lacks any of the 22 channels is skipped, and the skip is logged as `[skip] <file>`.

## 2. Seizure manifest (`seizures_manifest.csv`)

The manifest has one row per annotated seizure. The required columns are:

| Column | Type | Description |
|---|---|---|
| `patient` | str | case folder name, e.g. `chb01` |
| `edf` | str | EDF file name, e.g. `chb01_03.edf` |
| `seizure_idx` | int | index of the seizure (required, but not used for labelling) |
| `start_sec` | float | seizure onset, in seconds from the start of the EDF file |
| `end_sec` | float | seizure end, in seconds from the start of the EDF file |

Example:

```csv
patient,edf,seizure_idx,start_sec,end_sec
chb01,chb01_03.edf,1,2996,3036
chb01,chb01_04.edf,1,1467,1494
```

The onset and end times come from the `Seizure Start Time` / `Seizure End Time` entries of each case's `chbXX-summary.txt` file in CHB-MIT. The scripts match rows to recordings by (`patient`, `edf`).

### How the manifest is used for labelling

Each 2 s window `[t0, t1)` of a recording is assigned one state, checked in this order (Sec. III of the paper):

1. **ictal**: the window overlaps a seizure `[start_sec, end_sec]`;
2. **postictal**: the window overlaps the 5 min after `end_sec`;
3. **preictal**: the window overlaps the 5 min before `start_sec` (SPH = 0, SOP = 5 min);
4. **interictal**: otherwise.

Only the seizures listed for the same EDF file are taken into account. Ictal windows are the positive class and interictal windows the negative class. Preictal and postictal windows are never used as targets, but they can appear as context inside a 30-window chunk.
