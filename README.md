# Subject-level, leakage-controlled multimodal machine learning in ADNI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible code for a large, **subject-level, leakage-controlled** multimodal machine-learning
benchmark on the Alzheimer's Disease Neuroimaging Initiative (ADNI): baseline diagnosis
(CN / MCI / AD), baseline-only prediction of MCI-to-AD conversion, longitudinal rate-of-change and
leakage-safe landmark analysis, an end-to-end 3D CNN, time-to-event survival, and fluid/PET
biomarkers — all evaluated under a single, strict validation framework.

> **Central finding.** Under strict subject-level, leakage-controlled validation, predictive
> performance tracks the **information content of the data, not the model class**. Inexpensive
> cognitive and genetic variables dominate; cross-sectional structural MRI and an end-to-end
> 3D CNN add no value beyond cognition; an image-based longitudinal-change CNN likewise adds
> nothing; early rates of change and molecular biomarkers contribute modest, genuine prognostic signal.

---

## ⚠️ Data availability and ethics (read first)

**This repository contains code only. It contains NO ADNI data and NO subject-level derived data.**

ADNI data are governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).
To reproduce the analyses you must obtain the data yourself from
[adni.loni.usc.edu](https://adni.loni.usc.edu/) after approval:

- **Clinical / cognitive / genetic / imaging-derived tables** via the `ADNIMERGE2` R data package
  (read locally with `pyreadr`), plus CSF/PET/plasma biomarker tables.
- **Raw T1 MRI** (for the imaging / CNN arm) via the ADNI image collections.

Subject-level tables, manifests, model out-of-fold predictions and trained weights are **excluded
by design** (see `.gitignore`) and must not be redistributed.

---

## Repository structure

```
code/            Tabular pipeline (Python): extraction → cohorts → features → diagnosis,
                 conversion, model interpretation (SHAP/region importance), longitudinal
                 rate-of-change and leakage-safe landmark change, structural-MRI change
                 decomposition, Cox survival, fluid/PET biomarkers, ComBat harmonization,
                 calibration/decision-curve analysis, nested-tuning, and a full QC audit.
                 ml_common.py is the leakage-safe, subject-level CV engine.
gpu_deep/        End-to-end 3D CNN (Python + PyTorch/MONAI, GPU): leakage-safe manifest,
                 subject-level 5-fold training, fold-aligned fusion, and an image-based
                 longitudinal-change model (baseline + follow-up CAT12 maps).
preproc/         Imaging preprocessing (MATLAB + SPM12/CAT12): CAT12 segmentation of baseline
                 and follow-up T1 scans (+ optional high-accuracy re-segmentation) and QC.
requirements.txt Python dependencies.       
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. The imaging arm additionally requires MATLAB with SPM12 + CAT12 (segmentation) and,
for the CNN, an NVIDIA GPU with a CUDA-enabled PyTorch build.

## Reproducing the analyses

**Tabular pipeline (no raw images needed — uses the ADNIMERGE2 tables).** Run the `code/` scripts
in the order listed in `REPRODUCE.md`. All modelling tables are **one row per subject**, so every stratified
split is inherently subject-level; the CV engine (`code/ml_common.py`) asserts train/test subject
disjointness.

**Imaging arm (optional; requires raw T1 and a GPU).** Segment baseline (and follow-up) scans with
`run_local/RUN_1_cat12.bat` (and `run_local/RUN_5_all.bat`), then run `gpu_deep/RUN_2_deep_pipeline.bat`
(baseline 3D CNN + fusion) and `gpu_deep/RUN_5b_longitudinal.bat` (image-based longitudinal-change CNN).

## Leakage controls (design summary)

- **Subject-level partitioning** — one row per subject; no participant in both train and test.
- **Baseline-only prognosis** — MCI-conversion models use only baseline predictors; under-observed
  MCI subjects are excluded (not mislabeled stable).
- **Leakage-safe landmark change** — early change over `[0, L]` predicts conversion in `(L, H]`.
- **Fold-aligned fusion** — deep embeddings are out-of-fold; fusion is fit on outer-train only.
- **End-to-end control** — a label-permutation test collapses performance to chance.

## Citation

> Dündar, M. S. (2026). *Subject-level, leakage-controlled multimodal machine learning for
> Alzheimer's disease diagnosis and prediction of conversion from mild cognitive impairment.* Manuscript under review.

*(Update with DOI/journal once available.)*

## License

Code is released under the [MIT License](LICENSE). This license covers **the code only**; ADNI data
remain governed by the ADNI Data Use Agreement and are not redistributed here.

## Author

**Mehmet Sait Dündar**, Erciyes University, Halil Bayraktar Health Services Vocational School,
Department of Medical Imaging Techniques, Kayseri, Türkiye.
ORCID: [0000-0002-0336-4825](https://orcid.org/0000-0002-0336-4825).

## Acknowledgement

Data used in preparation of this work were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) database (adni.loni.usc.edu). The ADNI investigators contributed to the design
and implementation of ADNI and/or provided data but did not participate in the analysis or writing
of this work. A complete listing of ADNI investigators is available at the ADNI website.
