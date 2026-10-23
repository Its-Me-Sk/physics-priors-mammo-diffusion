# Physics priors in a conditional diffusion model for mammography

Code, pre-registrations, run reports and figures for the paper

> **Asef Ahmed Shimanto.** *Physics priors do not improve a clinical-scale conditional diffusion model for mammography:
> replicated negative results, mechanisms, and where physics does help.* Manuscript in preparation (2026).

The paper tests whether physics knowledge improves a class-conditional denoising diffusion model of mammograms:
a signal-dependent X-ray noise-level-function (NLF) loss, a score Fokker–Planck (FP) residual, FP-derived regularisers,
physics-based sample scoring, training-free fixes and an acquisition-noise layer. Each was judged against a decision
rule fixed before its run; the main comparisons were then replicated under a pre-registration (S8) on a second dataset
with a held-out test stage. A second pre-registered study (S11) tests physics-based low-dose simulation as **training
augmentation** for a benign/malignant classifier.

## Base model — prior work, not part of this contribution
The generator architecture and training recipe are **U24_synthesis** by Linardos, Thakur, Gastounioti, Makris and Bakas,
*"Evaluating the utility of AI-generated mammography images in breast cancer analysis"*, Scientific Reports 16:29981 (2026),
https://doi.org/10.1038/s41598-026-58260-0, code: https://github.com/Linardos/U24_synthesis.
Their original model was trained on EMBED. **Here the model was retrained from scratch on CBIS-DDSM and on CMMD; no
pretrained weights from the original authors were used.** The U24_synthesis code is *not* redistributed in this repository:
each notebook clones it at run time from the fork https://github.com/Its-Me-Sk/U24_synthesis (forked from
https://github.com/IUCompPath/U24_synthesis; the fork contains one additional commit, visible in its commit history) and
then applies two small compatibility patches shown in Cell 1 (removing `xformers` from the requirements and fixing a
hard-coded config path). All physics terms, evaluation code and data handling of this study are in this repository.

## Repository layout
| folder | contents |
|---|---|
| `notebooks/main/` | the Kaggle notebooks exactly as run for the main study, numbered in run order; each is self-contained (its code is embedded in Cell 1 and written to disk at run time) |
| `notebooks/preliminary_leaky_split/` | Experiments 1–5 (Appendix A of the paper) — run on the ORIGINAL CBIS-DDSM split, which was later found to leak patients; not part of the main claims |
| `notebooks/registered_not_run/` | S10 — pre-registered but not run (see Appendix B) |
| `src/` | readable copies of the Python modules embedded in the notebooks, one folder per study |
| `preregistrations/` | decision rules and pre-registration texts, written before the corresponding runs |
| `reports/` | the text reports produced by the runs; every number in the paper comes from these files |
| `figures/paper/` | figures used in the paper; `figures/extra/` additional diagnostic figures |

## Study map (paper section ↔ notebook ↔ code ↔ report)
| study | what it tests | status in paper | notebook (`notebooks/main/`) | code (`src/`) | report |
|---|---|---|---|---|---|
| Phase A | leakage audit of the original split, pilot diagnostics | Results: leakage | `01_…` | `phaseA_and_data_audit` | `PhaseA_report.txt` |
| data audit | patient-level clean split (CBIS-DDSM, rule B) | Methods: data | `02_…` | `phaseA_and_data_audit/prepare_clean.py`, `data_audit.py` | – |
| clean baseline | baseline retrained from scratch on the clean split (seeds 42, 43; 40 epochs) | Methods: base model | `03_…` | `exp2_reference/train_arm2.py` | – |
| S2 | baseline audit; ConvNeXt judge for KID-mammo; guidance choice; FP reference | Methods: evaluation | `04_…` | `S2_audit_judges_metrics` | – |
| S3 | training with the FP residual | rules fixed before the run | `05_…` | `S3_fp_training` | `S3_report.txt` |
| S4a | FP-derived regularisers, FP gradient SNR | rules fixed before the run | `06_…` | `S4a_regularisers` | `S4a_report.txt` |
| S5 | physics-based quality scoring / filtering | rules fixed before the run | `07_…` | `S5_physics_scoring` | `S5_report.txt` |
| S6 | training-free fixes (weight averaging, best-of-2, thresholding) | rules fixed before the run | `08_…` | `S6_training_free` | `analysis_pack_results.txt` (S6 rows) |
| S7 | acquisition-noise (Poisson–Gaussian) layer | rules fixed before the run | `09_…` | `S7_acquisition_noise` | `S7_report.txt` |
| S8 | **pre-registered** replication on CMMD + held-out test stage (C1–C5) | Results: replication | `10_…`, `11_…` | `S8_replication` | `S8_FINAL_DECISION_SUMMARY.txt`, `S8_session1_report.txt` |
| S9 | mechanism diagnostics (resolution sweep, noise curves, spectra, saturation, FP gradient) | exploratory | `12_…` | `S9_diagnostics` | `S9_PLUS_report.txt` |
| S11 | **pre-registered** physics-based dose-reduction augmentation (P1–P3) | Results: augmentation | `13_…` | `S11_dose_augmentation` | `S11_report_corrected.txt` (+ original) |
| analysis pack | offline analyses on existing results (correlations, effect sizes, figures C–H) | Results / Discussion | – | `analysis_pack_make_analysis.py` | `analysis_pack_results.txt` |

`S11_report_corrected.txt` fixes a label in the secondary-endpoint printout only; all numbers are identical to
`S11_report_original_kaggle.txt` (Appendix B of the paper).

## Data
No images are redistributed. Both datasets are public on The Cancer Imaging Archive (TCIA); please follow their licence and
citation requirements:
* **CBIS-DDSM** — https://www.cancerimagingarchive.net/collection/cbis-ddsm/ (used on Kaggle as "CBIS-DDSM Breast Cancer Image Dataset")
* **CMMD** — https://www.cancerimagingarchive.net/collection/cmmd/ (used on Kaggle as "The Chinese Mammography Database (CMMD) 2022")

The patient-level splits are **regenerated deterministically** from the public metadata by the code (`prepare_clean.py`
for CBIS-DDSM rule B, `prepare_cmmd.py` for CMMD, fixed seed 0; validation = 15 % of training patients, seed 0), so no split
file has to be downloaded.

## How to reproduce
1. Create a Kaggle notebook from a file in `notebooks/main/` (File → Import Notebook).
2. Add the datasets above as inputs (Add Input → Datasets). Later notebooks also need the **output of earlier notebooks**
   as input (the markdown cell at the top of each notebook lists exactly which); the code finds attached folders by file
   name, so paths do not need editing.
3. Accelerator: GPU T4 ×2 for training/generation notebooks; S9 runs on CPU. Internet ON (to clone U24_synthesis and
   install the requirements). Run with *Save & Run All (Commit)*.
4. Run in the numbered order. Training runs take several GPU-hours each (Kaggle's weekly GPU quota applies); trained
   checkpoints are not included because of their size.

Main software: Python 3.10/3.11 (Kaggle image), PyTorch, MONAI Generative (via U24_synthesis requirements), torchmetrics,
torch-fidelity, timm, scikit-learn, SciPy, NumPy, pandas, pydicom, Pillow, Matplotlib.

## Pre-registration
S8's rules were fixed in the document in `preregistrations/S8_preregistration.txt` before the replication run; S10 and S11 were
registered on OSF before their runs (S10 was not run). S3–S7 used decision rules fixed in the script header before each run
(`S7_preregistration.txt` and `S4a_rule_amendment.txt` are those texts); they are not formal pre-registrations.

## Use of AI assistants
AI coding assistants (Anthropic Claude) helped write and debug this code; all code was reviewed, smoke-tested on reduced
data and run by the author, and all reported results come from these runs.

## Licence and citation
Code: MIT licence (see `LICENSE`). Reports, figures and pre-registration texts: CC BY 4.0. If you use this work, please cite
the paper (see `CITATION.cff`) **and** the U24_synthesis paper above.

Contact: Asef Ahmed Shimanto, Independent Researcher, Dhaka, Bangladesh — itsmeshimantokhan@gmail.com
