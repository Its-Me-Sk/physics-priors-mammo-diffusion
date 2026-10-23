# a1_leakage.py - A1 HARD GATE: patient-level leakage + duplicate audit of the exact train/test split used by
# prepare.py (CBIS-DDSM official calc+mass train / test case-description CSVs, full mammograms at 256 px).
# PRE-STATED RULE:
#   FAIL  if any patient appears in both train and test, OR any train/test image pair is a (near-)duplicate
#         (Pearson correlation of 64x64 thumbnails > 0.995).
#   WARN  (not a leakage failure) for duplicates / label conflicts INSIDE a split; they are reported with counts.
import os, sys, json, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache

TAG = "A1"
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached")
RAW = os.path.dirname(os.path.dirname(hit[0]))
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
full = info[info["SeriesDescription"] == "full mammogram images"]
lookup = full.set_index("PatientName")["image_path"].to_dict()

def col(df, *names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f"none of {names} in columns {list(df.columns)[:12]}")

rows = []
for split, csvs in [("train", ["csv/calc_case_description_train_set.csv", "csv/mass_case_description_train_set.csv"]),
                    ("test", ["csv/calc_case_description_test_set.csv", "csv/mass_case_description_test_set.csv"])]:
    for c in csvs:
        df = pd.read_csv(os.path.join(RAW, c))
        pid, path, pat = col(df, "patient_id"), col(df, "image file path"), col(df, "pathology")
        side, view = col(df, "left or right breast"), col(df, "image view")
        for _, r in df.iterrows():
            name = str(r[path]).split("/")[0]
            rel = lookup.get(name)
            ok = rel is not None and os.path.exists(os.path.join(RAW, "jpeg", rel.replace("CBIS-DDSM/jpeg/", "")))
            rows.append({"split": split, "csv": os.path.basename(c), "patient": str(r[pid]), "side": str(r[side]),
                         "view": str(r[view]), "name": name, "jpeg": rel if ok else None,
                         "label": 1 if r[pat] == "MALIGNANT" else 0})
R = pd.DataFrame(rows)
R_ok = R[R.jpeg.notna()]
rep = {"rows_total": len(R), "rows_with_existing_jpeg": len(R_ok)}

# ---- 1. patient overlap train/test (true patient id, e.g. P_00001) ----
ptr, pte = set(R_ok[R_ok.split == "train"].patient), set(R_ok[R_ok.split == "test"].patient)
ov = sorted(ptr & pte)
rep["patients_train"], rep["patients_test"], rep["patients_in_both"] = len(ptr), len(pte), len(ov)
rep["patients_in_both_examples"] = ov[:10]
# ---- 2. same image (same jpeg) in both splits ----
itr, ite = set(R_ok[R_ok.split == "train"].jpeg), set(R_ok[R_ok.split == "test"].jpeg)
rep["images_in_both_splits"] = len(itr & ite)
# ---- 3. duplicates / label conflicts inside each split (one CSV row per abnormality) ----
for s in ("train", "test"):
    g = R_ok[R_ok.split == s].groupby("jpeg")
    n_rows, n_img = int((R_ok.split == s).sum()), int(g.ngroups)
    conflict = int((g.label.nunique() > 1).sum())
    rep[f"{s}_rows"], rep[f"{s}_unique_images"] = n_rows, n_img
    rep[f"{s}_rows_that_repeat_an_image"] = n_rows - n_img
    rep[f"{s}_images_with_conflicting_labels"] = conflict
    rep[f"{s}_breasts_per_patient_median"] = float(R_ok[R_ok.split == s].groupby("patient").side.nunique().median())

# ---- 4. near-duplicates in the ACTUAL caches used for training/testing ----
def thumbs(x):
    t = x.reshape(x.shape[0], 64, 4, 64, 4).mean((2, 4)).reshape(x.shape[0], -1).astype(np.float64)
    t -= t.mean(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return t
xtr, ytr = load_cache("train"); xte, yte = load_cache("test")
A, B = thumbs(xtr), thumbs(xte)
C = B @ A.T                                                   # (n_test, n_train) correlations
best = C.max(1)
rep["cache_train_n"], rep["cache_test_n"] = int(len(xtr)), int(len(xte))
rep["test_to_train_max_corr_quantiles"] = {q: float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99, 1.0)}
rep["test_train_pairs_corr_gt_0.995"] = int((C > 0.995).sum())
rep["test_train_pairs_exact_equal"] = int(sum(int(np.any(np.all(xtr.reshape(len(xtr), -1) == xte[i].reshape(1, -1), 1)))
                                              for i in np.where(best > 0.999)[0]))
CT = A @ A.T; np.fill_diagonal(CT, -1)
rep["train_cache_pairs_corr_gt_0.995"] = int((np.triu(CT, 1) > 0.995).sum())
same_lab = (ytr[:, None] == ytr[None, :])
rep["train_cache_dup_pairs_with_different_labels"] = int(((np.triu(CT, 1) > 0.995) & ~same_lab).sum())

leak = rep["patients_in_both"] > 0 or rep["images_in_both_splits"] > 0 or rep["test_train_pairs_corr_gt_0.995"] > 0
rep["verdict"] = "FAIL (leakage)" if leak else "PASS (no patient/image overlap between train and test)"
rep["warnings"] = []
if rep["train_rows_that_repeat_an_image"] > 0:
    rep["warnings"].append(f"train: {rep['train_rows_that_repeat_an_image']} CSV rows repeat an image "
                           f"(prepare.py adds one copy per abnormality row)")
if rep["train_images_with_conflicting_labels"] > 0:
    rep["warnings"].append(f"train: {rep['train_images_with_conflicting_labels']} images carry BOTH labels")
if rep["train_cache_dup_pairs_with_different_labels"] > 0:
    rep["warnings"].append(f"train cache: {rep['train_cache_dup_pairs_with_different_labels']} duplicate pairs "
                           f"with different labels")
save_json(rep, "A1_leakage.json")
for k, v in rep.items():
    log(TAG, f"{k}: {v}")
