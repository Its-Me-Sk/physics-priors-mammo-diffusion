# a10_metadata.py - A10 (CPU): what differs between benign and malignant images in the REAL clean data?
# For every clean image (same order as the cache): view (CC/MLO), laterality, breast density (1-4), abnormality type
# (calc/mass) from the CBIS CSVs, and digitiser (dicom_info SecondaryCaptureDeviceManufacturer). Reported by class for
# train-fit / validation / test, with a patient-cluster bootstrap 95 % CI for the malignant-minus-benign MLO share.
# Also writes the per-image view labels used to train the view classifier (A6b).
import os, sys, ast, json, subprocess, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, OUT

TAG = "A10"
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True, capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached: Add Input -> CBIS-DDSM Breast Cancer Image Dataset (needed for the metadata)")
RAW = os.path.dirname(os.path.dirname(hit[0])); JPEG = os.path.join(RAW, "jpeg")
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
full = info[info["SeriesDescription"] == "full mammogram images"]
by_name = full.set_index("PatientName")["image_path"].to_dict()
by_uid = full.set_index("SeriesInstanceUID")["image_path"].to_dict()
scan_col = "SecondaryCaptureDeviceManufacturer" if "SecondaryCaptureDeviceManufacturer" in info.columns else None
scanner_of = info.set_index("image_path")[scan_col].astype(str).to_dict() if scan_col else {}
jpeg_dirs = set(os.listdir(JPEG))
jp = lambda rel: os.path.join(JPEG, str(rel).replace("CBIS-DDSM/jpeg/", ""))

def col(df, *names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None

attr = collections.defaultdict(lambda: collections.defaultdict(set))     # rel -> field -> values
for c in ["calc_case_description_train_set", "mass_case_description_train_set", "calc_case_description_test_set",
          "mass_case_description_test_set"]:
    df = pd.read_csv(os.path.join(RAW, "csv", c + ".csv"))
    cv, cs, cd, ct = col(df, "image view"), col(df, "left or right breast"), col(df, "breast_density", "breast density"), \
        col(df, "abnormality type")
    for _, r in df.iterrows():
        parts = str(r["image file path"]).strip().split("/")
        rel = by_name.get(parts[0])
        if rel is not None and not os.path.exists(jp(rel)):
            rel = None
        if rel is None:
            for comp in parts[1:3]:
                if comp in by_uid and os.path.exists(jp(by_uid[comp])):
                    rel = by_uid[comp]; break
        if rel is None:
            for comp in parts[1:3]:
                if comp in jpeg_dirs:
                    fs = [f for f in os.listdir(os.path.join(JPEG, comp)) if f.lower().endswith(".jpg")]
                    if len(fs) == 1:
                        rel = os.path.join(comp, fs[0]); break
        if rel is None:
            continue
        a = attr[rel]
        a["view"].add(str(r[cv]).strip().upper()); a["side"].add(str(r[cs]).strip().upper())
        if cd: a["density"].add(str(r[cd]).strip())
        if ct: a["abn"].add(str(r[ct]).strip().lower())
        sc = scanner_of.get(rel) or scanner_of.get("CBIS-DDSM/jpeg/" + rel)
        a["scanner"].add(str(sc) if sc not in (None, "nan") else "unknown")

def one(vals):
    vals = {v for v in vals if v not in ("", "NAN", "nan")}
    return next(iter(vals)) if len(vals) == 1 else ("mixed" if vals else "unknown")

rep, labels = {}, {}
for split in ("train", "test"):
    man = pd.read_csv(os.path.join("/kaggle/working/clean_data", f"manifest_{split}.csv"))
    _, y, pat, md5 = load_clean_meta(split)
    assert (man.md5.astype(str).values == md5).all(), "manifest order != cache order"
    recs = []
    for files in man.files:
        acc = collections.defaultdict(set)
        for rel in ast.literal_eval(files):
            for k, v in attr.get(rel, {}).items():
                acc[k] |= v
        recs.append({k: one(acc.get(k, set())) if k != "abn" else "+".join(sorted(acc.get("abn", set()))) or "unknown"
                     for k in ("view", "side", "density", "scanner", "abn")})
    M = pd.DataFrame(recs); M["label"] = y; M["patient"] = pat
    labels[split] = np.array(M["view"].tolist(), dtype=str)
    groups = {"test": np.ones(len(M), bool)} if split == "test" else {"train_fit": ~val_mask(pat), "val": val_mask(pat)}
    for gname, gm in groups.items():
        G = M[gm]; out = {}
        for lab, nm in ((0, "benign"), (1, "malignant")):
            S = G[G.label == lab]
            out[nm] = {"n": int(len(S)), "share_MLO": float((S["view"] == "MLO").mean()), "share_LEFT": float((S.side == "LEFT").mean()),
                       "density": S.density.value_counts(normalize=True).round(3).to_dict(),
                       "scanner": S.scanner.value_counts(normalize=True).round(3).to_dict(),
                       "abnormality": S.abn.value_counts(normalize=True).round(3).to_dict(),
                       "view_unknown_or_mixed": int((~S["view"].isin(["CC", "MLO"])).sum())}
        # patient-cluster bootstrap CI of (MLO share malignant - MLO share benign)
        rng = np.random.default_rng(0)
        P = G.assign(mlo=(G["view"] == "MLO").astype(int)).groupby("patient").apply(
            lambda d: pd.Series({"nb": (d.label == 0).sum(), "nbm": ((d.label == 0) & (d.mlo == 1)).sum(),
                                 "nm": (d.label == 1).sum(), "nmm": ((d.label == 1) & (d.mlo == 1)).sum()})).to_numpy(float)
        diffs = []
        for _ in range(2000):
            S_ = P[rng.integers(0, len(P), len(P))].sum(0)
            if S_[0] > 0 and S_[2] > 0:
                diffs.append(S_[3] / S_[2] - S_[1] / S_[0])
        out["MLO_share_malignant_minus_benign"] = float(out["malignant"]["share_MLO"] - out["benign"]["share_MLO"])
        out["MLO_diff_ci95_patient_bootstrap"] = ([float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
                                                  if diffs else [float("nan"), float("nan")])
        rep[gname] = out
        log(TAG, f"{gname}: MLO share benign {out['benign']['share_MLO']:.3f} (n={out['benign']['n']}) | malignant "
                 f"{out['malignant']['share_MLO']:.3f} (n={out['malignant']['n']}) | diff {out['MLO_share_malignant_minus_benign']:+.3f} "
                 f"CI {out['MLO_diff_ci95_patient_bootstrap'][0]:+.3f}..{out['MLO_diff_ci95_patient_bootstrap'][1]:+.3f}")
        log(TAG, f"{gname}: density benign {out['benign']['density']} | malignant {out['malignant']['density']}")
        log(TAG, f"{gname}: scanner benign {out['benign']['scanner']} | malignant {out['malignant']['scanner']}")
np.savez(os.path.join(OUT, "A10_view_labels.npz"), view_train=labels["train"], view_test=labels["test"])
save_json(rep, "A10_metadata.json")
