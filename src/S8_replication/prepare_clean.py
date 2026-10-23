# prepare_clean.py - CLEAN patient-level CBIS-DDSM cache (rule B, decided 19 Oct 2026 after the data audit).
#   unit            = one unique image (pixel-identical after the 256x256 resize = the same image), not one CSV row
#   resolution      = PatientName in dicom_info (as before); if missing: series UID of the CSV path -> dicom_info, or
#                     the jpeg folder named by that UID holding exactly one image (recovers the 282 skipped test rows)
#   image check     = native min side >= 1000 px and not a binary mask (a full mammogram), else rejected + reported
#   label conflicts = an image carrying both BENIGN* and MALIGNANT rows -> excluded
#   split (rule B)  = patients that occur in both official train and test CSVs go to TEST (all their images);
#                     an image that occurs in both splits therefore ends up in test; train = everything else
#   no class subsampling (classes are already near-balanced)
# Writes /tmp/train_cache.npz and /tmp/test_cache.npz (x uint8 (N,256,256), y int64) exactly like the old prepare.py,
# plus /kaggle/working/clean_data/ (compressed caches with md5 + patient, manifests, summary) so that EVERY later
# notebook can attach this output and use byte-identical data.
import os, glob, json, shutil, subprocess, hashlib, collections, time
def _walk_find(name, root="/kaggle/input"):
    """S8: like glob('/kaggle/input/**/name') but skips the huge image folders (CMMD DICOM tree, CBIS jpeg)"""
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)

import numpy as np, pandas as pd
from PIL import Image

OUT = "/kaggle/working/clean_data"
os.makedirs(OUT, exist_ok=True)
EXPECTED = {"train": 2336, "test": 663}          # data audit, rule B (before the full-mammogram check)

def to_tmp(src_dir):
    for s in ("train", "test"):
        d = np.load(os.path.join(src_dir, f"{s}_cache.npz"))
        np.savez(f"/tmp/{s}_cache.npz", x=d["x"], y=d["y"])
        print(f"{s}: {d['x'].shape} benign={int((d['y'] == 0).sum())} malignant={int((d['y'] == 1).sum())}")

# ---------- 0. already built in this session, or attached from an earlier session -> reuse (identical bytes) ----------
if os.path.exists(os.path.join(OUT, "clean_summary.json")):
    to_tmp(OUT); print("clean caches already built in this session"); raise SystemExit(0)
prev = [p for p in _walk_find("clean_summary.json") if os.path.basename(os.path.dirname(p)) == "clean_data"]
if prev:
    src = os.path.dirname(prev[0])
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(OUT, f))
    to_tmp(OUT)
    print("clean caches REUSED from", src); raise SystemExit(0)

t0 = time.time()
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached: Add Input -> CBIS-DDSM Breast Cancer Image Dataset")
RAW = os.path.dirname(os.path.dirname(hit[0]))
JPEG = os.path.join(RAW, "jpeg")
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
jp = lambda rel: os.path.join(JPEG, str(rel).replace("CBIS-DDSM/jpeg/", ""))
full = info[info["SeriesDescription"] == "full mammogram images"]
by_name = full.set_index("PatientName")["image_path"].to_dict()
by_uid = full.set_index("SeriesInstanceUID")["image_path"].to_dict()
jpeg_dirs = set(os.listdir(JPEG))

# ---------- 1. every CSV row -> image file ----------
rows, how_n = [], collections.Counter()
for split, csvs in [("train", ["csv/calc_case_description_train_set.csv", "csv/mass_case_description_train_set.csv"]),
                    ("test", ["csv/calc_case_description_test_set.csv", "csv/mass_case_description_test_set.csv"])]:
    for c in csvs:
        for _, r in pd.read_csv(os.path.join(RAW, c)).iterrows():
            parts = str(r["image file path"]).strip().split("/")
            rel, how = by_name.get(parts[0]), "patientname"
            if rel is not None and not os.path.exists(jp(rel)):
                rel = None
            if rel is None:
                for comp in parts[1:3]:
                    if comp in by_uid and os.path.exists(jp(by_uid[comp])):
                        rel, how = by_uid[comp], "series_uid_dicom_info"
                        break
            if rel is None:
                for comp in parts[1:3]:
                    if comp in jpeg_dirs:
                        fs = [f for f in os.listdir(os.path.join(JPEG, comp)) if f.lower().endswith(".jpg")]
                        if len(fs) == 1:
                            rel, how = os.path.join(comp, fs[0]), "uid_folder_single_jpeg"
                            break
            if rel is None:
                how = "unresolved"
            how_n[f"{split}:{how}"] += 1
            rows.append({"split": split, "csv": os.path.basename(c), "patient": str(r["patient_id"]),
                         "label": 1 if r["pathology"] == "MALIGNANT" else 0, "rel": rel, "how": how})
R = pd.DataFrame(rows)
print("row resolution:", dict(how_n), flush=True)
Rk = R[R.rel.notna()].copy()

# ---------- 2. load every unique file once: full-mammogram check + 256x256 array + md5 ----------
arr, md5, native, reject = {}, {}, {}, {}
files = sorted(Rk.rel.unique())
for i, rel in enumerate(files):
    with Image.open(jp(rel)) as im:
        W, H = im.size
        g = im.convert("L")
        native[rel] = (W, H)
        if min(W, H) < 1000:
            reject[rel] = f"small ({W}x{H}) - not a full mammogram"
            continue
        small = np.asarray(g.resize((512, 512)))
        if np.mean((small > 30) & (small < 225)) < 0.02:
            reject[rel] = "binary image - looks like an ROI mask"
            continue
        a = np.array(g.resize((256, 256)), dtype=np.uint8)            # identical resize to the old prepare.py
    arr[rel] = a
    md5[rel] = hashlib.md5(a.tobytes()).hexdigest()
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(files)} files ({(time.time() - t0) / 60:.1f} min)", flush=True)
rej_rows = Rk[Rk.rel.isin(reject)]
Rk = Rk[~Rk.rel.isin(reject)].copy()
Rk["md5"] = Rk.rel.map(md5)

# ---------- 3. one record per image, conflicts, rule-B split ----------
ptr, pte = set(R[(R.split == "train") & R.rel.notna()].patient), set(R[(R.split == "test") & R.rel.notna()].patient)
both = ptr & pte                                    # same definition as the data audit
img = Rk.groupby("md5").agg(rel=("rel", "first"), files=("rel", lambda s: sorted(set(s))),
                            splits=("split", lambda s: sorted(set(s))), patients=("patient", lambda s: sorted(set(s))),
                            labels=("label", lambda s: sorted(set(s))), csvs=("csv", lambda s: sorted(set(s)))).reset_index()
conflict = img.labels.map(len) > 1
clean = img[~conflict].copy()
clean["label"] = clean.labels.map(lambda s: s[0])
touch = clean.patients.map(lambda ps: any(p in both for p in ps))
in_tr = clean.splits.map(lambda s: "train" in s)
in_te = clean.splits.map(lambda s: "test" in s)
clean["final_split"] = np.where(in_te | touch, "test", "train")
clean = clean.sort_values("md5").reset_index(drop=True)
assert not (set(p for ps in clean[clean.final_split == "train"].patients for p in ps) &
            set(p for ps in clean[clean.final_split == "test"].patients for p in ps)), "patient overlap remains!"

summary = {"rule": "B: patients in both official splits -> test; label-conflict images excluded; one record per "
                   "pixel-identical image; full-mammogram check", "decided": "2026-10-19",
           "row_resolution": dict(how_n), "rejected_files": len(reject), "rejected_examples": dict(list(reject.items())[:10]),
           "rejected_rows": int(len(rej_rows)), "unique_images": int(len(img)), "label_conflict_images_excluded": int(conflict.sum()),
           "patients_in_both_official_splits": len(both), "build_minutes": None}
for s in ("train", "test"):
    d = clean[clean.final_split == s]
    x = np.stack([arr[r] for r in d.rel]); y = d.label.to_numpy().astype(np.int64)
    np.savez(f"/tmp/{s}_cache.npz", x=x, y=y)
    np.savez_compressed(os.path.join(OUT, f"{s}_cache.npz"), x=x, y=y, md5=d.md5.to_numpy().astype(str),
                        patient=d.patients.map(lambda p: "|".join(p)).to_numpy().astype(str))
    m = d[["md5", "label", "patients", "files", "csvs", "splits"]].copy()
    m["native_WxH"] = d.rel.map(lambda r: "x".join(map(str, native[r])))
    m.to_csv(os.path.join(OUT, f"manifest_{s}.csv"), index=False)
    pats = set(p for ps in d.patients for p in ps)
    summary[s] = {"images": int(len(d)), "benign": int((y == 0).sum()), "malignant": int((y == 1).sum()),
                  "patients": len(pats), "expected_from_audit": EXPECTED[s], "difference": int(len(d)) - EXPECTED[s]}
    print(f"{s}: {x.shape} benign={int((y == 0).sum())} malignant={int((y == 1).sum())} patients={len(pats)} "
          f"(audit expected {EXPECTED[s]}, difference {int(len(d)) - EXPECTED[s]})", flush=True)
summary["build_minutes"] = (time.time() - t0) / 60
json.dump(summary, open(os.path.join(OUT, "clean_summary.json"), "w"), indent=1, default=str)
print(f"rejected files (full-mammogram check): {len(reject)}", dict(list(reject.items())[:5]))
print("clean caches written to /tmp and", OUT, f"({summary['build_minutes']:.1f} min)")
