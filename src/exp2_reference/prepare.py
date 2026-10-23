
import os, subprocess
import numpy as np, pandas as pd
from PIL import Image
if os.path.exists("/tmp/train_cache.npz") and os.path.exists("/tmp/test_cache.npz"):
    print("caches already built"); raise SystemExit(0)
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached: Add Input -> CBIS-DDSM Breast Cancer Image Dataset")
RAW = os.path.dirname(os.path.dirname(hit[0]))
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
lookup = info[info["SeriesDescription"] == "full mammogram images"].set_index("PatientName")["image_path"].to_dict()
for split, csvs in [("train", ["csv/calc_case_description_train_set.csv", "csv/mass_case_description_train_set.csv"]),
                    ("test",  ["csv/calc_case_description_test_set.csv",  "csv/mass_case_description_test_set.csv"])]:
    xs, skipped = {0: [], 1: []}, 0
    for c in csvs:
        for _, row in pd.read_csv(os.path.join(RAW, c)).iterrows():
            lab = 1 if row["pathology"] == "MALIGNANT" else 0
            rel = lookup.get(row["image file path"].split("/")[0])
            if rel is None:
                skipped += 1; continue
            p = os.path.join(RAW, "jpeg", rel.replace("CBIS-DDSM/jpeg/", ""))
            if not os.path.exists(p):
                skipped += 1; continue
            xs[lab].append(np.array(Image.open(p).convert("L").resize((256, 256)), dtype=np.uint8))
    if split == "train" and len(xs[0]) > 1300:          # same balancing as before: 1300 benign, all malignant
        keep = np.sort(np.random.default_rng(42).choice(len(xs[0]), 1300, replace=False))
        xs[0] = [xs[0][i] for i in keep]
    x = np.stack(xs[0] + xs[1]); y = np.array([0] * len(xs[0]) + [1] * len(xs[1]), dtype=np.int64)
    np.savez(f"/tmp/{split}_cache.npz", x=x, y=y)
    print(f"{split}: {x.shape} benign={int((y == 0).sum())} malignant={int((y == 1).sum())} skipped={skipped}")
