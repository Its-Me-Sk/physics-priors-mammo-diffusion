# prepare_cmmd.py - CMMD cache for S8 (rules in S8_preregistration.txt, fixed before any CMMD model was trained).
#   cohort D1 only (D2 is malignant-only -> class would be confounded with cohort)
#   label = clinical-table "classification" joined per (patient ID, breast side) with the DICOM ImageLaterality tag
#   8-bit grey (MONOCHROME1 inverted), PIL resize to 256 x 256 (same call as for CBIS-DDSM)
#   patient-level split, seed 0, 80 % train / 20 % test, stratified by patient label
# Writes $PA_TMP/{train,test}_cache.npz (x, y) and $PA_DATA/{train,test}_cache.npz (x, y, md5, patient) + cmmd_summary.json.
# Reuses an earlier session's cmmd_data/ if attached (identical bytes).
import os, glob, json, shutil, hashlib, time, collections
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

TMP = os.environ.get("PA_TMP", "/tmp/cmmd"); DATA = os.environ.get("PA_DATA", "/kaggle/working/cmmd_data")
os.makedirs(TMP, exist_ok=True); os.makedirs(DATA, exist_ok=True)

def to_tmp(src):
    for s in ("train", "test"):
        d = np.load(os.path.join(src, f"{s}_cache.npz"))
        np.savez(os.path.join(TMP, f"{s}_cache.npz"), x=d["x"], y=d["y"])
        print(f"CMMD {s}: {d['x'].shape} benign={int((d['y'] == 0).sum())} malignant={int((d['y'] == 1).sum())}")

if os.path.exists(os.path.join(DATA, "cmmd_summary.json")):
    to_tmp(DATA); print("CMMD cache already built in this session"); raise SystemExit(0)
prev = [p for p in _walk_find("cmmd_summary.json") if os.path.basename(os.path.dirname(p)) == "cmmd_data"]
if prev:
    for f in os.listdir(os.path.dirname(prev[0])):
        shutil.copy(os.path.join(os.path.dirname(prev[0]), f), os.path.join(DATA, f))
    to_tmp(DATA); print("CMMD cache REUSED from", os.path.dirname(prev[0])); raise SystemExit(0)

t0 = time.time()
xl = _walk_find("CMMD_clinicaldata_revision.xlsx")
if not xl:
    raise SystemExit("CMMD not attached: Add Input -> search 'CMMD' -> 'The Chinese Mammography Database (CMMD) 2022' (tommyngx)")
ROOT = os.path.dirname(xl[0])
clin = pd.read_excel(xl[0])
meta = pd.read_csv(os.path.join(ROOT, "metadata.csv"))
clin = clin[clin["ID1"].astype(str).str.startswith("D1-")].copy()
clin["side"] = clin["LeftRight"].astype(str).str.strip().str.upper()
clin["label"] = clin["classification"].astype(str).str.strip().str.lower().map({"benign": 0, "malignant": 1})
dup = clin.groupby(["ID1", "side"]).label.nunique()
amb = set(dup[dup > 1].index)                                   # same breast listed with two labels -> excluded
LAB = {k: int(v) for k, v in clin.groupby(["ID1", "side"]).label.first().items() if k not in amb and not pd.isna(v)}
print(f"clinical table: {len(clin)} D1 rows -> {len(LAB)} labelled breasts ({len(amb)} ambiguous excluded)", flush=True)

import pydicom
meta = meta[meta["Subject ID"].astype(str).str.startswith("D1-")]
rows, why = [], collections.Counter()
for _, r in meta.iterrows():
    folder = os.path.join(ROOT, str(r["File Location"]).replace("\\", "/").lstrip("./"))
    if not os.path.isdir(folder):
        why["series folder missing"] += 1; continue
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".dcm"):
            continue
        p = os.path.join(folder, f)
        ds = pydicom.dcmread(p)
        side = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).strip().upper()
        pid = str(r["Subject ID"])
        if (pid, side) not in LAB:
            why["no unique label for this breast"] += 1; continue
        a = ds.pixel_array.astype(np.float64)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
        if ds.pixel_array.dtype != np.uint8:
            a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255
        im = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
        x = np.array(im.resize((256, 256)), dtype=np.uint8)
        rows.append({"patient": pid, "side": side, "label": LAB[(pid, side)], "file": p,
                     "md5": hashlib.md5(x.tobytes()).hexdigest(), "x": x})
        why["used"] += 1
    if len(rows) and len(rows) % 400 == 0:
        print(f"  {len(rows)} images ({(time.time() - t0) / 60:.1f} min)", flush=True)
print("image resolution:", dict(why), flush=True)
D = pd.DataFrame([{k: v for k, v in r.items() if k != "x"} for r in rows])
X = np.stack([r["x"] for r in rows])
D = D.drop_duplicates("md5"); X = X[D.index.to_numpy()]; D = D.reset_index(drop=True)

# patient-level split, seed 0, stratified by patient label (a patient with any malignant breast = malignant)
plab = D.groupby("patient").label.max()
rng = np.random.default_rng(0); test_p = set()
for c in (0, 1):
    ps = sorted(plab[plab == c].index)
    test_p |= set(rng.choice(ps, int(round(0.2 * len(ps))), replace=False).tolist())
D["split"] = np.where(D.patient.isin(test_p), "test", "train")
assert not (set(D[D.split == "train"].patient) & set(D[D.split == "test"].patient))
summ = {"rule": "CMMD cohort D1; label per (patient, side) via DICOM ImageLaterality; 256x256; patient split 80/20 seed 0",
        "image_resolution": dict(why), "ambiguous_breasts_excluded": len(amb), "duplicates_removed": int(len(rows) - len(D))}
for s in ("train", "test"):
    m = (D.split == s).to_numpy(); d = D[m]
    x, y = X[m], d.label.to_numpy().astype(np.int64)
    np.savez(os.path.join(TMP, f"{s}_cache.npz"), x=x, y=y)
    np.savez_compressed(os.path.join(DATA, f"{s}_cache.npz"), x=x, y=y, md5=d.md5.to_numpy().astype(str),
                        patient=d.patient.to_numpy().astype(str))
    d[["patient", "side", "label", "md5", "file"]].to_csv(os.path.join(DATA, f"manifest_{s}.csv"), index=False)
    summ[s] = {"images": int(len(d)), "benign": int((y == 0).sum()), "malignant": int((y == 1).sum()), "patients": int(d.patient.nunique())}
    print(f"CMMD {s}: {x.shape} benign={summ[s]['benign']} malignant={summ[s]['malignant']} patients={summ[s]['patients']}", flush=True)
summ["build_minutes"] = (time.time() - t0) / 60
json.dump(summ, open(os.path.join(DATA, "cmmd_summary.json"), "w"), indent=1)
print(f"CMMD cache written ({summ['build_minutes']:.1f} min)")
