# S8 data check (CPU only, ~2 min): shows what is inside every attached Kaggle dataset, so the S8 notebook can
# read CMMD without guessing paths. Prints: folder tree, file types, label tables (csv / xlsx) and sample images.
import os, glob, collections
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
ROOT = "/kaggle/input"
IMG = (".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff")
DS = sorted(glob.glob(f"{ROOT}/datasets/*/*") + glob.glob(f"{ROOT}/notebooks/*/*") +
            [d for d in glob.glob(f"{ROOT}/*") if os.path.basename(d) not in ("datasets", "notebooks")])
for ds in DS:
    if not os.path.isdir(ds):
        continue
    ext = collections.Counter(); n = 0; first = []
    for dp, dns, fns in os.walk(ds):
        for f in fns:
            e = os.path.splitext(f)[1].lower(); ext[e] += 1; n += 1
            if len(first) < 8:
                first.append(os.path.join(dp, f))
    if n == 0:
        continue
    print("=" * 100); print("DATASET FOLDER:", ds, "| files:", n, "| types:", dict(ext.most_common(8)))
    for p in first:
        print("   sample:", p)
    depth = {}
    for dp, dns, fns in os.walk(ds):
        d = dp[len(ds):].count(os.sep)
        if d <= 2 and (dns or fns):
            depth.setdefault(d, []).append((dp, len(dns), len(fns)))
    for d in sorted(depth):
        for dp, nd, nf in depth[d][:6]:
            print(f"   level {d}: {dp} | {nd} sub-folders, {nf} files")
    for t in [f for f in glob.glob(f"{ds}/**/*", recursive=True) if f.lower().endswith((".csv", ".xlsx", ".xls"))][:6]:
        print("   --- TABLE:", t)
        if pd is None:
            continue
        try:
            df = pd.read_excel(t) if t.lower().endswith(("xlsx", "xls")) else pd.read_csv(t)
            print("   rows", len(df), "| columns", list(df.columns))
            print(df.head(4).to_string(max_colwidth=40))
            for c in df.columns:
                if df[c].nunique() <= 12:
                    print(f"   value counts [{c}]:", df[c].value_counts(dropna=False).to_dict())
        except Exception as e:
            print("   could not read:", repr(e)[:150])
    ims = [f for f in first if f.lower().endswith(IMG)][:3]
    for p in ims:
        try:
            if p.lower().endswith(".dcm"):
                import pydicom
                d = pydicom.dcmread(p); a = d.pixel_array
                print(f"   DICOM {p}: shape {a.shape} dtype {a.dtype} | Manufacturer {getattr(d, 'Manufacturer', '?')} | "
                      f"PatientID {getattr(d, 'PatientID', '?')} | Laterality {getattr(d, 'ImageLaterality', getattr(d, 'Laterality', '?'))}")
            else:
                from PIL import Image
                a = np.array(Image.open(p)); print(f"   IMAGE {p}: shape {a.shape} dtype {a.dtype} min {a.min()} max {a.max()}")
        except Exception as e:
            print("   could not open", p, repr(e)[:120])
print("=" * 100); print("CHECK DONE - copy everything above and send it to Claude.")
