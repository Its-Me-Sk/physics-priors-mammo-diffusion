# data_audit.py - CHECK ONLY (writes no cache, changes nothing): what a clean, patient-level CBIS-DDSM split would look like.
# Runs on CPU in ~15 min. Inputs: the Kaggle "CBIS-DDSM Breast Cancer Image Dataset" (same as all experiments).
# Reports:
#   1. why test rows were "skipped" in prepare.py (282 of 704) and whether they can be recovered through the series UID
#   2. one record per IMAGE (not per abnormality row): label agreement, conflicts
#   3. pixel duplicates (exact, after the same 256x256 resize) inside and across splits and across patients
#   4. patient overlap between the official train and test CSVs, and the size of the two possible fixes
#   5. the class counts of the resulting clean train / test
import os, json, subprocess, hashlib, collections
import numpy as np, pandas as pd
from PIL import Image

OUT = "/kaggle/working/data_audit"
os.makedirs(OUT, exist_ok=True)
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached: Add Input -> CBIS-DDSM Breast Cancer Image Dataset")
RAW = os.path.dirname(os.path.dirname(hit[0]))
JPEG = os.path.join(RAW, "jpeg")
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
jp = lambda rel: os.path.join(JPEG, str(rel).replace("CBIS-DDSM/jpeg/", ""))
rep = {"dicom_info_columns": list(info.columns),
       "series_descriptions": {str(k): int(v) for k, v in info["SeriesDescription"].value_counts(dropna=False).items()}}
full = info[info["SeriesDescription"] == "full mammogram images"]
by_name = full.set_index("PatientName")["image_path"].to_dict()
uid_col = next((c for c in ("SeriesInstanceUID", "SeriesInstanceUid") if c in info.columns), None)
by_uid = full.set_index(uid_col)["image_path"].to_dict() if uid_col else {}
jpeg_dirs = set(os.listdir(JPEG))
log = lambda *m: print(*m, flush=True)

# ---------------- 1. every CSV row -> image file (old rule, then series-UID rescue) ----------------
rows, reasons = [], collections.Counter()
examples = collections.defaultdict(list)
for split, csvs in [("train", ["csv/calc_case_description_train_set.csv", "csv/mass_case_description_train_set.csv"]),
                    ("test", ["csv/calc_case_description_test_set.csv", "csv/mass_case_description_test_set.csv"])]:
    for c in csvs:
        df = pd.read_csv(os.path.join(RAW, c))
        for _, r in df.iterrows():
            parts = str(r["image file path"]).strip().split("/")
            name = parts[0]
            rel, how = by_name.get(name), "patientname"
            if rel is not None and not os.path.exists(jp(rel)):
                rel, how = None, "patientname_file_missing"
            if rel is None:
                # rescue 1: series UID of the CSV path -> dicom_info full-mammogram row
                for comp in parts[1:3]:
                    if comp in by_uid and os.path.exists(jp(by_uid[comp])):
                        rel, how = by_uid[comp], "series_uid_dicom_info"
                        break
            if rel is None:
                # rescue 2: a jpeg folder named like a UID of the CSV path containing exactly one image
                for comp in parts[1:3]:
                    if comp in jpeg_dirs:
                        fs = [f for f in os.listdir(os.path.join(JPEG, comp)) if f.lower().endswith(".jpg")]
                        if len(fs) == 1:
                            rel, how = os.path.join(comp, fs[0]), "uid_folder_single_jpeg"
                            break
            if rel is None:
                how = "unresolved"
            reasons[f"{split}:{how}"] += 1
            if how != "patientname" and len(examples[f"{split}:{how}"]) < 3:
                examples[f"{split}:{how}"].append(str(r["image file path"])[:160])
            rows.append({"split": split, "csv": os.path.basename(c), "patient": str(r["patient_id"]),
                         "side": str(r["left or right breast"]), "view": str(r["image view"]),
                         "label": 1 if r["pathology"] == "MALIGNANT" else 0, "pathology": str(r["pathology"]),
                         "rel": rel, "how": how})
R = pd.DataFrame(rows)
rep["row_resolution"] = dict(reasons)
rep["row_resolution_examples"] = dict(examples)
log("row resolution:", dict(reasons))

# ---------------- 2. one record per image file ----------------
Rk = R[R.rel.notna()].copy()
log(f"loading {Rk.rel.nunique()} unique image files at 256x256 (same resize as prepare.py) ...")
digest, thumbs = {}, {}
for i, rel in enumerate(sorted(Rk.rel.unique())):
    a = np.array(Image.open(jp(rel)).convert("L").resize((256, 256)), dtype=np.uint8)
    digest[rel] = hashlib.md5(a.tobytes()).hexdigest()
    t = a.reshape(64, 4, 64, 4).mean((1, 3)).ravel().astype(np.float64)
    t -= t.mean(); t /= np.linalg.norm(t) + 1e-9
    thumbs[rel] = t
    if (i + 1) % 500 == 0:
        log(f"  {i + 1} images")
Rk["md5"] = Rk.rel.map(digest)
img = Rk.groupby("md5").agg(n_rows=("rel", "size"), files=("rel", "nunique"), splits=("split", lambda s: sorted(set(s))),
                            patients=("patient", lambda s: sorted(set(s))), labels=("label", lambda s: sorted(set(s))),
                            csvs=("csv", lambda s: sorted(set(s)))).reset_index()
img["conflict"] = img.labels.map(len) > 1
img["cross_split"] = img.splits.map(len) > 1
img["cross_patient"] = img.patients.map(len) > 1
rep["unique_pixel_images"] = int(len(img))
rep["images_with_label_conflict"] = int(img.conflict.sum())
rep["images_in_both_splits_pixel_identical"] = int(img.cross_split.sum())
rep["images_shared_by_different_patient_ids"] = int(img.cross_patient.sum())
rep["images_stored_as_more_than_one_file"] = int((img.files > 1).sum())
rep["examples_cross_split"] = img[img.cross_split].head(5)[["patients", "csvs", "labels"]].to_dict("records")
rep["examples_cross_patient"] = img[img.cross_patient].head(5)[["patients", "csvs", "labels"]].to_dict("records")

# near-duplicates that are not pixel-identical (thumbnail correlation > 0.995), first file of each md5
reps_ = img.md5.tolist()
first_file = Rk.drop_duplicates("md5").set_index("md5").rel
M = np.stack([thumbs[first_file[m]] for m in reps_])
C = M @ M.T
np.fill_diagonal(C, -1)
iu = np.triu_indices(len(M), 1)
near = C[iu] > 0.995
pairs = [(reps_[a], reps_[b]) for a, b in zip(iu[0][near], iu[1][near])]
pat = img.set_index("md5").patients
spl = img.set_index("md5").splits
rep["near_duplicate_pairs_not_identical"] = len(pairs)
rep["near_duplicate_pairs_cross_split"] = int(sum(("train" in spl[a] and "test" in spl[b]) or ("test" in spl[a] and "train" in spl[b])
                                                  for a, b in pairs))
rep["near_duplicate_pairs_cross_patient"] = int(sum(set(pat[a]).isdisjoint(pat[b]) for a, b in pairs))

# ---------------- 3. patient overlap and the two possible fixes ----------------
ptr = set(Rk[Rk.split == "train"].patient); pte = set(Rk[Rk.split == "test"].patient)
both = sorted(ptr & pte)
clean = img[~img.conflict].copy()
clean["label"] = clean.labels.map(lambda s: s[0])
def is_in(split):
    return clean.splits.map(lambda s: split in s)
def counts(df):
    return {"images": int(len(df)), "benign": int((df.label == 0).sum()), "malignant": int((df.label == 1).sum()),
            "patients": int(len(set(p for ps in df.patients for p in ps)))}
touch = clean.patients.map(lambda ps: any(p in both for p in ps))
# fix A: overlapping patients stay in TRAIN, their test images are removed
trA, teA = clean[is_in("train")], clean[is_in("test") & ~touch & ~is_in("train")]
# fix B: overlapping patients go to TEST only, their train images are removed
trB, teB = clean[is_in("train") & ~touch & ~is_in("test")], clean[is_in("test") | touch]
rep["patients_train_test_both"] = [len(ptr), len(pte), len(both)]
rep["fix_A_keep_overlap_patients_in_train"] = {"train": counts(trA), "test": counts(teA)}
rep["fix_B_move_overlap_patients_to_test"] = {"train": counts(trB), "test": counts(teB)}
rep["old_prepare_py"] = {"train_rows_used": int((R.split == "train").sum() - reasons["train:unresolved"]),
                         "test_rows_used_patientname_only": int(reasons["test:patientname"])}
with open(os.path.join(OUT, "data_audit.json"), "w") as fh:
    json.dump(rep, fh, indent=1, default=str)
R.to_csv(os.path.join(OUT, "rows_resolved.csv"), index=False)
img.to_csv(os.path.join(OUT, "images_unique.csv"), index=False)
log("\n" + "=" * 90)
for k, v in rep.items():
    if k not in ("dicom_info_columns",):
        log(f"{k}: {v}")
log("=" * 90 + "\nSend data_audit/data_audit.json (or this whole printout) to Claude.")
