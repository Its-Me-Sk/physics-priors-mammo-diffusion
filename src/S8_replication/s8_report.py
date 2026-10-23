# s8_report.py - S8 summary (plain Python + json only): job status, data, training curves, FP logs, all reports.
import os, json, glob
OUT = os.environ.get("PA_OUT", "/kaggle/working/S8"); CK = "/kaggle/working/ckpt"
L = ["S8 REPORT - replication on CMMD + longer CBIS-DDSM training (rules: S8_preregistration.txt)", "=" * 100]
st = json.load(open(f"{OUT}/S8_state.json")) if os.path.exists(f"{OUT}/S8_state.json") else {}
if st:
    L.append(f"session {st.get('session')} | jobs done {len(st.get('done', []))}/{st.get('n_jobs')} | "
             f"failed {st.get('failed')} | still to do {st.get('todo')}")
cs = "/kaggle/working/cmmd_data/cmmd_summary.json"
if os.path.exists(cs):
    c = json.load(open(cs)); L.append(f"CMMD data: train {c.get('train')} | test {c.get('test')} | resolution {c.get('image_resolution')}")
for f in sorted(glob.glob(f"{OUT}/*_clf_s0.json")):
    L.append(f"classifier {os.path.basename(f)}: {json.load(open(f))}")
L.append("-" * 100 + "\nTRAINING")
for arm in ("cmmd_s42", "base_clean_s42", "cmmd_m0", "cmmd_l016", "cl_m0", "cl_l016"):
    f = f"{CK}/{arm}_history.json"
    if not os.path.exists(f):
        L.append(f"  {arm}: not started"); continue
    h = json.load(open(f)); last = h[-1]
    pick = [x for x in h if x["epoch"] in (1, 10, 20, 30, 40, 41, 42, 44, 50, 60, 70, 80, 90, 91, 92)]
    L.append(f"  {arm}: {last['epoch']} epochs | last loss {last['loss']:.4f} mse {last['mse']:.5f} | "
             f"{sum(x['minutes'] for x in h[-3:]) / len(h[-3:]):.1f} min/epoch")
    L.append("     mse by epoch: " + ", ".join(f"{x['epoch']}: {x['mse']:.5f}" for x in pick))
    if "fp_norm_mean" in last:
        L.append("     FP term / clipped share by epoch: " + ", ".join(
            f"{x['epoch']}: {x['fp_norm_mean'] if x['fp_norm_mean'] is None else round(x['fp_norm_mean'], 4)} / {x['clipped_share']:.2f}" for x in h))
for f in sorted(glob.glob(f"{OUT}/sref_*.json")):
    d = json.load(open(f)); L.append(f"  {os.path.basename(f)}: epoch {d.get('epoch')} S_ref {d['S_ref_table']}")
L.append("-" * 100 + "\nCOMPARISONS (validation, then the single test stage)")
for tag in ("S8_cmmd_fp", "S8_cbis_fp", "S8_cmmd_main", "S8_cbis_long"):
    for stage in ("val", "test"):
        f = f"{OUT}/{tag}_{stage}_report.txt"
        L.append(open(f).read() if os.path.exists(f) else f"[{tag} {stage.upper()}] not run yet")
open(f"{OUT}/S8_report.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
