# s11_stats.py - S11 endpoints (pure numpy/scipy; rules in S11_preregistration.txt).
import os, sys, glob, json
import numpy as np
from scipy.stats import rankdata

D = sys.argv[1]
def auc(p, y):
    r = rankdata(p); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def load(ds, arm):
    fs = sorted(glob.glob(f"{D}/pred_{ds}_{arm}_s*.npz"))
    return [dict(np.load(f)) for f in fs]
def ens(runs, cond):
    return np.mean([r[cond] for r in runs], 0), runs[0][f"y_{cond}"]
rng = np.random.default_rng(0)
def boot_diff(pa, pb, y, alpha, n=2000):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; d = []
    for _ in range(n):
        b = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]); d.append(auc(pa[b], y[b]) - auc(pb[b], y[b]))
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
L = ["S11 REPORT - physics dose augmentation for the benign/malignant classifier (rules: S11_preregistration.txt)", "=" * 100]
R = {}
for ds in ("cmmd", "cbis"):
    runs = {arm: load(ds, arm) for arm in ("R0", "R1", "R2")}
    if not all(runs.values()):
        L.append(f"{ds}-trained: incomplete ({ {k: len(v) for k, v in runs.items()} })"); continue
    L.append(f"\n{ds.upper()}-TRAINED models ({len(runs['R0'])} seeds per arm) - single-seed AUC mean +- SD | 5-seed ensemble AUC")
    for cond in ("val", "cmmd_clean", "cmmd_dose50", "cmmd_dose25", "cbis_clean"):
        row = []
        for arm in ("R0", "R1", "R2"):
            s = [auc(r[cond], r[f"y_{cond}"] if cond != "val" else r["y_val"]) for r in runs[arm]]
            pe = np.mean([r[cond] for r in runs[arm]], 0); ye = runs[arm][0][f"y_{cond}"] if cond != "val" else runs[arm][0]["y_val"]
            row.append(f"{arm} {np.mean(s):.3f}+-{np.std(s, ddof=1):.3f} | {auc(pe, ye):.3f}")
            R.setdefault(ds, {}).setdefault(cond, {})[arm] = {"seed_aucs": s, "ensemble_auc": auc(pe, ye)}
        L.append(f"  {cond:12s}: " + "   ".join(row))
def endpoint(name, ds, cond, A, B, alpha, kind="sup"):
    pa, y = ens(load(ds, A), cond); pb, _ = ens(load(ds, B), cond)
    d = auc(pa, y) - auc(pb, y); lo, hi = boot_diff(pa, pb, y, alpha)
    sa = [auc(r[cond], r[f"y_{cond}"]) for r in load(ds, A)]; sb = [auc(r[cond], r[f"y_{cond}"]) for r in load(ds, B)]
    same = int(sum(np.sign(x - z) == np.sign(d) for x, z in zip(sa, sb)))
    ok = (lo > 0) if kind == "sup" else (lo > -0.02)
    excl = (lo > 0) or (hi < 0)                       # descriptive: CI excludes 0 in EITHER direction (fix 2026-10-23)
    L.append(f"  {name:4s} {ds}-trained, {cond:12s}: AUC {A} - {B} = {d:+.4f}  CI{100 * (1 - alpha):.2f}% {lo:+.4f}..{hi:+.4f} "
             f"| same sign in {same}/{len(sa)} seeds -> {('PASS' if ok else 'fail') if alpha < 0.05 else ('CI excludes 0' if excl else 'CI includes 0')}" + (" (non-inferiority, margin -0.02)" if kind == "ni" else ""))
    return {"diff": d, "ci": [lo, hi], "same_sign_seeds": same, "passed": bool(ok)}
if "cmmd" in R:
    L.append("\nPRIMARY ENDPOINTS (98.33 % CI, Bonferroni over 3)")
    E = {"P1": endpoint("P1", "cmmd", "cmmd_dose25", "R2", "R0", 0.05 / 3),
         "P2": endpoint("P2", "cmmd", "cmmd_dose25", "R2", "R1", 0.05 / 3),
         "P3": endpoint("P3", "cmmd", "cmmd_clean", "R2", "R0", 0.05 / 3, "ni")}
    L.append("\nSECONDARY (95 % CI, descriptive)")
    for nm, ds, cond, A, B in (("S1a", "cmmd", "cmmd_dose50", "R2", "R0"), ("S1b", "cmmd", "cmmd_dose50", "R2", "R1"),
                               ("S2a", "cmmd", "cbis_clean", "R2", "R0"), ("S2b", "cmmd", "cbis_clean", "R1", "R0"),
                               ("S3a", "cbis", "cmmd_clean", "R2", "R0"), ("S3b", "cbis", "cmmd_clean", "R1", "R0"),
                               ("S4", "cbis", "cbis_clean", "R2", "R0"), ("S5", "cmmd", "cmmd_dose25", "R1", "R0")):
        if ds in R:
            E[nm] = endpoint(nm, ds, cond, A, B, 0.05)
    R["endpoints"] = E
txt = "\n".join(L); print(txt)
open(f"{D}/S11_report.txt", "w").write(txt + "\n"); json.dump(R, open(f"{D}/S11_results.json", "w"), indent=1, default=float)
