# s9_analysis.py - S9+ EXPLORATORY CPU diagnostics (no training, no GPU, changes no pre-registered result).
# Parts: B CMMD resolution sweep of the noise-level function | C measured noise curves real vs generated |
# D radial power spectra | E intensity distributions | F KID sample-size stability | G model table |
# I FP gradient conditioning by timestep (from the S8 training logs).
# usage: python s9_analysis.py <S8 folder> <cmmd_data> <clean_data> <out> [n_sweep] [CMMD dicom root or ""]
import os, sys, glob, json, time
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S8, CMMD_DATA, CBIS_DATA, OUT = sys.argv[1:5]
N_SWEEP = int(sys.argv[5]) if len(sys.argv) > 5 else 120
CMMD_ROOT = sys.argv[6] if len(sys.argv) > 6 else ""
os.makedirs(OUT, exist_ok=True)
REP, RES = [], {}
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.append(s)
C = {"real": "#0b0b0b", "grid": "#e4e3df", "surf": "#fcfcfb"}
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#c79a1e", "#3aa6b9"]
plt.rcParams.update({"figure.facecolor": C["surf"], "axes.facecolor": C["surf"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": C["grid"], "font.size": 9, "savefig.dpi": 170, "savefig.bbox": "tight"})

def val_mask(patients, frac=0.15, seed=0):              # identical to phaseA_common.val_mask
    ups = sorted(set(patients)); rng = np.random.default_rng(seed)
    val = set(rng.choice(ups, int(round(frac * len(ups))), replace=False).tolist())
    return np.array([p in val for p in patients])
def trainfit(folder):
    d = np.load(os.path.join(folder, "train_cache.npz")); return d["x"][~val_mask(d["patient"].astype(str))]
def gen(name):
    f = os.path.join(S8, f"gen_{name}.npz"); return np.load(f)["x"] if os.path.exists(f) else None
def mm01(X):
    X = X.astype(np.float32); mn = X.min(axis=(-2, -1), keepdims=True); mx = X.max(axis=(-2, -1), keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

# ------------- noise-level function: exact numpy port of pinn_physics.PhysicsNLF.stats / .fit -------------
# torch: separable 7-tap Gaussian blur with 'reflect' padding (= scipy 'mirror'); 7x7 avg_pool, count_include_pad=False;
# 16 Gaussian intensity bins 0.08..0.95 (width 0.6 x spacing); background mask local mean > 0.04; input = uint8 / 255.
G1 = np.exp(-(np.arange(7) - 3.0) ** 2 / 2); G1 /= G1.sum()
CEN = np.linspace(0.08, 0.95, 16); BW = (0.95 - 0.08) / 15 * 0.6
_CNT = {}
def avg7(A):
    if A.shape not in _CNT:
        _CNT[A.shape] = ndimage.uniform_filter(np.ones(A.shape), 7, mode="constant", cval=0.0)
    return ndimage.uniform_filter(A, 7, mode="constant", cval=0.0) / _CNT[A.shape]
def nlf_image(u8):
    """u8: (H,W) uint8 -> (per-bin sum of weight*variance, per-bin weight) for one image"""
    I = u8.astype(np.float64) / 255.0
    b = ndimage.convolve1d(ndimage.convolve1d(I, G1, axis=1, mode="mirror"), G1, axis=0, mode="mirror")
    h = I - b
    mu = avg7(I); v = avg7(h * h)
    m = mu > 0.04; mu, v = mu[m], v[m]
    vs, ms = np.zeros(16), np.zeros(16)
    for k, c in enumerate(CEN):
        w = np.exp(-(mu - c) ** 2 / (2 * BW ** 2)); vs[k] = float((w * v).sum()); ms[k] = float(w.sum())
    return vs, ms
def nlf_from_P(P, nboot=200, seed=0):
    P = np.asarray(P)                                            # (n, 2, 16)
    V = P[:, 0].sum(0) / np.maximum(P[:, 1].sum(0), 1e-9); wt = P[:, 1].sum(0) / P[:, 1].sum()
    rng = np.random.default_rng(seed); B = []
    for _ in range(nboot):
        i = rng.integers(0, len(P), len(P)); B.append(P[i, 0].sum(0) / np.maximum(P[i, 1].sum(0), 1e-9))
    lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
    def wls(deg):
        A = np.vander(CEN, deg + 1, increasing=True)
        th = np.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * V)); pred = A @ th
        r2 = 1 - (wt * (V - pred) ** 2).sum() / (wt * (V - (wt * V).sum()) ** 2).sum(); return th.tolist(), float(r2)
    return {"V": V.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "weight": wt.tolist(), "linear": wls(1), "quadratic": wls(2),
            "n": int(len(P)), "peak_intensity": float(CEN[int(np.argmax(V))]), "rising_share": float(np.mean(np.diff(V) > 0))}
def nlf_set(X_u8):
    return nlf_from_P([nlf_image(x) for x in X_u8])
def nlf_line(tag, r):
    (b, a), r2l = r["linear"]
    return (f"  {tag:30s} n {r['n']:4d} | linear: var = {b:.2e} {a:+.2e}*I (R2 {r2l:.2f}) | quadratic R2 {r['quadratic'][1]:.2f}"
            f" | variance peak at I = {r['peak_intensity']:.2f} | rises in {100 * r['rising_share']:.0f} % of bin steps")

say("=" * 112); say("S9+ EXPLORATORY DIAGNOSTICS (CPU only; nothing here changes any pre-registered result)"); say("=" * 112)

# ================================ PART B - CMMD resolution sweep ================================
say("B. CMMD RESOLUTION SWEEP of the noise-level function: original DICOM -> 1/2 -> 1/4 -> 256 x 256 (the DDPM input).")
say("   Question: is the signal-dependent (quantum) noise visible at full resolution and lost by the downsampling?")
def resolve(p):
    if os.path.exists(p) or not CMMD_ROOT:
        return p
    parts = p.replace("\\", "/").split("/")
    for k in range(1, len(parts)):
        q = os.path.join(CMMD_ROOT, *parts[k:])
        if os.path.exists(q):
            return q
    return p
try:
    import pandas as pd, pydicom
    M = pd.read_csv(os.path.join(CMMD_DATA, "manifest_train.csv"))
    rng = np.random.default_rng(1); order = rng.permutation(len(M)); files = []
    for i in order:
        q = resolve(str(M["file"].iloc[i]))
        if os.path.exists(q):
            files.append(q)
        if len(files) >= N_SWEEP:
            break
    if not files:
        raise RuntimeError("no CMMD DICOM found: attach the CMMD dataset (tommyngx) as an input")
    say(f"  {len(files)} CMMD training DICOMs (random order, seed 1)")
    LEV = {"original": [], "half": [], "quarter": [], "256 px (DDPM)": []}; shapes = []; t0 = time.time()
    for j, f in enumerate(files):
        ds = pydicom.dcmread(f); a = ds.pixel_array.astype(np.float64)        # same conversion as prepare_cmmd.py
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
        if ds.pixel_array.dtype != np.uint8:
            a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255
        u8 = np.clip(np.round(a), 0, 255).astype(np.uint8); im = Image.fromarray(u8); W, H = im.size; shapes.append((H, W))
        for k, sz in zip(LEV, (None, (W // 2, H // 2), (W // 4, H // 4), (256, 256))):
            LEV[k].append(nlf_image(u8 if sz is None else np.array(im.resize(sz), dtype=np.uint8)))
        if j == 4:
            say(f"  ~{(time.time() - t0) / 5 * len(files) / 60:.0f} min expected for the sweep")
    say(f"  original sizes: median {int(np.median([s[0] for s in shapes]))} x {int(np.median([s[1] for s in shapes]))} px | "
        f"{(time.time() - t0) / 60:.1f} min")
    RES["B"] = {k: nlf_from_P(P) for k, P in LEV.items()}
    for k, r in RES["B"].items():
        say(nlf_line(f"CMMD {k}", r))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for c, (k, r) in zip(PAL, RES["B"].items()):
        V = np.array(r["V"])
        ax[0].plot(CEN, V, "-o", color=c, ms=3.5, lw=1.8, label=f"{k}  (linear R2 {r['linear'][1]:.2f})")
        ax[1].plot(CEN, V / V.max(), "-o", color=c, ms=3.5, lw=1.8, label=k)
    ax[0].set_yscale("log"); ax[0].set_ylabel("high-pass variance (log)"); ax[1].set_ylabel("variance / its maximum (shape)")
    for a_ in ax:
        a_.set_xlabel("local mean intensity (uint8 / 255)")
    ax[0].set_title("CMMD noise-level function vs resolution", loc="left"); ax[1].set_title("same curves, shape only", loc="left")
    ax[0].legend(frameon=False, fontsize=7.5); fig.savefig(f"{OUT}/S9_B_cmmd_resolution_sweep.png"); plt.close(fig)
except Exception as e:
    say(f"  !! part B skipped: {e!r}")

# ================================ load real + generated 256-px sets ================================
REAL = {"CBIS": trainfit(CBIS_DATA), "CMMD": trainfit(CMMD_DATA)}
SETS = {"CBIS": ["b40", "b60", "b90", "bm0", "bfp", "sm0", "snlf"], "CMMD": ["c40", "cm0", "cfp", "cavg", "c40_nl"]}
GEN = {ds: {n: gen(n) for n in L if gen(n) is not None} for ds, L in SETS.items()}
say("=" * 112)
for ds in REAL:
    say(f"loaded {ds}: real train-fit {len(REAL[ds])} images | generated sets {list(GEN[ds])}")
rs = np.random.default_rng(2); SUB = {ds: REAL[ds][rs.choice(len(REAL[ds]), min(400, len(REAL[ds])), replace=False)] for ds in REAL}

# ================================ PART C - measured noise curves ================================
say("=" * 112); say("C. MEASURED NOISE-LEVEL FUNCTIONS at 256 px (same estimator as the physics term; 95 % bootstrap CI over images)")
RES["C"] = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for a_, ds in zip(ax, ("CBIS", "CMMD")):
    show = ["b40", "b60"] if ds == "CBIS" else ["c40", "c40_nl", "cavg"]
    for nm, X in [("real", SUB[ds])] + list(GEN[ds].items()):
        r = nlf_set(X); RES["C"][f"{ds}_{nm}"] = r; say(nlf_line(f"{ds} {nm}", r))
        if nm == "real" or nm in show:
            V = np.array(r["V"]) * 1e3; col = C["real"] if nm == "real" else PAL[show.index(nm)]
            a_.errorbar(CEN, V, yerr=[V - np.array(r["lo"]) * 1e3, np.array(r["hi"]) * 1e3 - V], fmt="-o", ms=3.5, lw=1.6,
                        color=col, capsize=2, label=nm)
    a_.set_title(f"{ds}: high-pass variance vs intensity", loc="left"); a_.set_xlabel("local mean intensity (uint8 / 255)")
    a_.set_ylabel("variance (x 1e-3)"); a_.legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/S9_C_measured_noise_curves.png"); plt.close(fig)

# ================================ PART D - radial power spectrum ================================
say("=" * 112); say("D. RADIAL POWER SPECTRUM (per-image min-max, mean removed, Hann window; frequency in cycles / pixel at 256 px)")
fr = np.fft.fftfreq(256); FX, FY = np.meshgrid(fr, fr); RAD = np.sqrt(FX ** 2 + FY ** 2)
EDG = np.linspace(0, 0.5, 65); CF = 0.5 * (EDG[1:] + EDG[:-1]); IDX = (np.digitize(RAD.ravel(), EDG) - 1)
OK = (IDX >= 0) & (IDX < 64); CNTF = np.bincount(IDX[OK], minlength=64); HANN = np.outer(np.hanning(256), np.hanning(256))
def rps(X):
    acc = np.zeros(64)
    for im in mm01(X):
        p = (np.abs(np.fft.fft2((im - im.mean()) * HANN)) ** 2).ravel()
        acc += np.bincount(IDX[OK], weights=p[OK], minlength=64)
    return acc / np.maximum(CNTF, 1) / len(X)
def bands(p):
    tot = p.sum(); m = (CF > 0.02) & (CF < 0.4)
    return {"low": float(p[CF < 0.05].sum() / tot), "mid": float(p[(CF >= 0.05) & (CF < 0.2)].sum() / tot),
            "high": float(p[CF >= 0.2].sum() / tot), "slope": float(np.polyfit(np.log10(CF[m]), np.log10(p[m]), 1)[0])}
RES["D"] = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for a_, ds in zip(ax, ("CBIS", "CMMD")):
    pr = rps(SUB[ds]); b = bands(pr); RES["D"][f"{ds}_real"] = {"spectrum": pr.tolist(), **b}
    say(f"  {ds} real        : power share low {b['low']:.4f} | mid {b['mid']:.5f} | high {b['high']:.2e} | log-log slope {b['slope']:.2f}")
    for i, (nm, X) in enumerate(GEN[ds].items()):
        p = rps(X); bd = bands(p); RES["D"][f"{ds}_{nm}"] = {"spectrum": p.tolist(), **bd, "high_vs_real": float(p[CF >= 0.2].mean() / pr[CF >= 0.2].mean())}
        say(f"  {ds} {nm:10s}: power share low {bd['low']:.4f} | mid {bd['mid']:.5f} | high {bd['high']:.2e} | slope {bd['slope']:.2f} | "
            f"high-band power vs real x{RES['D'][f'{ds}_{nm}']['high_vs_real']:.2f}")
        a_.plot(CF, p / pr, color=PAL[i % len(PAL)], lw=1.6, label=nm)
    a_.axhline(1, color=C["real"], lw=1); a_.set_yscale("log"); a_.set_xlabel("spatial frequency (cycles / pixel)")
    a_.set_ylabel("power: generated / real"); a_.set_title(f"{ds}: radial power-spectrum ratio", loc="left"); a_.legend(frameon=False, fontsize=7.5)
fig.savefig(f"{OUT}/S9_D_power_spectra.png"); plt.close(fig)

# ================================ PART E - intensity distribution ================================
say("=" * 112); say("E. INTENSITY DISTRIBUTION of breast pixels (per-image min-max; breast = 7x7 local mean > 0.04; 2000 pixels / image)")
def istats(X):
    vals, sat = [], []; r = np.random.default_rng(0)
    for im in mm01(X):
        v = im[ndimage.uniform_filter(im, 7) > 0.04]; vals.append(r.choice(v, min(2000, len(v)), replace=False)); sat.append(float((v > 0.95).mean()))
    v = np.concatenate(vals)
    return v, {"p01": float(np.quantile(v, .01)), "p50": float(np.quantile(v, .5)), "p99": float(np.quantile(v, .99)),
               "mean": float(v.mean()), "sd": float(v.std()), "saturation": float(np.mean(sat))}
RES["E"] = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for a_, ds in zip(ax, ("CBIS", "CMMD")):
    for i, (nm, X) in enumerate([("real", SUB[ds])] + list(GEN[ds].items())):
        v, st = istats(X); RES["E"][f"{ds}_{nm}"] = st
        say(f"  {ds} {nm:10s}: p1 {st['p01']:.3f} p50 {st['p50']:.3f} p99 {st['p99']:.3f} | mean {st['mean']:.3f} sd {st['sd']:.3f} | "
            f"saturated (> 0.95) {100 * st['saturation']:.2f} %")
        xs = np.sort(v)
        a_.plot(xs, np.linspace(0, 1, len(xs)), color=C["real"] if nm == "real" else PAL[(i - 1) % len(PAL)], lw=2.2 if nm == "real" else 1.2, label=nm)
    a_.set_xlabel("intensity (per-image min-max)"); a_.set_ylabel("CDF of breast pixels"); a_.set_title(f"{ds}: intensity CDF", loc="left")
    a_.legend(frameon=False, fontsize=7.5)
fig.savefig(f"{OUT}/S9_E_intensity_cdf.png"); plt.close(fig)

# ================================ PART F - KID sample-size stability ================================
say("=" * 112); say("F. KID SAMPLE-SIZE STABILITY (paired subsamples of n generated images vs the full reference; 200 repeats)")
kern = lambda A, B: (A @ B.T / A.shape[1] + 1.0) ** 3                  # = s7_stats.kern
def krr(R):
    K = kern(R, R); return float((K.sum() - np.trace(K)) / (len(R) * (len(R) - 1)))
def kid(G, R, KRR):
    K = kern(G, G); n = len(G); return float((K.sum() - np.trace(K)) / (n * (n - 1)) - 2 * kern(G, R).mean() + KRR)
RES["F"] = {}
figF, axF = plt.subplots(1, 3, figsize=(13, 3.6)); fi = 0
for tag, pairs in (("S8_cbis_long_val", [("b40", "b60"), ("b40", "b90")]), ("S8_cmmd_main_val", [("c40", "cavg"), ("c40", "c40_nl")]),
                   ("S8_cmmd_fp_val", [("cm0", "cfp")])):
    f = os.path.join(S8, f"{tag}_features.npz")
    if not os.path.exists(f):
        say(f"  {tag}: features file missing - skipped"); continue
    Fd = np.load(f)
    for sp in ("mammo", "dino"):
        R = Fd[f"ref_{sp}"]; KRR = krr(R)
        for a_, b_ in pairs:
            A, B = Fd[f"{a_}_{sp}"], Fd[f"{b_}_{sp}"]; full = np.sign(kid(B, R, KRR) - kid(A, R, KRR)); line, ns, kept = [], [], []
            for n in (25, 50, 100, 150, 200):
                if n > min(len(A), len(B)):
                    continue
                rng = np.random.default_rng(n); ka, kb, agree = [], [], 0
                for _ in range(200):
                    i = rng.choice(min(len(A), len(B)), n, replace=False)          # the same noise-bank images in both sets
                    x, y = kid(A[i], R, KRR), kid(B[i], R, KRR); ka.append(x); kb.append(y); agree += int(np.sign(y - x) == full)
                RES["F"][f"{tag}_{sp}_{a_}_vs_{b_}_n{n}"] = {"kid_a": [float(np.mean(ka)), float(np.std(ka))],
                                                            "kid_b": [float(np.mean(kb)), float(np.std(kb))], "order_kept": agree / 200}
                line.append(f"n {n}: {a_} {np.mean(ka):.3f}+-{np.std(ka):.3f} {b_} {np.mean(kb):.3f}+-{np.std(kb):.3f} order kept {agree / 2:.0f} %")
                ns.append(n); kept.append(100 * agree / 200)
            say(f"  {tag} {sp:5s} {a_} vs {b_}: " + " | ".join(line))
            if sp == "mammo" and fi < 3:
                axF[fi].plot(ns, kept, "-o", color=PAL[fi], lw=1.8); axF[fi].set_ylim(40, 102); axF[fi].axhline(50, color=C["real"], lw=0.8)
                axF[fi].set_title(f"{a_} vs {b_} (KID-mammo)", loc="left"); axF[fi].set_xlabel("generated images per set")
                axF[fi].set_ylabel("% subsamples with the full-set order"); fi += 1
figF.savefig(f"{OUT}/S9_F_kid_sample_size.png"); plt.close(figF)

# ================================ PART G - model table ================================
say("=" * 112); say("G. MODEL TABLE (validation features saved by S8; exploratory, no checkpoint is selected here)")
RES["G"] = {}
for tag in ("S8_cbis_long_val", "S8_cbis_fp_val", "S8_cbis_sep_val", "S8_cmmd_main_val", "S8_cmmd_fp_val"):
    f = os.path.join(S8, f"{tag}_features.npz")
    if not os.path.exists(f):
        say(f"  {tag}: features file missing - skipped"); continue
    Fd = np.load(f); names = [k[:-6] for k in Fd.files if k.endswith("_mammo") and not k.startswith("ref")]
    REFK = {sp: (Fd[f"ref_{sp}"], krr(Fd[f"ref_{sp}"])) for sp in ("mammo", "dino", "incep") if f"ref_{sp}" in Fd.files}
    ds = "CBIS" if "cbis" in tag else "CMMD"
    for nm in names:
        row = {sp: kid(Fd[f"{nm}_{sp}"], *REFK[sp]) for sp in REFK if f"{nm}_{sp}" in Fd.files}
        st = Fd[f"{nm}_istats"]; row.update({"saturation": float(st[:, 1].mean()), "hf": float(st[:, 2].mean()), "tag": tag})
        d_ = RES["D"].get(f"{ds}_{nm}"); row["high_band_vs_real"] = d_["high_vs_real"] if d_ else None
        RES["G"][nm] = row
        say(f"  {tag:17s} {nm:7s}: KID-mammo {row.get('mammo', float('nan')):.4f} | KID-DINOv2 {row.get('dino', float('nan')):.3f} | "
            f"KID-Inception {row.get('incep', float('nan')):.4f} | saturated {100 * row['saturation']:.2f} % | HF {row['hf']:.5f}"
            + (f" | high-band power vs real x{row['high_band_vs_real']:.2f}" if row["high_band_vs_real"] else ""))

# ================================ PART I - FP gradient conditioning by timestep ================================
say("=" * 112); say("I. FP GRADIENT CONDITIONING (S8 training logs): |lambda g_FP| / |g_base|, cosine(g_base, g_FP), clipping, by timestep")
RES["I"] = {}
for f in sorted(glob.glob(os.path.join(S8, "train_logs", "*_fp_log.jsonl"))):
    arm = os.path.basename(f).replace("_fp_log.jsonl", "")
    recs = [json.loads(l) for l in open(f) if l.strip()]
    if not recs:
        continue
    g = lambda k: np.array([np.nan if r.get(k) is None else r.get(k) for r in recs], dtype=float)
    t, bn, fw, cs, cf = g("t"), g("base_norm"), g("fp_norm_weighted"), g("cos_base_fp"), g("clip_factor")
    out = {"steps": len(recs), "clipped_share": float(np.nanmean(cf < 1)), "base_norm_median": float(np.nanmedian(bn))}
    line = f"  {arm:10s}: {len(recs)} optimizer steps | clipped {100 * out['clipped_share']:.0f} % | median |g_base| {out['base_norm_median']:.2f}"
    if np.isfinite(fw).any():
        out.update({"ratio_median": float(np.nanmedian(fw / bn)), "cos_mean": float(np.nanmean(cs)), "cos_positive_share": float(np.nanmean(cs[np.isfinite(cs)] > 0))})
        line += f" | median ratio {out['ratio_median']:.2f} | mean cos {out['cos_mean']:+.3f} (positive in {100 * out['cos_positive_share']:.0f} %)"
        for lo, hi in ((10, 50), (50, 100), (100, 200), (200, 301)):
            m = (t >= lo) & (t < hi) & np.isfinite(fw)
            if m.sum() > 5:
                out[f"t{lo}-{hi - 1}"] = {"ratio_median": float(np.nanmedian(fw[m] / bn[m])), "cos_mean": float(np.nanmean(cs[m])), "n": int(m.sum())}
                line += f"\n{'':14s}t {lo}-{hi - 1}: ratio {out[f't{lo}-{hi - 1}']['ratio_median']:.2f}, cos {out[f't{lo}-{hi - 1}']['cos_mean']:+.3f} (n {m.sum()})"
    RES["I"][arm] = out; say(line)
if not RES["I"]:
    say("  no *_fp_log.jsonl found in S8/train_logs - skipped")

json.dump(RES, open(f"{OUT}/S9_results.json", "w"), indent=1, default=float)
open(f"{OUT}/S9_PLUS_report.txt", "w").write("\n".join(REP) + "\n")
say("S9 DONE ->", sorted(os.listdir(OUT)))
