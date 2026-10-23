# a3_texture.py - A3 (minimal, CPU): is the measured "noise curve" noise or tissue texture, and how much detail
# survives at 256 x 256?
# PRE-STATED RULES:
#   resolution test: halving the resolution of pure white-like noise divides its high-pass variance by ~4.
#     ratio V(half res)/V(full res) in FLAT breast regions <= 0.35 -> "noise-like";  >= 0.70 -> "texture-dominated";
#     otherwise "mixed".
#   calibration: noise with a KNOWN curve is added to heavily smoothed real images; the estimator must recover it
#     up to the analytically known attenuation of the 7-tap high-pass filter (reported as recovered/expected).
import os, sys, json, subprocess
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache
from pinn_physics import PhysicsNLF

TAG = "A3"
rng = np.random.default_rng(0)
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
RAW = os.path.dirname(os.path.dirname(hit[0]))
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
jp = lambda rel: os.path.join(RAW, "jpeg", rel.replace("CBIS-DDSM/jpeg/", ""))
full = info[info["SeriesDescription"] == "full mammogram images"]
paths = [jp(p) for p in full["image_path"].tolist() if os.path.exists(jp(p))]
rep = {}
nlf = PhysicsNLF(torch.device("cpu"))

# ---- (a) native resolution of the JPEG files actually used ----
sel = rng.choice(len(paths), min(60, len(paths)), replace=False)
sizes = []
for i in sel:
    with Image.open(paths[i]) as im:
        sizes.append(im.size)                                         # (W, H)
W = np.array([s[0] for s in sizes]); H = np.array([s[1] for s in sizes])
rep["native_size_median_WxH"] = [int(np.median(W)), int(np.median(H))]
rep["downsample_factor_to_256_median"] = [float(np.median(W) / 256), float(np.median(H) / 256)]
rep["jpeg_mode_examples"] = list({Image.open(paths[i]).mode for i in sel[:10]})

# ---- (b) resolution test in flat and all breast regions ----
def hp_var(img01, flat_q=0.3):
    I = torch.from_numpy(img01).float()[None, None]
    h = I - nlf.blur(I)
    mu = torch.nn.functional.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    gy, gx = torch.gradient(nlf.blur(I)[0, 0])
    gm = (gx ** 2 + gy ** 2).sqrt()
    breast = (mu[0, 0] > 0.04)
    if breast.sum() < 100:
        return None
    gb = gm[breast].numpy()
    thr = float(np.quantile(gb[np.random.default_rng(1).choice(len(gb), min(len(gb), 2_000_000), replace=False)], flat_q))
    flat = breast & (gm <= thr)
    hh = h[0, 0] ** 2
    return float(hh[flat].mean()), float(hh[breast].mean())

ratios = {"flat_512_over_1024": [], "flat_256_over_512": [], "all_512_over_1024": [], "all_256_over_512": []}
for i in sel[:40]:
    g = Image.open(paths[i]).convert("L")
    v = {}
    for r in (1024, 512, 256):
        a = np.asarray(g.resize((r, r)), dtype=np.float32)          # same PIL default resize as prepare.py
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        v[r] = hp_var(a)
    if any(x is None for x in v.values()):
        continue
    ratios["flat_512_over_1024"].append(v[512][0] / v[1024][0]); ratios["flat_256_over_512"].append(v[256][0] / v[512][0])
    ratios["all_512_over_1024"].append(v[512][1] / v[1024][1]); ratios["all_256_over_512"].append(v[256][1] / v[512][1])
rep["resolution_ratios_median"] = {k: float(np.median(x)) for k, x in ratios.items() if x}
rf = rep["resolution_ratios_median"].get("flat_256_over_512", np.nan)
rep["resolution_verdict"] = ("noise-like" if rf <= 0.35 else "texture-dominated" if rf >= 0.70 else "mixed") \
    if np.isfinite(rf) else "not computed"

# ---- (c) region test at 256: binned high-pass variance, flat vs edge pixels ----
x, y = load_cache("train")
idx = rng.choice(len(x), min(300, len(x)), replace=False)
I = torch.from_numpy(x[idx]).float().unsqueeze(1)
mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True)
I = (I - mn) / (mx - mn + 1e-8)
h = I - nlf.blur(I)
mu = torch.nn.functional.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
v = torch.nn.functional.avg_pool2d(h * h, 7, 1, 3, count_include_pad=False)
B = nlf.blur(I)
gx = B[..., :, 2:] - B[..., :, :-2]; gy = B[..., 2:, :] - B[..., :-2, :]
gm = torch.zeros_like(I); gm[..., 1:-1, 1:-1] = (gx[..., 1:-1, :] ** 2 + gy[..., :, 1:-1] ** 2).sqrt()
breast = mu > 0.04
_g = gm[breast].numpy(); _g = _g[rng.choice(len(_g), min(len(_g), 2_000_000), replace=False)]
thr_lo, thr_hi = float(np.quantile(_g, 0.3)), float(np.quantile(_g, 0.7))
edges = np.linspace(0.08, 0.95, 17)
curve = {"flat": [], "edge": [], "all": []}
for name, m in (("flat", breast & (gm <= thr_lo)), ("edge", breast & (gm >= thr_hi)), ("all", breast)):
    mu_m, v_m = mu[m].numpy(), v[m].numpy()
    b = np.digitize(mu_m, edges) - 1
    curve[name] = [float(v_m[b == k].mean()) if np.sum(b == k) > 50 else None for k in range(16)]
rep["region_curves_bin_edges"] = edges.tolist()
rep["region_curves"] = curve
ratio = [e / f for e, f in zip(curve["edge"], curve["flat"]) if e and f]
rep["edge_over_flat_variance_median"] = float(np.median(ratio)) if ratio else None

# ---- (d) synthetic calibration: known noise curve on smoothed real backgrounds ----
b_, a_, c_ = 1.784e-3, -4.218e-3, 3.308e-3                       # the real Experiment-2 fit
g = nlf.g.numpy(); g2 = np.outer(g, g); delta = np.zeros((7, 7)); delta[3, 3] = 1
atten = float(((delta - g2) ** 2).sum())                          # fraction of white-noise variance kept by I - blur(I)
bg = torch.nn.functional.avg_pool2d(I[:50], 9, 1, 4, count_include_pad=False)
bg = nlf.blur(nlf.blur(bg))                                        # heavily smoothed "clean" background
sig2 = (b_ + a_ * bg + c_ * bg ** 2).clamp_min(1e-7)
noisy = bg + torch.randn(bg.shape, generator=torch.Generator().manual_seed(0)) * sig2.sqrt()
vb_clean, m_clean = nlf.stats(bg)
vb_noisy, m_noisy = nlf.stats(noisy)
cent = nlf.c.numpy(); known = b_ + a_ * cent + c_ * cent ** 2
rec = (vb_noisy * m_noisy).sum(0) / m_noisy.sum(0).clamp_min(1e-6)
cle = (vb_clean * m_clean).sum(0) / m_clean.sum(0).clamp_min(1e-6)
ok = (m_noisy.sum(0) > 1000).numpy()
rep["calibration_highpass_attenuation_expected"] = atten
rep["calibration_recovered_over_known_median"] = float(np.median((rec.numpy() / known)[ok]))
rep["calibration_recovered_over_expected_median"] = float(np.median((rec.numpy() / (known * atten))[ok]))
rep["calibration_texture_leak_clean_over_known_median"] = float(np.median((cle.numpy() / known)[ok]))

# ---- (e) lesion size at 256 x 256 from ROI masks ----
try:
    masks = info[info["SeriesDescription"] == "ROI mask images"].set_index("PatientName")["image_path"].to_dict()
    les = []
    for c in ("csv/mass_case_description_train_set.csv", "csv/calc_case_description_train_set.csv"):
        df = pd.read_csv(os.path.join(RAW, c))
        mcol = [k for k in df.columns if k.lower().strip() == "roi mask file path"][0]
        typ = "mass" if "mass" in c else "calc"
        for p in df[mcol].sample(min(100, len(df)), random_state=0):
            rel = masks.get(str(p).split("/")[0])
            if rel is None or not os.path.exists(jp(rel)):
                continue
            m = np.asarray(Image.open(jp(rel)).convert("L"))
            if np.mean((m > 30) & (m < 225)) > 0.02:                 # not a binary mask (dataset mislabel) -> skip
                continue
            mb = m > 127
            if mb.sum() == 0:
                continue
            ys, xs = np.where(mb)
            Hn, Wn = m.shape
            les.append({"type": typ, "area_px_256": float(mb.mean() * 256 * 256),
                        "bbox_h_256": float((ys.max() - ys.min() + 1) * 256 / Hn),
                        "bbox_w_256": float((xs.max() - xs.min() + 1) * 256 / Wn)})
    L = pd.DataFrame(les)
    out = {}
    for typ, g_ in L.groupby("type"):
        d = 2 * np.sqrt(g_.area_px_256 / np.pi)
        out[typ] = {"n": int(len(g_)), "equiv_diameter_px_median": float(d.median()),
                    "equiv_diameter_px_q10_q90": [float(d.quantile(0.1)), float(d.quantile(0.9))],
                    "bbox_min_side_px_median": float(np.minimum(g_.bbox_h_256, g_.bbox_w_256).median()),
                    "fraction_diameter_below_5px": float((d < 5).mean())}
    rep["lesion_size_at_256"] = out
except Exception as e:
    rep["lesion_size_at_256"] = f"skipped: {repr(e)[:200]}"

save_json(rep, "A3_texture.json")
for k, v in rep.items():
    log(TAG, f"{k}: {v}")
