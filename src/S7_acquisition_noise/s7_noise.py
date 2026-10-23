# s7_noise.py - S7 "physics-based acquisition layer" (see S7_preregistration.txt + Addendum 1; nothing here is tuned).
#   1. fit the Poisson-Gaussian noise-level function sigma^2(I) = a*I + b on the train-fit REAL images
#      (pinn_physics.PhysicsNLF, linear fit, per-image min-max normalised images - exactly as in S5)
#   2. for every generated set: I = per-image min-max to [0,1]; I_out = clip(I + sqrt(max(a*I + b, 0)) * z, 0, 1),
#      z ~ N(0,1) from one fixed seed; rounded back to uint8 -> gen_<name><suffix>.npz
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, OUT
from pinn_physics import PhysicsNLF

ap = argparse.ArgumentParser()
ap.add_argument("--sets", required=True)            # comma list of gen_<name>.npz names
ap.add_argument("--seed", type=int, required=True)  # noise seed (validation 7007, test 9707)
ap.add_argument("--suffix", default="_nl")
a = ap.parse_args()
TAG = "S7-noise"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mm01(X_u8):
    X = X_u8.astype(np.float64); mn = X.min(axis=(1, 2), keepdims=True); mx = X.max(axis=(1, 2), keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

xtr, ytr, pat, _ = load_clean_meta("train"); XF = xtr[~val_mask(pat)]
nlf = PhysicsNLF(dev)
rep = nlf.fit(torch.from_numpy(np.round(mm01(XF) * 255).clip(0, 255).astype(np.uint8)))
lin = rep["linear_poisson_gaussian_only"]; A, B = lin["a"], lin["b"]
log(TAG, f"noise-level function fitted on {len(XF)} train-fit images: sigma^2 = {A:.4e} * I + {B:.4e} "
         f"(linear Poisson-Gaussian, R2 {lin['R2_var']:.3f}) | quadratic fit for reference: b {rep['b']:.3e} a {rep['a']:.3e} c {rep['c']:.3e}")
if A < 0:
    log(TAG, "NOTE: fitted Poisson coefficient a < 0 (variance falls with brightness) - used as fitted (pre-stated), sigma^2 clamped at 0")
save_json({"a": A, "b": B, "fit": rep, "seed": a.seed}, f"S7_nlf_seed{a.seed}.json")

rng = np.random.default_rng(a.seed)
for nm in a.sets.split(","):
    d = np.load(os.path.join(OUT, f"gen_{nm}.npz"))
    I = mm01(d["x"])
    sig = np.sqrt(np.maximum(A * I + B, 0.0))
    Io = np.clip(I + sig * rng.standard_normal(I.shape), 0.0, 1.0)
    Xo = np.round(Io * 255).astype(np.uint8)
    np.savez_compressed(os.path.join(OUT, f"gen_{nm}{a.suffix}.npz"), x=Xo, y=d["y"])
    log(TAG, f"{nm} -> gen_{nm}{a.suffix}.npz | {len(Xo)} images | mean added noise std {sig.mean():.4f} "
             f"(= {255 * sig.mean():.1f} grey levels)")
