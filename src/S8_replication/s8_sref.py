# s8_sref.py - S_ref(t) table for a given checkpoint, computed EXACTLY as in S2 (a8_fp_clean.py, section 2):
# RMS of u . d_tau s on 6 validation images (3 / class, rng seed 0), 3 random unit directions u (torch generator seed 11,
# drawn in the same order as A8: U, V, NOISE), t grid 10..300. Output: {"S_ref_table": {...}} for train_fp.py.
import os, sys, json, argparse, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, to_model_domain, load_model, OUT
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)                       # file name inside $PA_OUT
ap.add_argument("--n_img", type=int, default=6)
ap.add_argument("--n_u", type=int, default=3)
ap.add_argument("--n_v", type=int, default=4)
ap.add_argument("--sref_grid", default="10,15,20,30,50,75,100,150,200,250,300")
a = ap.parse_args()
TAG = "S8-sref"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
unet, ac, meta = load_model(a.ckpt, dev)
xtr, ytr, pat, _ = load_clean_meta("train"); vm = val_mask(pat)
xv, yv = xtr[vm], ytr[vm]
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(yv == c)[0], a.n_img // 2, replace=False) for c in (0, 1)])
X0 = to_model_domain(xv[idx], dev); LAB = [int(v) for v in yv[idx]]
gen = torch.Generator(device=dev).manual_seed(11)
H, W = X0.shape[-2:]
U = torch.randn((a.n_u, 1, H, W), generator=gen, device=dev); U = U / U.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
V = (torch.randint(0, 2, (a.n_v, 1, H, W), generator=gen, device=dev) * 2 - 1).float()
NOISE = torch.randn(X0.shape, generator=gen, device=dev)
SF = {lab: fc.make_score_fn(unet, ac, lab) for lab in (0, 1)}
sref = {}
with torch.no_grad():
    for t in [int(v) for v in a.sref_grid.split(",")]:
        ab = float(ac[t]); vals = []
        for i in range(X0.shape[0]):
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i:i + 1]
            d = (SF[LAB[i]](x, t + 1) - SF[LAB[i]](x, t - 1)) / (2 * fc.DTAU)
            vals += [float((U[j:j + 1] * d).sum()) for j in range(a.n_u)]
        sref[str(t)] = float(np.sqrt(np.mean(np.square(vals))))
log(TAG, f"{a.ckpt} (epoch {meta.get('epoch')}) | S_ref(t): " + ", ".join(f"{k}: {v:.3g}" for k, v in sref.items()))
save_json({"checkpoint": a.ckpt, "epoch": meta.get("epoch"), "S_ref_table": sref}, a.out)
