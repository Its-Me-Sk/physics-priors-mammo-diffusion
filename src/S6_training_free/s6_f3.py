# s6_f3.py - per-image denoising-error score F3 (same definition as S5): eps-MSE of the epoch-40 baseline on the image
# noised at t = 50, 200, 500 (fixed noise per image index, fp16 autocast, true label), each t divided by the median over
# the 347 real validation images, averaged over t. LOWER = more typical for the model (direction fixed in S5).
# --select a,b,...: best-of-N - for every slot i (same label in every candidate set) keep the candidate with the lowest F3
# and write gen_<out>.npz.
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, minmax_pm1, load_model, eps_pred, ctx_of, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sets", required=True)             # comma list of gen_<name>.npz names (without gen_ / .npz)
ap.add_argument("--select_out", default="")          # if given: best-of-N over --sets -> gen_<select_out>.npz
ap.add_argument("--seed", type=int, default=77)
a = ap.parse_args()
TAG = "S6-F3"
dev = torch.device("cuda")
unet, ac, _ = load_model(a.ckpt, dev); acf = ac.float()
MT = [50, 200, 500]

@torch.no_grad()
def mse(X, Y, seed):
    g = torch.Generator().manual_seed(seed); out = np.zeros((len(Y), len(MT)))
    noise = [torch.randn((len(Y), 1) + X.shape[1:], generator=g) for _ in MT]
    for i in range(0, len(Y), 16):
        x0 = minmax_pm1(torch.from_numpy(X[i:i + 16]).float().unsqueeze(1)).to(dev); k = x0.shape[0]
        for ti, t in enumerate(MT):
            eps = noise[ti][i:i + k].to(dev); ab = acf[t]
            eh = eps_pred(unet, ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, ctx_of(Y[i:i + k], dev), amp=True, chunk=8)
            out[i:i + k, ti] = ((eh - eps) ** 2).mean((1, 2, 3)).cpu().numpy()
    return out

xtr, ytr, pat, _ = load_clean_meta("train"); vm = val_mask(pat)
med = np.median(mse(xtr[vm], ytr[vm], a.seed), 0)
names = a.sets.split(","); scores, data = {}, {}
for nm in names:
    d = np.load(os.path.join(OUT, f"gen_{nm}.npz")); data[nm] = d
    sc = (mse(d["x"], d["y"], a.seed + 1) / med).mean(1)       # same noise for slot i in every set (paired)
    scores[nm] = sc; np.save(os.path.join(OUT, f"f3_{nm}.npy"), sc)
    log(TAG, f"{nm}: F3 mean {sc.mean():.4f} median {np.median(sc):.4f}")
if a.select_out:
    ys = [data[n]["y"] for n in names]
    assert all((y == ys[0]).all() for y in ys), "candidate sets must have the same label per slot"
    S = np.stack([scores[n] for n in names]); best = S.argmin(0)
    X = np.stack([data[n]["x"] for n in names])[best, np.arange(len(best))]
    np.savez_compressed(os.path.join(OUT, f"gen_{a.select_out}.npz"), x=X, y=ys[0])
    save_json({"name": a.select_out, "candidates": names, "chosen_share": np.bincount(best, minlength=len(names)).tolist()},
              f"gen_{a.select_out}.json")
    log(TAG, f"best-of-{len(names)} -> gen_{a.select_out}.npz | chosen per candidate set {np.bincount(best, minlength=len(names)).tolist()}")
