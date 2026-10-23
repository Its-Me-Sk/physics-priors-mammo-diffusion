# s5_score.py - S5 per-image scores computed with the epoch-40 clean baseline (no training):
#   F1 (physics: score Fokker-Planck consistency) - each image x0 (generated or real) is noised to x_t at t in --t; the
#      projected score-FPE residual r(u,v) of the MODEL at x_t is measured with the validated fp32 finite-difference
#      estimator (fp_core, step table p(t)), one unit direction u and two Rademacher probes, with the image's class label:
#      P(t) = r1 r2 / S_ref(t)^2 (unbiased for (u.R)^2 / S_ref^2). Per-image draws come from a fixed seed per image.
#   F3 (non-physics: the model's own denoising error) - eps-MSE of the model on the image noised at t in --mse_t
#      (fixed noise per image, fp16 autocast exactly as in training, true label).
# Sets: real = the 347 validation images; pool = a generated set (gen_*.npz); retest = the first --n_retest pool images
# scored again with an independent seed (test-retest reliability of F1).
import os, sys, time, json, math, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, minmax_pm1, load_model, find, eps_pred, ctx_of, OUT
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["real", "pool", "retest"], required=True)
ap.add_argument("--gen", default="")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--t", default="10,50")
ap.add_argument("--mse_t", default="50,200,500")
ap.add_argument("--n_retest", type=int, default=80)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
TAG = f"S5-score-{a.set}"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
TT = [int(v) for v in a.t.split(",")]; MT = [int(v) for v in a.mse_t.split(",")]
SREF = json.load(open(find("A8_fp_clean.json")[0]))["S_ref_table"]

if a.set == "real":
    xtr, ytr, pat, _ = load_clean_meta("train"); vm = val_mask(pat)
    X, Y = xtr[vm], ytr[vm]
else:
    d = np.load(a.gen); X, Y = d["x"], d["y"]
    if a.set == "retest":
        X, Y = X[:a.n_retest], Y[:a.n_retest]
if a.limit:
    X, Y = X[:a.limit], Y[:a.limit]
seed = a.seed + (1 if a.set == "retest" else 0)
unet, ac, meta = load_model(a.ckpt, dev)                    # fp32, eval, no grad
LOGAB = torch.log(ac.cpu())
SF = {lab: fc.make_score_fn(unet, ac, lab, chunk=12) for lab in (0, 1)}
n = len(Y); H_, W_ = X.shape[1:]
log(TAG, f"{n} images | FP t {TT} | denoising-error t {MT} | model {a.ckpt} (epoch {meta.get('epoch')}) | seed {seed}")

R = np.zeros((n, len(TT), 2)); A_ = np.zeros((n, len(TT)))
t0 = time.time()
for i in range(n):
    x0 = minmax_pm1(torch.from_numpy(X[i:i + 1]).float().unsqueeze(1)).to(dev)
    g = torch.Generator().manual_seed(seed * 1_000_003 + i)
    for ti, t in enumerate(TT):
        ab = float(ac[t]); p = fc.p_of_t(t)
        nz = torch.randn((1, 1, H_, W_), generator=g); u = torch.randn((1, 1, H_, W_), generator=g); u = u / u.norm()
        vs = [(torch.randint(0, 2, (1, 1, H_, W_), generator=g) * 2 - 1).float().to(dev) for _ in range(2)]
        x = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * nz.to(dev); u = u.to(dev)
        s = fc.evaluate_nograd(SF[int(Y[i])], fc.plan_n(x, t, u, vs, p))
        rs, A, _ = fc.residuals_n(s, x, u, vs, p, 1, fc.beta_eff(LOGAB, t))
        R[i, ti] = [float(r) for r in rs]; A_[i, ti] = float(A)
    if (i + 1) % 50 == 0 or i == n - 1:
        log(TAG, f"  FP score {i + 1}/{n} | {(time.time() - t0) / 60:.1f} min")
sr = np.array([fc.sref_interp(SREF, t) for t in TT])
P = R[:, :, 0] * R[:, :, 1] / sr[None, :] ** 2

MSE = np.zeros((n, len(MT)))
gm = torch.Generator().manual_seed(seed * 7 + 3)
acf = ac.float()
with torch.no_grad():
    for i in range(0, n, 16):
        x0 = minmax_pm1(torch.from_numpy(X[i:i + 16]).float().unsqueeze(1)).to(dev); k = x0.shape[0]
        for ti, t in enumerate(MT):
            eps = torch.randn(x0.shape, generator=gm).to(dev); ab = acf[t]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            eh = eps_pred(unet, xt, t, ctx_of(Y[i:i + k], dev), amp=True, chunk=8)
            MSE[i:i + k, ti] = ((eh - eps) ** 2).mean((1, 2, 3)).cpu().numpy()
np.savez(os.path.join(OUT, f"score_{a.set}.npz"), R=R, A=A_, P=P, H=P.mean(1), MSE=MSE, y=Y, t=np.array(TT), mse_t=np.array(MT))
save_json({"set": a.set, "n": n, "t": TT, "mse_t": MT, "seed": seed, "minutes": (time.time() - t0) / 60,
           "H_mean": float(P.mean()), "P_mean_per_t": P.mean(0).tolist()}, f"score_{a.set}.json")
log(TAG, f"done: H mean {P.mean():.4f} | per t " + " ".join(f"{t}:{v:.3f}" for t, v in zip(TT, P.mean(0)))
         + f" | denoising MSE per t " + " ".join(f"{t}:{v:.5f}" for t, v in zip(MT, MSE.mean(0))) + f" | {(time.time() - t0) / 60:.1f} min")
