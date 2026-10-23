# a2_jensen.py - A2: conditional-mean ("Jensen") test on the REAL trained network.
# For real test images x0 and noise levels t: x_t -> (i) x0_hat = the network's one-step estimate (what the
# Experiment 2-5 physics loss was applied to), (ii) Bastek-style sample estimate (deterministic DDIM jump to t=0, then
# x0_hat there = the "sample estimate" of Bastek et al., ICLR 2025, App. A.4: one DDIM step t -> least-noisy step, then
# x0_hat; two network passes), (iii) K stochastic samples of the network's OWN reverse process started at x_t (exact DDPM ancestral
# sampling, conditional label, no guidance). Statistics on all of them, ladder from linear to strongly nonlinear:
#   L1 mean intensity (breast) | L2 high-pass variance | L3 log L2 | L4 the training physics residual (PhysicsNLF)
# PRE-STATED t* RULE: band = 10-90 % quantiles of L4 on ALL real test images; f_out(t) = share of x0_hat outside the
#   band; t* = the largest grid t such that f_out(t') <= f_out(real) + 0.10 for every grid t' <= t (t* = 0 if t=5 fails).
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache, to_model_domain, ctx_of, eps_pred, load_model, main_ckpt, \
    optional_25_ckpt
from pinn_physics import PhysicsNLF

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="main")            # main | 25
ap.add_argument("--n_img", type=int, default=16)
ap.add_argument("--k", type=int, default=4)              # samples per image for t <= 100
ap.add_argument("--k_hi", type=int, default=2)           # samples per image for t > 100 (exact sampling costs t passes)
ap.add_argument("--grid", default="5,10,20,50,100,200,300,500")
a = ap.parse_args()
TAG = f"A2-{a.which}"
dev = torch.device("cuda")
path = main_ckpt() if a.which == "main" else optional_25_ckpt()
if path is None:
    log(TAG, "no 25 %-data checkpoint found -> skipped"); raise SystemExit(0)
unet, ac, meta = load_model(path, dev)
log(TAG, f"checkpoint {path} | epoch {meta.get('epoch')}")
nlf = PhysicsNLF(dev)
if meta["nlf"] is not None:
    nlf.load(meta["nlf"])
else:
    raise SystemExit("snapshot has no NLF state")
betas = 1 - ac / torch.cat([torch.ones(1, device=dev, dtype=ac.dtype), ac[:-1]])

def ladder(x_pm1):
    I = ((x_pm1 + 1) / 2).clamp(0, 1).float()
    h = I - nlf.blur(I)
    mu = torch.nn.functional.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    m = (mu > 0.04).float()
    n = m.sum((1, 2, 3)).clamp_min(1)
    L1 = (I * m).sum((1, 2, 3)) / n
    L2 = (h * h * m).sum((1, 2, 3)) / n
    L3 = torch.log(L2 + 1e-12)
    L4 = nlf.residual(I)
    return torch.stack([L1, L2, L3, L4], 1).cpu().numpy()

@torch.no_grad()
def x0hat(x, t, ctx):
    e = eps_pred(unet, x, t, ctx, amp=True)
    ab = ac[t].float()
    return ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1), e

@torch.no_grad()
def ancestral(x, t, ctx, gen):
    """exact DDPM reverse process (FIXED_SMALL variance, clip like the MONAI scheduler) from step t down to 0"""
    for s in range(t, 0, -1):
        x0, _ = x0hat(x, s, ctx)
        ab, abp, b = ac[s].float(), ac[s - 1].float(), betas[s].float()
        mean = (abp.sqrt() * b / (1 - ab)) * x0 + ((1 - b).sqrt() * (1 - abp) / (1 - ab)) * x
        var = ((1 - abp) / (1 - ab) * b).clamp_min(1e-20)
        x = mean + var.sqrt() * torch.randn(x.shape, generator=gen, device=dev)
    return x0hat(x, 0, ctx)[0]

xte, yte = load_cache("test")
real_all = np.concatenate([ladder(to_model_domain(xte[i:i + 32], dev)) for i in range(0, len(xte), 32)])
lo, hi = np.quantile(real_all[:, 3], [0.1, 0.9]); sd = real_all.std(0); mu_r = real_all.mean(0)
f_real = float(np.mean((real_all[:, 3] < lo) | (real_all[:, 3] > hi)))
r = np.random.default_rng(0)
idx = np.concatenate([r.choice(np.where(yte == c)[0], a.n_img // 2, replace=False) for c in (0, 1)])
X0 = to_model_domain(xte[idx], dev); lab = yte[idx]
grid = [int(g) for g in a.grid.split(",")]
gen = torch.Generator(device=dev).manual_seed(123)
rows = []
t0 = time.time()
for t in grid:
    ab = ac[t].float()
    eps = torch.randn(X0.shape, generator=gen, device=dev)
    xt = ab.sqrt() * X0 + (1 - ab).sqrt() * eps
    ctx = ctx_of(lab, dev)
    m_hat, e_hat = x0hat(xt, t, ctx)
    # Bastek sample estimate: deterministic DDIM jump x_t -> step 0, then x0_hat at step 0
    x_0 = ac[0].float().sqrt() * m_hat + (1 - ac[0].float()).sqrt() * e_hat
    bast = x0hat(x_0, 0, ctx)[0]
    K = a.k if t <= 100 else a.k_hi
    samples = []
    for k in range(K):
        samples.append(ancestral(xt.clone(), t, ctx, gen))
    S = torch.stack(samples, 1)                                  # (N, K, 1, H, W)
    Ls = np.stack([ladder(S[:, k]) for k in range(K)], 1)        # (N, K, 4)
    Lm, Lb, Lr = ladder(m_hat), ladder(bast), ladder(X0)
    spread = float(((S - S.mean(1, keepdim=True)) ** 2).mean())  # posterior variance per pixel (empirical)
    gapmean = float(((m_hat - S.mean(1)) ** 2).mean())
    out = lambda L4: float(np.mean((L4 < lo) | (L4 > hi)))
    rows.append({"t": t, "abar": float(ab), "k_samples": K,
                 "gap_sd_xhat": ((Lm.mean(0) - mu_r) / sd).tolist(),
                 "gap_sd_bastek": ((Lb.mean(0) - mu_r) / sd).tolist(),
                 "gap_sd_samples": ((Ls.reshape(-1, 4).mean(0) - mu_r) / sd).tolist(),
                 "gap_sd_real_subset": ((Lr.mean(0) - mu_r) / sd).tolist(),
                 "jensen_S_of_mean_minus_mean_of_S_sd": ((Lm - Ls.mean(1)).mean(0) / sd).tolist(),
                 "out_band_xhat": out(Lm[:, 3]), "out_band_bastek": out(Lb[:, 3]),
                 "out_band_samples": out(Ls[:, :, 3].ravel()), "out_band_real_subset": out(Lr[:, 3]),
                 "posterior_var_per_px": spread, "sq_dist_xhat_to_sample_mean_per_px": gapmean})
    log(TAG, f"t={t:4d} out-of-band x0_hat {rows[-1]['out_band_xhat']:.2f} | samples {rows[-1]['out_band_samples']:.2f} | "
             f"bastek {rows[-1]['out_band_bastek']:.2f} | L4 gap (SD) x0_hat {rows[-1]['gap_sd_xhat'][3]:+.2f} "
             f"samples {rows[-1]['gap_sd_samples'][3]:+.2f} | {(time.time() - t0) / 60:.1f} min")
thr = f_real + 0.10
t_star = 0
for row in rows:
    if row["out_band_xhat"] <= thr:
        t_star = row["t"]
    else:
        break
res = {"checkpoint": path, "epoch": meta.get("epoch"), "n_img": int(a.n_img), "k_samples": [a.k, a.k_hi],
       "real_band_L4": [float(lo), float(hi)], "f_out_real_all": f_real, "threshold": thr, "t_star": t_star,
       "real_mean": mu_r.tolist(), "real_sd": sd.tolist(), "rows": rows}
save_json(res, f"A2_jensen_{a.which}.json")
log(TAG, f"t* = {t_star} (rule: x0_hat out-of-band share <= {thr:.2f} for every grid t up to t*)")
