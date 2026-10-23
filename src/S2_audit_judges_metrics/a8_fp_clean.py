# a8_fp_clean.py - A8: Fokker-Planck residual on the CLEAN baseline + the numbers Phase B needs (no training).
#  1. residual size and detectability on 6 validation images, t in {10,20,50,100,200,300}, 3 directions x 4 probes,
#     fp32 finite differences with the pre-fixed step p(t) (fp_core.p_of_t); exact-JVP spot check at t = 20 and 200
#  2. S_ref(t) = RMS of u.d_tau s on a dense t grid -> the fixed normalisation of the training loss
#     L_FP = r(u,v1) r(u,v2) / S_ref(t)^2   (so every t in [10,300] gets comparable weight; ~ rho^2)
#  3. gradient diagnostics (12 training images, t stratified over [10,300]):
#     |g_FP| vs |g_base| (base = original recipe loss, mean over 4 micro-batches = one optimizer step),
#     cos(g_FP, g_base), mean pairwise cos among FP gradients (is the FP direction consistent or noise?),
#     and the lambda that makes lambda*|g_FP| equal to 10 % / 30 % / 100 % of |g_base| (candidates for the LOCK)
# PRE-STATED RULES:
#   spot check: |r_FD - r_JVP| RMS / RMS(r_JVP) <= 0.10 at both t  -> the p(t) table also holds on the clean model
#   detectability: z = mean(product)/SE > 3 at each t (reported per t)
import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache, load_clean_meta, val_mask, to_model_domain, minmax_pm1, load_model, clean_ckpt, OUT
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--seed_model", type=int, default=42)
ap.add_argument("--epoch", type=int, default=40)
ap.add_argument("--n_img", type=int, default=6)
ap.add_argument("--n_u", type=int, default=3)
ap.add_argument("--n_v", type=int, default=4)
ap.add_argument("--grid", default="10,20,50,100,200,300")
ap.add_argument("--sref_grid", default="10,15,20,30,50,75,100,150,200,250,300")
ap.add_argument("--n_grad", type=int, default=12)
a = ap.parse_args()
TAG = "A8"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
path = clean_ckpt(a.seed_model, a.epoch)
unet, ac, meta = load_model(path, dev)                     # fp32, eval mode, requires_grad False
LOGAB = torch.log(ac)
log(TAG, f"checkpoint {path} | epoch {meta.get('epoch')}")
rep = {"checkpoint": path, "epoch": meta.get("epoch"), "p_of_t": {t: fc.p_of_t(t) for t in (10, 20, 50, 100, 300)}}

xtr, ytr, _pat, _ = load_clean_meta("train")
_vm = val_mask(_pat)
xte, yte = xtr[_vm], ytr[_vm]            # residual measured on VALIDATION images (the test set stays untouched in S2)
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(yte == c)[0], a.n_img // 2, replace=False) for c in (0, 1)])
X0 = to_model_domain(xte[idx], dev); LAB = [int(v) for v in yte[idx]]
gen = torch.Generator(device=dev).manual_seed(11)
H, W = X0.shape[-2:]
U = torch.randn((a.n_u, 1, H, W), generator=gen, device=dev); U = U / U.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
V = (torch.randint(0, 2, (a.n_v, 1, H, W), generator=gen, device=dev) * 2 - 1).float()
NOISE = torch.randn(X0.shape, generator=gen, device=dev)
SF = {lab: fc.make_score_fn(unet, ac, lab) for lab in (0, 1)}

# ---------------- 1. residual size / detectability ----------------
rows = []
t_all = time.time()
for t in [int(v) for v in a.grid.split(",")]:
    p, beta, ab = fc.p_of_t(t), fc.beta_eff(LOGAB, t), float(ac[t])
    R, L_, terms = [], [], {"us": [], "Jt": [], "D2": []}
    for i in range(X0.shape[0]):
        x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i:i + 1]
        for j in range(a.n_u):
            u = U[j:j + 1]
            s = fc.evaluate_nograd(SF[LAB[i]], fc.plan_n(x, t, u, [V[q:q + 1] for q in range(a.n_v)], p))
            rs, A, tm = fc.residuals_n(s, x, u, [V[q:q + 1] for q in range(a.n_v)], p, 1, beta)
            R.append([float(r) for r in rs]); L_.append(float(A))
            terms["us"].append(float(tm["us"])); terms["Jt"].append(float(tm["Jt"])); terms["D2"] += [float(d) for d in tm["D2"]]
    R = np.array(R)                                                   # (n_img*n_u, n_v)
    V_ = R.shape[1]
    P = (R.sum(1) ** 2 - (R ** 2).sum(1)) / (V_ * (V_ - 1))
    m, se = float(P.mean()), float(P.std(ddof=1) / np.sqrt(len(P)))
    lhs = float(np.sqrt(np.mean(np.square(L_))))
    rows.append({"t": t, "p": p, "z": m / (se + 1e-300), "rho": float(np.sqrt(max(m, 0))) / (lhs + 1e-300),
                 "lhs_rms": lhs, "mean_sq_residual": m, "single_probe_var_over_mean_sq": float(R.var(1, ddof=1).mean() / (m + 1e-300)),
                 "term_rms": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in terms.items()}})
    log(TAG, f"t={t:4d} p={p:g} | z={rows[-1]['z']:5.2f} rho={rows[-1]['rho']:.3f} | LHS RMS {lhs:.3e} | "
             f"{(time.time() - t_all) / 60:.1f} min")
rep["residual_rows"] = rows

# ---------------- 1b. exact-JVP spot check of the p(t) table on this model ----------------
from torch.func import jvp
def r_jvp(x, t, sfn, u, v):
    beta = fc.beta_eff(LOGAB, t)
    with torch.no_grad():
        s0 = sfn(x, t); A = (u * (sfn(x, t + 1) - sfn(x, t - 1))).sum() / (2 * fc.DTAU)
    _, Ju = jvp(lambda z: sfn(z, t), (x,), (u,))
    def g(z):
        _, jv = jvp(lambda w: sfn(w, t), (z,), (v,))
        return (v * jv).sum()
    _, d2 = jvp(g, (x,), (u,))
    return float(A - 0.5 * beta * ((u * s0).sum() + (Ju * (x + 2 * s0)).sum() + d2))
spot = []
try:
    for t in (20, 200):
        ab = float(ac[t]); e_fd, e_ref = [], []
        for i in (0, X0.shape[0] - 1):
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i:i + 1]
            for j in range(2):
                u, v = U[j:j + 1], V[j:j + 1]
                s = fc.evaluate_nograd(SF[LAB[i]], fc.plan_n(x, t, u, [v], fc.p_of_t(t)))
                e_fd.append(float(fc.residuals_n(s, x, u, [v], fc.p_of_t(t), 1, fc.beta_eff(LOGAB, t))[0][0]))
                e_ref.append(r_jvp(x, t, SF[LAB[i]], u, v))
        e_fd, e_ref = np.array(e_fd), np.array(e_ref)
        rel = float(np.sqrt(np.mean((e_fd - e_ref) ** 2)) / (np.sqrt(np.mean(e_ref ** 2)) + 1e-300))
        spot.append({"t": t, "rel_err_fd_vs_jvp": rel, "pass": bool(rel <= 0.10)})
        log(TAG, f"JVP spot check t={t}: rel err {rel:.4f} -> {'PASS' if rel <= 0.10 else 'FAIL'}")
except Exception as e:
    spot.append({"error": repr(e)[:300]})
rep["jvp_spot_check"] = spot

# ---------------- 2. S_ref(t) table (fixed loss normalisation) ----------------
sref = {}
with torch.no_grad():
    for t in [int(v) for v in a.sref_grid.split(",")]:
        ab = float(ac[t]); vals = []
        for i in range(X0.shape[0]):
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i:i + 1]
            d = (SF[LAB[i]](x, t + 1) - SF[LAB[i]](x, t - 1)) / (2 * fc.DTAU)
            vals += [float((U[j:j + 1] * d).sum()) for j in range(a.n_u)]
        sref[str(t)] = float(np.sqrt(np.mean(np.square(vals))))
rep["S_ref_table"] = sref
log(TAG, "S_ref(t): " + ", ".join(f"{k}: {v:.3g}" for k, v in sref.items()))

# ---------------- 3. gradient diagnostics ----------------
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX, KY = (_sob[None, None] / 8).to(dev), (_sob.t()[None, None] / 8).to(dev)
def hf_loss(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean()
for q in unet.parameters():
    q.requires_grad_(True)
unet.train()
params = [q for q in unet.parameters()]
acf = ac.float()
def flat_grad():
    g = torch.cat([q.grad.detach().flatten() if q.grad is not None else torch.zeros(q.numel(), device=dev)
                   for q in params]).bfloat16().cpu()    # bf16: fp32 range (fp16 would underflow)
    unet.zero_grad(set_to_none=True)
    return g
Xg, Yg = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).long().to(dev)
gg = torch.Generator(device=dev).manual_seed(5)
SCALE = 1024.0
def base_grad():
    """gradient of the ORIGINAL recipe loss on one micro-batch of 2 (as in train_arm2.py, without augmentation)"""
    idx = torch.randint(0, Xg.shape[0], (2,), generator=gg, device=dev)
    x0 = minmax_pm1(Xg[idx].float().unsqueeze(1)); y = Yg[idx]
    t = torch.randint(0, 1000, (2,), generator=gg, device=dev)
    eps = torch.randn(x0.shape, generator=gg, device=dev)
    drop = torch.rand(2, generator=gg, device=dev) < 0.15
    cond = torch.where(drop, torch.full_like(y, -1), y).float().view(2, 1, 1)
    ab = acf[t].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    with torch.autocast("cuda", dtype=torch.float16):
        eh = unet(xt, timesteps=t, context=cond)
    eh = eh.float()
    x0h = ((xt - (1 - ab).sqrt() * eh) / ab.sqrt()).clamp(-1, 1)
    loss = 0.7 * F.mse_loss(eh, eps) + 0.2 * (1 - msssim((x0h + 1) / 2, (x0 + 1) / 2, data_range=2.0, gaussian_kernel=False,
                                                      normalize="relu")) + 0.05 * hf_loss(x0h, x0)
    (loss * SCALE).backward()
    for q in params:
        if q.grad is not None:
            q.grad.div_(SCALE)
    return flat_grad(), float(loss)

bins = [(10, 20), (21, 50), (51, 100), (101, 150), (151, 225), (226, 300)]
rg = np.random.default_rng(1)
G_fp, G_base, info = [], [], []
t0 = time.time()
for n in range(a.n_grad):
    lo, hi = bins[n % len(bins)]
    t = int(rg.integers(lo, hi + 1)); i = int(rg.integers(0, len(ytr)))
    ab = float(ac[t])
    x0 = minmax_pm1(torch.from_numpy(xtr[i:i + 1]).float().unsqueeze(1).to(dev))
    x = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * torch.randn(x0.shape, generator=gg, device=dev)
    u = torch.randn(x.shape, generator=gg, device=dev); u = u / u.norm()
    v1 = (torch.randint(0, 2, x.shape, generator=gg, device=dev) * 2 - 1).float()
    v2 = (torch.randint(0, 2, x.shape, generator=gg, device=dev) * 2 - 1).float()
    w = 1.0 / fc.sref_interp(sref, t) ** 2
    Lfp, r1, r2, A = fc.fp_loss_and_grad(SF[int(ytr[i])], x, t, u, v1, v2, fc.p_of_t(t), 1, fc.beta_eff(LOGAB, t), w=w)
    G_fp.append(flat_grad())
    gb, lb = base_grad()
    G_base.append(gb)
    info.append({"t": t, "label": int(ytr[i]), "L_fp_normalised": Lfp, "r1": r1, "r2": r2, "lhs": A, "base_loss": lb})
    log(TAG, f"grad sample {n + 1}/{a.n_grad}: t={t} L_fp={Lfp:+.4f} | {(time.time() - t0) / 60:.1f} min")

# Gram matrix of all gradient vectors, computed in parameter chunks on the GPU
Gall = G_fp + G_base; K = len(Gall); D = Gall[0].numel()
gram = torch.zeros(K, K, dtype=torch.float64, device=dev)
step = 20_000_000
for s0 in range(0, D, step):
    C = torch.stack([g[s0:s0 + step] for g in Gall]).to(dev).float()
    gram += (C @ C.T).double()
gram = gram.cpu().numpy()
nf = a.n_grad
norm = np.sqrt(np.diag(gram))
cos = gram / np.outer(norm, norm)
fp_n, base_n = norm[:nf], norm[nf:]
# mean of 4 base micro-batch gradients = one optimizer step: |mean| from the Gram matrix
step_norms = []
for gidx in range(nf // 4):
    sl = list(range(nf + 4 * gidx, nf + 4 * gidx + 4))
    step_norms.append(float(np.sqrt(gram[np.ix_(sl, sl)].sum()) / 4))
off = lambda M: M[~np.eye(len(M), dtype=bool)]
gd = {"samples": info,
      "fp_grad_norm": fp_n.tolist(), "base_microbatch_grad_norm": base_n.tolist(),
      "base_optimizer_step_grad_norm(mean of 4)": step_norms,
      "cos_fp_vs_own_base": [float(cos[k, nf + k]) for k in range(nf)],
      "mean_pairwise_cos_fp": float(off(cos[:nf, :nf]).mean()),
      "mean_pairwise_cos_base": float(off(cos[nf:, nf:]).mean()),
      "cos_meanFP_vs_meanBase": float(gram[:nf, nf:].sum() / np.sqrt(gram[:nf, :nf].sum() * gram[nf:, nf:].sum()))}
ratio = float(np.median(step_norms) / np.median(fp_n))
gd["ratio_median_base_step_over_median_fp"] = ratio
gd["lambda_candidates"] = {"10%": 0.1 * ratio, "30%": 0.3 * ratio, "100%": ratio}
gd["clip_note"] = f"training clips the total gradient norm at 1.0; median optimizer-step base norm = {np.median(step_norms):.3g}"
rep["gradient_diagnostics"] = gd
log(TAG, f"median |g_fp| {np.median(fp_n):.3g} | median |g_base step| {np.median(step_norms):.3g} | ratio {ratio:.3g} | "
         f"mean cos(fp, base) {np.mean(gd['cos_fp_vs_own_base']):+.3f} | mean pairwise cos fp {gd['mean_pairwise_cos_fp']:+.3f} "
         f"(base {gd['mean_pairwise_cos_base']:+.3f})")
rep["minutes"] = (time.time() - t_all) / 60
save_json(rep, "A8_fp_clean.json")
