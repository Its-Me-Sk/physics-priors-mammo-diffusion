# s4_gradsnr.py - S4a part B (no training): how much of the Fokker-Planck gradient is SIGNAL and how much is NOISE?
# On the epoch-40 clean baseline (seed 42, fp32 weights of base_clean_s42_last.pt):
#  1. RESIDUAL PROFILE on the 32-image validation probe set (same selection as s3_eval), t = 1 ... 990, two probes:
#     unnormalised mean r1 r2 (+ z), RMS of the time-derivative term A = u.d_tau s, relative violation
#     rho(t) = sqrt(mean r1 r2) / RMS(A), and an extended normalisation table S_ext(t) = RMS(A) (covers t < 10).
#  2. EXACT-JVP CHECK of the finite-difference step table at small t (1, 2, 5, 10): 4 images x 2 probes.
#  3. BASE GRADIENT: 32 micro-batches (batch 2, training recipe incl. augmentation / label dropout) in 4 groups of 8.
#  4. FP GRADIENT SNR: at t in {2, 10, 50, 150, 300}: 8 training images (4/4) x K = 4 independent draws (noise, u, v1, v2)
#     of the normalised FP-loss gradient g = grad( r1 r2 / S_ext(t)^2 ). With group sums S_a (one group = one image at
#     one t) and their exact Gram matrix (float64, parameter chunks on the GPU):
#       |mu|^2 (population mean, unbiased)  = sum_{a!=b} S_a.S_b / (I (I-1) K^2)      [pairs of DIFFERENT images]
#       image-level mean |mu_a|^2           = sum_a (|S_a|^2 - sum_k |g_ak|^2) / (I K (K-1))
#       E|g|^2                              = sum |g|^2 / N ;  tr(Sigma) = E|g|^2 - |mu|^2
#       SNR (one sample) = |mu| / sqrt(tr Sigma);  K_needed = tr Sigma / |mu|^2  (samples to average for SNR = 1)
#     jackknife SE over images; cos(mu_FP, mu_base) from cross products of independent samples.
#     The same statistics for the base gradient (per micro-batch) give the reference noise level of ordinary training.
import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, minmax_pm1, schedule, find, OUT
import fp_core as fc
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--profile_t", default="1,2,5,10,20,50,100,200,300,500,700,900,990")
ap.add_argument("--jvp_t", default="1,2,5,10")
ap.add_argument("--snr_t", default="2,10,50,150,300")
ap.add_argument("--n_probe", type=int, default=32)
ap.add_argument("--n_img", type=int, default=8)
ap.add_argument("--k_draws", type=int, default=4)
ap.add_argument("--n_base_groups", type=int, default=4)
ap.add_argument("--base_group_size", type=int, default=8)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.profile_t, a.jvp_t, a.snr_t, a.n_probe, a.n_img, a.k_draws, a.n_base_groups, a.base_group_size = "1,20", "2", "10", 2, 2, 2, 2, 2
TAG = "S4-B"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
t_all = time.time()
PT = [int(v) for v in a.profile_t.split(",")]; JT = [int(v) for v in a.jvp_t.split(",")]; ST = [int(v) for v in a.snr_t.split(",")]
SREF2 = json.load(open(find("A8_fp_clean.json")[0]))["S_ref_table"]

unet, ac = schedule(dev)                                   # ac float64
ck = torch.load(a.init, map_location="cpu", weights_only=False)
unet.load_state_dict(ck["unet"]); del ck
unet = unet.to(dev).float().eval()
params = [p for p in unet.parameters()]
D = sum(p.numel() for p in params)
LOGAB = torch.log(ac.cpu())
acf = ac.float()
log(TAG, f"model {a.init} | {D / 1e6:.1f} M parameters")

xtr, ytr, pat, _ = load_clean_meta("train")
vm = val_mask(pat)
XV, YV, PV = xtr[vm], ytr[vm], pat[vm]
rng = np.random.default_rng(0)
pick, used, cnt = {0: [], 1: []}, set(), {0: 0, 1: 0}
for j in rng.permutation(len(YV)):
    c = int(YV[j])
    if PV[j] in used or cnt[c] >= a.n_probe // 2:
        continue
    pick[c].append(int(j)); used.add(PV[j]); cnt[c] += 1
PIDX = np.array(pick[0] + pick[1])
X0 = minmax_pm1(torch.from_numpy(XV[PIDX]).float().unsqueeze(1)).to(dev); LAB = [int(v) for v in YV[PIDX]]
H, W = X0.shape[-2:]
SF = {lab: fc.make_score_fn(unet, ac, lab) for lab in (-1, 0, 1)}
rep = {"init": a.init, "parameters_M": D / 1e6}

# ---------------- 1. residual profile ----------------
gp = torch.Generator().manual_seed(31337)
rows, sref_ext, R_all = [], {}, {}
t0 = time.time()
with torch.no_grad():
    for t in PT:
        p, beta, ab = fc.p_of_t(t), fc.beta_eff(LOGAB, t), float(ac[t])
        R, A_ = [], []
        for i in range(a.n_probe):
            nz = torch.randn((1, 1, H, W), generator=gp); u = torch.randn((1, 1, H, W), generator=gp); u = u / u.norm()
            vs = [(torch.randint(0, 2, (1, 1, H, W), generator=gp) * 2 - 1).float() for _ in range(2)]
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * nz.to(dev)
            u, vs = u.to(dev), [v.to(dev) for v in vs]
            s = fc.evaluate_nograd(SF[LAB[i]], fc.plan_n(x, t, u, vs, p))
            rs, A, _ = fc.residuals_n(s, x, u, vs, p, 1, beta)
            R.append([float(r) for r in rs]); A_.append(float(A))
        R, A_ = np.array(R), np.array(A_)
        prod = R[:, 0] * R[:, 1]
        lhs = float(np.sqrt(np.mean(A_ ** 2)))
        sref_ext[str(t)] = lhs
        m, se = float(prod.mean()), float(prod.std(ddof=1) / np.sqrt(len(prod))) if len(prod) > 1 else float("nan")
        rows.append({"t": t, "p": p, "sigma": math.sqrt(1 - ab), "mean_r1r2": m, "z": m / (se + 1e-300), "lhs_rms": lhs,
                     "rho_rel": math.sqrt(max(m, 0.0)) / (lhs + 1e-300), "P_norm_ext": m / lhs ** 2,
                     "P_norm_S2_table": m / fc.sref_interp(SREF2, t) ** 2 if 10 <= t <= 300 else None})
        R_all[t] = R
        log(TAG, f"profile t={t:4d} sigma={math.sqrt(1 - ab):.4f}: mean r1r2 {m:+.4e} (z {rows[-1]['z']:5.2f}) | RMS(u.d_tau s) {lhs:.3e} | "
                 f"relative violation {rows[-1]['rho_rel']:.3f} | {(time.time() - t0) / 60:.1f} min")
rep["profile"] = rows; rep["S_ext"] = sref_ext

# ---------------- 2. exact-JVP check of the FD step at small t ----------------
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
jv = []
gj = torch.Generator().manual_seed(2718)
for p_ in params:
    p_.requires_grad_(False)
for t in JT:
    try:
        ab = float(ac[t]); e_fd, e_ref = [], []
        for i in list(range(min(2, a.n_probe // 2))) + list(range(a.n_probe // 2, a.n_probe // 2 + min(2, a.n_probe // 2))):
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * torch.randn((1, 1, H, W), generator=gj).to(dev)
            u = torch.randn((1, 1, H, W), generator=gj); u = (u / u.norm()).to(dev)
            for _ in range(2):
                v = (torch.randint(0, 2, (1, 1, H, W), generator=gj) * 2 - 1).float().to(dev)
                s = fc.evaluate_nograd(SF[LAB[i]], fc.plan_n(x, t, u, [v], fc.p_of_t(t)))
                e_fd.append(float(fc.residuals_n(s, x, u, [v], fc.p_of_t(t), 1, fc.beta_eff(LOGAB, t))[0][0]))
                e_ref.append(r_jvp(x, t, SF[LAB[i]], u, v))
        e_fd, e_ref = np.array(e_fd), np.array(e_ref)
        rel = float(np.sqrt(np.mean((e_fd - e_ref) ** 2)) / (np.sqrt(np.mean(e_ref ** 2)) + 1e-300))
        jv.append({"t": t, "p": fc.p_of_t(t), "rel_err_fd_vs_jvp": rel, "pass": bool(rel <= 0.10)})
        log(TAG, f"JVP check t={t}: FD-vs-JVP rel err {rel:.4f} -> {'PASS' if rel <= 0.10 else 'FAIL'}")
    except Exception as e:
        jv.append({"t": t, "error": repr(e)[:300]}); log(TAG, f"JVP check t={t} failed: {repr(e)[:200]}")
for p_ in params:
    p_.requires_grad_(True)
rep["jvp_small_t"] = jv

# ---------------- helpers: float64 Gram of CPU float32 vectors, chunked on the GPU ----------------
def flat_grad():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in params])
def sq64(v):
    return float(sum((v[s:s + 20_000_000].double() ** 2).sum() for s in range(0, v.numel(), 20_000_000)))
def gram(vecs, ch=8_000_000):
    K_ = len(vecs); G = torch.zeros(K_, K_, dtype=torch.float64, device=dev)
    for s in range(0, D, ch):
        C = torch.stack([v[s:s + ch] for v in vecs]).to(dev).double()
        G += C @ C.T
    return G.cpu().numpy()

# ---------------- 3. base gradient (training recipe) ----------------
_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX, KY = (_sob[None, None] / 8).to(dev), (_sob.t()[None, None] / 8).to(dev)
def hf_loss(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean()
Xg, Yg = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).long().to(dev)
gb_gen = torch.Generator(device=dev).manual_seed(5150)
def augment(img, gg):
    B = img.shape[0]
    do = (torch.rand(B, generator=gg, device=dev) < 0.9).float()
    uu = lambda: (torch.rand(B, generator=gg, device=dev) * 2 - 1)
    sx, sy, shx, shy = 1 + 0.05 * uu() * do, 1 + 0.05 * uu() * do, 0.1 * uu() * do, 0.1 * uu() * do
    zero = torch.zeros(B, device=dev)
    theta = torch.stack([torch.stack([sx, shx, zero], 1), torch.stack([shy, sy, zero], 1)], 1)
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    flip = torch.rand(B, generator=gg, device=dev) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), out.flip(-1), out)
SC = 1024.0
base_groups, base_self = [], 0.0
unet.train()
for gi in range(a.n_base_groups):
    acc = torch.zeros(D, device=dev); n_ok, tries = 0, 0
    while n_ok < a.base_group_size and tries < 3 * a.base_group_size:
        tries += 1
        unet.zero_grad(set_to_none=True)
        idx = torch.randint(0, Xg.shape[0], (2,), generator=gb_gen, device=dev)
        x0 = minmax_pm1(augment(Xg[idx].float().unsqueeze(1), gb_gen)); y = Yg[idx]
        t = torch.randint(0, 1000, (2,), generator=gb_gen, device=dev)
        eps = torch.randn(x0.shape, generator=gb_gen, device=dev)
        drop = torch.rand(2, generator=gb_gen, device=dev) < 0.15
        cond = torch.where(drop, torch.full_like(y, -1), y).float().view(2, 1, 1)
        ab = acf[t].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        with torch.autocast("cuda", dtype=torch.float16):
            eh = unet(xt, timesteps=t, context=cond)
        eh = eh.float()
        x0h = ((xt - (1 - ab).sqrt() * eh) / ab.sqrt()).clamp(-1, 1)
        loss = 0.7 * F.mse_loss(eh, eps) + 0.2 * (1 - msssim((x0h + 1) / 2, (x0 + 1) / 2, data_range=2.0, gaussian_kernel=False,
                                                          normalize="relu")) + 0.05 * hf_loss(x0h, x0)
        (loss * SC).backward()
        g = flat_grad() / SC
        if not bool(torch.isfinite(g).all()):            # fp16 overflow in this micro-batch -> draw another one
            continue
        base_self += sq64(g); acc += g; n_ok += 1
    if n_ok < a.base_group_size:
        raise RuntimeError("too many non-finite base gradients")
    base_groups.append(acc.cpu()); del acc
unet.eval(); unet.zero_grad(set_to_none=True)
nb_ = a.n_base_groups * a.base_group_size
Gb = gram(base_groups)
mu_b2 = (Gb.sum() - base_self) / (nb_ * (nb_ - 1))
base = {"micro_batches": nb_, "E_norm2": base_self / nb_, "mu_norm2_unbiased": float(mu_b2),
        "trace_Sigma": base_self / nb_ - float(mu_b2)}
base["snr_single"] = math.sqrt(mu_b2 / base["trace_Sigma"]) if mu_b2 > 0 else 0.0
base["K_needed"] = base["trace_Sigma"] / mu_b2 if mu_b2 > 0 else float("inf")
rep["base_gradient"] = base
log(TAG, f"base gradient (per micro-batch of 2): |g| RMS {math.sqrt(base['E_norm2']):.3g} | |mean| {math.sqrt(max(mu_b2, 0)):.3g} | "
         f"SNR {base['snr_single']:.3f} | micro-batches needed for SNR 1: {base['K_needed']:.1f}")

# ---------------- 4. FP gradient SNR ----------------
gs_ = torch.Generator(device=dev).manual_seed(8086); rs_ = np.random.default_rng(8086)
timg = np.concatenate([rs_.choice(np.where(ytr == c)[0], a.n_img // 2, replace=False) for c in (0, 1)])
snr_rows = []
I, K = a.n_img, a.k_draws
for t in ST:
    t1 = time.time()
    ab = float(ac[t]); w = 1.0 / fc.sref_interp(sref_ext, t) ** 2
    groups, selfs, Ls = [], [], []
    for j in timg:
        x0 = minmax_pm1(torch.from_numpy(xtr[int(j):int(j) + 1]).float().unsqueeze(1).to(dev))
        acc = torch.zeros(D, device=dev); sa = 0.0
        for _ in range(K):
            x = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * torch.randn(x0.shape, generator=gs_, device=dev)
            u = torch.randn(x.shape, generator=gs_, device=dev); u = u / u.norm()
            v1 = (torch.randint(0, 2, x.shape, generator=gs_, device=dev) * 2 - 1).float()
            v2 = (torch.randint(0, 2, x.shape, generator=gs_, device=dev) * 2 - 1).float()
            unet.zero_grad(set_to_none=True)
            L, r1, r2, A = fc.fp_loss_and_grad(SF[int(ytr[int(j)])], x, t, u, v1, v2, fc.p_of_t(t), 1, fc.beta_eff(LOGAB, t), w=w)
            g = flat_grad()
            if not (math.isfinite(L) and bool(torch.isfinite(g).all())):
                raise RuntimeError(f"non-finite FP gradient at t={t} (image {int(j)})")
            sa += sq64(g); acc += g; Ls.append(L)
        groups.append(acc.cpu()); selfs.append(sa); del acc
    log(TAG, f"  t={t}: {I * K} FP gradients done ({(time.time() - t1) / 60:.1f} min)")
    unet.zero_grad(set_to_none=True)
    G = gram(groups + base_groups)
    Gf = G[:I, :I]; selfs = np.array(selfs)
    off = Gf.sum() - np.trace(Gf)
    m_cross = off / (I * (I - 1) * K * K)
    m_within = (np.trace(Gf) - selfs.sum()) / (I * K * (K - 1))
    m_self = selfs.sum() / (I * K)
    jk = []
    for q in range(I):
        offq = off - 2 * (Gf[q].sum() - Gf[q, q])
        jk.append(offq / ((I - 1) * (I - 2) * K * K) if I > 2 else float("nan"))
    jk = np.array(jk); se = float(np.sqrt((I - 1) / I * np.sum((jk - jk.mean()) ** 2))) if I > 2 else float("nan")
    trS = m_self - m_cross
    Xfb = G[:I, I:I + len(base_groups)].sum() / (I * K * nb_)
    cosb = Xfb / math.sqrt(m_cross * mu_b2) if m_cross > 0 and mu_b2 > 0 else None
    del groups
    row = {"t": t, "samples": I * K, "E_norm2": m_self, "mu_norm2_unbiased": float(m_cross), "mu_norm2_jackknife_se": se,
           "image_level_mu_norm2": float(m_within), "trace_Sigma": float(trS),
           "snr_single": math.sqrt(m_cross / trS) if m_cross > 0 and trS > 0 else 0.0,
           "K_needed": float(trS / m_cross) if m_cross > 0 else float("inf"),
           "K_needed_at_upper95_mu": float(trS / (m_cross + 1.96 * se)) if (m_cross + 1.96 * se) > 0 else float("inf"),
           "cos_mean_fp_vs_mean_base": cosb,
           "mean_L_fp_norm": float(np.mean(Ls)), "minutes": (time.time() - t1) / 60}
    for lam in (0.016, 0.054, 0.107):
        row[f"mean_pull_ratio_lam{lam}"] = lam * math.sqrt(max(m_cross, 0)) / math.sqrt(mu_b2) if mu_b2 > 0 else None
    snr_rows.append(row)
    log(TAG, f"FP SNR t={t}: |g| RMS {math.sqrt(m_self):.3g} | |mean| {math.sqrt(max(m_cross, 0)):.3g} (se of |mean|^2 {se:.2e}) | "
             f"image-level |mean| {math.sqrt(max(m_within, 0)):.3g} | SNR {row['snr_single']:.3f} | samples needed for SNR 1: "
             f"{row['K_needed']:.1f} | cos(mean FP, mean base) {cosb if cosb is None else round(cosb, 3)} | {row['minutes']:.1f} min")
rep["fp_snr"] = snr_rows
rep["minutes"] = (time.time() - t_all) / 60
save_json(rep, "S4_B_gradsnr.json")
log(TAG, f"done in {rep['minutes']:.1f} min")
