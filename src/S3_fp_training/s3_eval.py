# s3_eval.py - S3 evaluation of one arm at one or more epochs (VALIDATION images only; the test set is not touched).
#  1. H1 PROBE SET (every epoch): 32 validation images from 32 different patients (16 benign / 16 malignant, fixed),
#     t in {10,20,50,100,200,300}, one unit direction u and n_v = 2 Rademacher probes per (image, t), all drawn once from
#     a fixed CPU seed -> identical x_t, u, v in every arm and epoch (paired). fp32 finite differences with the fixed
#     step table p(t). P(i,t) = r(u,v1) r(u,v2) / S_ref(t)^2 (unbiased for (u.R)^2 / S_ref^2), H(i) = mean_t P(i,t).
#  2. VALIDATION LOSS (every epoch): the training loss (0.7 MSE + 0.2 (1-MS-SSIM) + 0.05 HF) and the MSE on all 347
#     validation images x 4 fixed (t, noise) pairs, true labels, no augmentation, fp16 autocast as in training.
#  3. FINAL EPOCH only (--final):
#     a. exact-JVP check: 8 probe images (4/4) x t in {20, 200}, same x_t, u, v1, v2 -> P_jvp (decides "R_jvp > 0")
#        and the FD-vs-JVP relative error after training (FP training must not exploit the FD error).
#     b. samples: n_gen per class at guidance 2 (primary), DDIM 50, the S2 noise bank (seed 2025 + 100000*label + batch)
#        -> saturation share / brightness / HF energy (istats of A9), KID-mammo vs the train-fit images (A6 seed-0
#        ConvNeXt penultimate features, same kid() as A9), memorisation (64x64 pixel correlation to every train image vs
#        the validation->train-fit reference q99), 8-per-class grid png.
import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, minmax_pm1, load_model, find, OUT
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--epochs", required=True)               # e.g. "41,42"
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
ap.add_argument("--ckpt", default="")                    # explicit snapshot path (single epoch), e.g. the epoch-40 baseline
ap.add_argument("--final", action="store_true")          # run part 3 at the LAST listed epoch
ap.add_argument("--n_probe", type=int, default=32)
ap.add_argument("--n_v", type=int, default=2)
ap.add_argument("--grid", default="10,20,50,100,200,300")
ap.add_argument("--n_rep", type=int, default=4)
ap.add_argument("--n_gen", type=int, default=100)
ap.add_argument("--gs", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.n_probe, a.grid, a.n_rep, a.n_gen, a.steps = 2, "20,200", 1, 2, 5
TAG = f"S3-eval-{a.arm}"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
GRID = [int(v) for v in a.grid.split(",")]
EPOCHS = [int(v) for v in a.epochs.split(",")]
SREF = json.load(open(find("A8_fp_clean.json")[0]))["S_ref_table"]

# ---------------- fixed validation material (identical in every arm / epoch / session) ----------------
xtr, ytr, pat, _ = load_clean_meta("train")
vm = val_mask(pat)
XV, YV, PV = xtr[vm], ytr[vm], pat[vm]
XF = xtr[~vm]
rng = np.random.default_rng(0)
order = rng.permutation(len(YV))
pick, used, cnt = {0: [], 1: []}, set(), {0: 0, 1: 0}
for j in order:
    c = int(YV[j])
    if PV[j] in used or cnt[c] >= a.n_probe // 2:
        continue
    pick[c].append(int(j)); used.add(PV[j]); cnt[c] += 1
PIDX = np.array(pick[0] + pick[1])                       # first half benign, second half malignant
assert len(PIDX) == a.n_probe and len(set(PV[PIDX])) == a.n_probe
X0 = minmax_pm1(torch.from_numpy(XV[PIDX]).float().unsqueeze(1)).to(dev)
LAB = [int(v) for v in YV[PIDX]]
H, W = X0.shape[-2:]
gcpu = torch.Generator().manual_seed(20261020)
NOISE = torch.randn((a.n_probe, len(GRID), 1, H, W), generator=gcpu)
U = torch.randn((a.n_probe, len(GRID), 1, H, W), generator=gcpu)
U = U / U.flatten(2).norm(dim=2).view(a.n_probe, len(GRID), 1, 1, 1)
V = (torch.randint(0, 2, (a.n_probe, len(GRID), a.n_v, 1, H, W), generator=gcpu) * 2 - 1).float()
NV = 4 if a.smoke else len(YV)
rv = np.random.default_rng(123)
TV = rv.integers(0, 1000, size=(NV, a.n_rep))
gnv = torch.Generator().manual_seed(321)
EPSV = [torch.randn((NV, 1, H, W), generator=gnv) for _ in range(a.n_rep)]

from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX, KY = (_sob[None, None] / 8).to(dev), (_sob.t()[None, None] / 8).to(dev)
def hf_per_image(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean((1, 2, 3))

def ms_per_image(p, q):
    """MS-SSIM per image with the training arguments (reduction='none'; per-image loop if this torchmetrics lacks it)"""
    try:
        return msssim(p, q, data_range=2.0, gaussian_kernel=False, normalize="relu", reduction="none").reshape(-1).float()
    except Exception:
        return torch.stack([msssim(p[j:j + 1], q[j:j + 1], data_range=2.0, gaussian_kernel=False, normalize="relu")
                            for j in range(p.shape[0])]).reshape(-1).float()

def probe_set(unet, ac, LOGAB):
    SF = {lab: fc.make_score_fn(unet, ac, lab) for lab in (0, 1)}
    R = np.zeros((a.n_probe, len(GRID), a.n_v)); A_ = np.zeros((a.n_probe, len(GRID)))
    t0 = time.time()
    for ti, t in enumerate(GRID):
        p, beta, ab = fc.p_of_t(t), fc.beta_eff(LOGAB, t), float(ac[t])
        for i in range(a.n_probe):
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i, ti:ti + 1].to(dev)
            u = U[i, ti:ti + 1].to(dev); vs = [V[i, ti, q:q + 1].to(dev) for q in range(a.n_v)]
            s = fc.evaluate_nograd(SF[LAB[i]], fc.plan_n(x, t, u, vs, p))
            rs, A, _ = fc.residuals_n(s, x, u, vs, p, 1, beta)
            R[i, ti] = [float(r) for r in rs]; A_[i, ti] = float(A)
        log(TAG, f"  probe t={t:4d}: done ({(time.time() - t0) / 60:.1f} min)")
    sr = np.array([fc.sref_interp(SREF, t) for t in GRID])
    nv = a.n_v
    P = (R.sum(2) ** 2 - (R ** 2).sum(2)) / (nv * (nv - 1)) / sr[None, :] ** 2      # = r1 r2 / S_ref^2 for n_v = 2
    return R, A_, P

@torch.no_grad()
def val_loss(unet, ac):
    acf = ac.float(); comp = np.zeros((NV, a.n_rep)); mse = np.zeros((NV, a.n_rep))
    for k in range(a.n_rep):
        for i in range(0, NV, 8):
            x0 = minmax_pm1(torch.from_numpy(XV[i:min(i + 8, NV)]).float().unsqueeze(1).to(dev))
            n = x0.shape[0]
            t = torch.as_tensor(TV[i:i + n, k], device=dev, dtype=torch.long)
            eps = EPSV[k][i:i + n].to(dev)
            cond = torch.as_tensor(YV[i:i + n], device=dev).float().view(n, 1, 1)
            ab = acf[t].view(-1, 1, 1, 1)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            with torch.autocast("cuda", dtype=torch.float16):
                eh = unet(xt, timesteps=t, context=cond)
            eh = eh.float()
            m = ((eh - eps) ** 2).mean((1, 2, 3))
            x0h = ((xt - (1 - ab).sqrt() * eh) / ab.sqrt()).clamp(-1, 1)
            ms = 1.0 - ms_per_image((x0h + 1) / 2, (x0 + 1) / 2)
            c = 0.7 * m + 0.2 * ms + 0.05 * hf_per_image(x0h, x0)
            comp[i:i + n, k] = c.cpu().numpy(); mse[i:i + n, k] = m.cpu().numpy()
    return comp, mse

from torch.func import jvp
def r_jvp(x, t, sfn, u, v, LOGAB):
    beta = fc.beta_eff(LOGAB, t)
    with torch.no_grad():
        s0 = sfn(x, t); A = (u * (sfn(x, t + 1) - sfn(x, t - 1))).sum() / (2 * fc.DTAU)
    _, Ju = jvp(lambda z: sfn(z, t), (x,), (u,))
    def g(z):
        _, jv = jvp(lambda w: sfn(w, t), (z,), (v,))
        return (v * jv).sum()
    _, d2 = jvp(g, (x,), (u,))
    return float(A - 0.5 * beta * ((u * s0).sum() + (Ju * (x + 2 * s0)).sum() + d2))

def jvp_check(unet, ac, LOGAB, R_fd):
    SF = {lab: fc.make_score_fn(unet, ac, lab) for lab in (0, 1)}
    h = a.n_probe // 2; imgs = list(range(min(4, h))) + list(range(h, h + min(4, h)))
    out = {"images": imgs, "t": [], "P_jvp": [], "P_fd": [], "r_jvp": [], "r_fd": []}
    for t in (20, 200):
        if t not in GRID:
            continue
        ti = GRID.index(t); ab = float(ac[t]); sr2 = fc.sref_interp(SREF, t) ** 2
        for i in imgs:
            x = math.sqrt(ab) * X0[i:i + 1] + math.sqrt(1 - ab) * NOISE[i, ti:ti + 1].to(dev)
            u = U[i, ti:ti + 1].to(dev)
            rj = [r_jvp(x, t, SF[LAB[i]], u, V[i, ti, q:q + 1].to(dev), LOGAB) for q in range(2)]
            out["t"].append(t); out["r_jvp"].append(rj); out["r_fd"].append([float(R_fd[i, ti, q]) for q in range(2)])
            out["P_jvp"].append(rj[0] * rj[1] / sr2); out["P_fd"].append(float(R_fd[i, ti, 0] * R_fd[i, ti, 1]) / sr2)
    rj, rf = np.array(out["r_jvp"]).ravel(), np.array(out["r_fd"]).ravel()
    out["rel_err_fd_vs_jvp"] = float(np.sqrt(np.mean((rf - rj) ** 2)) / (np.sqrt(np.mean(rj ** 2)) + 1e-300))
    return out

# ---------------- part 3b helpers (copied from a9_metrics.py: same definitions) ----------------
def istats(X_u8):
    I = torch.from_numpy(X_u8).float().div(255).unsqueeze(1)
    mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True); I = (I - mn) / (mx - mn + 1e-8)
    mu = F.avg_pool2d(I, 7, 1, 3, count_include_pad=False); m = (mu > 0.04).float(); n = m.sum((1, 2, 3)).clamp_min(1)
    hp = I - F.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    fr = torch.ones_like(I); fr[..., 4:-4, 4:-4] = 0
    return {"mean_breast": float(((I * m).sum((1, 2, 3)) / n).mean()),
            "saturation_share": float((((I > 0.95).float() * m).sum((1, 2, 3)) / n).mean()),
            "hf_energy": float(((hp * hp * m).sum((1, 2, 3)) / n).mean()),
            "breast_area": float(m.mean((1, 2, 3)).mean()), "border_brightness": float(((I * fr).sum((1, 2, 3)) / fr.sum((1, 2, 3))).mean())}

def kid(fa, fb, n_sub=100, size=100, seed=0):
    r = np.random.default_rng(seed); d = fa.shape[1]; vals = []; size = min(size, len(fa), len(fb))
    for _ in range(n_sub):
        A_ = fa[r.choice(len(fa), size, replace=False)]; B_ = fb[r.choice(len(fb), size, replace=False)]
        kxx = (A_ @ A_.T / d + 1) ** 3; kyy = (B_ @ B_.T / d + 1) ** 3; kxy = (A_ @ B_.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (size * (size - 1)) + (kyy.sum() - np.trace(kyy)) / (size * (size - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))

def thumbs(X_u8):
    t = X_u8.reshape(len(X_u8), 64, 4, 64, 4).mean((2, 4)).reshape(len(X_u8), -1).astype(np.float64)
    t -= t.mean(1, keepdims=True); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return t

@torch.no_grad()
def ddim(unet, acf, label, n, gen, gs):
    T = acf.numel()
    x = torch.randn(n, 1, 256, 256, generator=gen, device=dev)
    ts = torch.linspace(T - 1, 0, a.steps, device=dev).round().long()
    cond = torch.full((n, 1, 1), float(label), device=dev)
    for i, t in enumerate(ts):
        ab_t = acf[t]
        ab_prev = acf[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        tb = torch.full((2 * n,), int(t), device=dev, dtype=torch.long)
        ctx = torch.cat([-torch.ones_like(cond), cond]).half()
        eu, ec = unet(torch.cat([x, x]).half(), timesteps=tb, context=ctx).float().chunk(2)
        eps = eu + gs * (ec - eu)
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return ((x + 1) / 2).clamp(0, 1)

def final_quality(unet, ac, name):
    from PIL import Image
    from a6_classifier import build, prep
    acf = ac.float(); unet = unet.half(); t0 = time.time(); imgs, labels = [], []
    for label in (0, 1):
        done, b = 0, 0
        while done < a.n_gen:
            k = min(16, a.n_gen - done)
            out = ddim(unet, acf, label, k, torch.Generator(device=dev).manual_seed(2025 + 100000 * label + b), a.gs)[:, 0].cpu()
            imgs.append((out.numpy() * 255).round().astype(np.uint8)); labels += [label] * k; done += k; b += 1
    G = np.concatenate(imgs); L = np.array(labels, dtype=np.int64)
    np.savez_compressed(os.path.join(OUT, f"gen_{name}_gs{a.gs:g}.npz"), x=G, y=L)
    kk = min(8, a.n_gen)
    Image.fromarray(np.concatenate([np.concatenate([G[j + c * a.n_gen] for j in range(kk)], 1) for c in (0, 1)], 0)).save(
        os.path.join(OUT, f"grid_{name}_gs{a.gs:g}.png"))
    q = {"n_per_class": a.n_gen, "gs": a.gs, "sampling_minutes": (time.time() - t0) / 60,
         "intensity": {nm: istats(G[L == c]) for c, nm in ((0, "benign"), (1, "malignant"))},
         "intensity_real_val": {nm: istats(XV[YV == c]) for c, nm in ((0, "benign"), (1, "malignant"))}}
    q["saturation_max_class"] = max(q["intensity"][c]["saturation_share"] for c in ("benign", "malignant"))
    del unet; torch.cuda.empty_cache()
    mam = build(); mam.load_state_dict({k: v.float() for k, v in torch.load(find("a6_clf_s0.pt")[0], map_location="cpu").items()})
    mam = mam.to(dev).eval()
    @torch.no_grad()
    def feats(X_u8, bs=64):
        out = []
        for i in range(0, len(X_u8), bs):
            xb = torch.from_numpy(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                h = mam.classifier[1](mam.classifier[0](mam.avgpool(mam.features(prep(xb)))))
            out.append(h.float().cpu())
        return torch.cat(out).numpy().astype(np.float64)
    ffit = feats(XF)
    q["kid_mammo_vs_trainfit"] = kid(feats(G), ffit)
    q["kid_mammo_real_val_vs_trainfit_floor"] = kid(feats(XV), ffit)
    ref = (thumbs(XV) @ thumbs(XF).T).max(1); q99 = float(np.quantile(ref, 0.99))
    pc = (thumbs(G) @ thumbs(xtr).T).max(1)
    q["memorisation"] = {"reference_pixel_corr_q99": q99, "share_above_q99": float((pc > q99).mean()), "max_pixel_corr": float(pc.max())}
    log(TAG, f"  quality: saturation {q['intensity']['benign']['saturation_share']:.3f}/{q['intensity']['malignant']['saturation_share']:.3f} "
             f"(real val {q['intensity_real_val']['benign']['saturation_share']:.3f}/{q['intensity_real_val']['malignant']['saturation_share']:.3f}) | "
             f"KID-mammo vs train-fit {q['kid_mammo_vs_trainfit'][0]:.4f} (floor {q['kid_mammo_real_val_vs_trainfit_floor'][0]:.4f}) | "
             f"memorisation share {q['memorisation']['share_above_q99']:.3f} | {(time.time() - t0) / 60:.1f} min")
    return q

# ---------------- run ----------------
log(TAG, f"probe set: {a.n_probe} validation images / {a.n_probe} patients, t {GRID}, {a.n_v} probes | validation loss: "
         f"{NV} images x {a.n_rep} | epochs {EPOCHS} | final {a.final}")
for ei, ep in enumerate(EPOCHS):
    path = a.ckpt or os.path.join(a.ckpt_dir, f"{a.arm}_w_ep{ep:03d}.pt")
    if not os.path.exists(path):
        log(TAG, f"!! {path} missing (training not finished?) -> epoch {ep} skipped"); continue
    t_ep = time.time()
    unet, ac, meta = load_model(path, dev)
    LOGAB = torch.log(ac.cpu())
    name = f"{a.arm}_ep{ep:03d}"
    log(TAG, f"epoch {ep}: {path} (snapshot epoch {meta.get('epoch')}, lam {meta.get('lam')})")
    R, A_, P = probe_set(unet, ac, LOGAB)
    Hh = P.mean(1)
    comp, mse = val_loss(unet, ac)
    rep = {"arm": a.arm, "epoch": ep, "checkpoint": path, "lam": meta.get("lam"), "grid": GRID, "n_v": a.n_v,
           "probe_val_index": PIDX.tolist(), "probe_patients": PV[PIDX].tolist(), "probe_labels": LAB,
           "H_mean": float(Hh.mean()), "P_mean_per_t": dict(zip(map(str, GRID), P.mean(0).tolist())),
           "val_composite_mean": float(comp.mean()), "val_mse_mean": float(mse.mean())}
    extra = {}
    if a.final and ei == len(EPOCHS) - 1:
        try:
            rep["jvp_check"] = jvp_check(unet, ac, LOGAB, R)
            log(TAG, f"  JVP check: FD-vs-JVP rel err {rep['jvp_check']['rel_err_fd_vs_jvp']:.4f} | mean P_jvp "
                     f"{np.mean(rep['jvp_check']['P_jvp']):.4f} vs P_fd {np.mean(rep['jvp_check']['P_fd']):.4f}")
        except Exception as e:
            rep["jvp_check"] = {"error": repr(e)[:300]}; log(TAG, f"  !! JVP check failed: {repr(e)[:200]}")
        rep["quality"] = final_quality(unet, ac, name)
    np.savez(os.path.join(OUT, f"eval_{name}.npz"), R=R, A=A_, P=P, H=Hh, val_comp=comp, val_mse=mse, val_t=TV)
    rep["minutes"] = (time.time() - t_ep) / 60
    save_json(rep, f"eval_{name}.json")
    log(TAG, f"epoch {ep}: H mean {Hh.mean():.4f} | P per t " + " ".join(f"{t}:{v:.3f}" for t, v in zip(GRID, P.mean(0)))
             + f" | val loss {comp.mean():.5f} mse {mse.mean():.5f} | {rep['minutes']:.1f} min")
    del unet; torch.cuda.empty_cache()
