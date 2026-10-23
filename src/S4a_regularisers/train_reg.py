# train_reg.py - S4 screening: continue training a clean baseline with ONE cheap regulariser derived from terms of the
# score Fokker-Planck operator, or with none (control). The BASE recipe and the data / noise / dropout order are EXACTLY
# those of train_arm2.py / train_fp.py (0.7 MSE + 0.2 (1-MS-SSIM) + 0.05 HF on x0_hat, min-max to [-1,1], shear/scale/
# flip augmentation, 15 % label dropout, batch 2 x accum 4, Adam 1e-5, fp16 + GradScaler, global clip 1.0, no EMA).
# Regularisers (sigma-weighted = written for the noise prediction eps_hat = -sigma_t * score, i.e. the same weighting as
# the eps-MSE term of the base loss):
#   none : control
#   sn   : score-norm penalty  P = mean_pixels(eps_hat^2) for the training samples with reg_tmin <= t <= reg_tmax,
#          computed on every micro-batch from the SAME forward pass (sigma^2 |s|^2 / D); its gradient is taken separately
#          (torch.autograd.grad) so base and penalty gradients can be logged.
#   div  : divergence penalty  P = d1 d2 / D_ref(t)^2, d_j = v_j . (eps(x+h v_j) - eps(x-h v_j)) / 2h (Hutchinson with two
#          independent Rademacher probes -> unbiased for (div eps_hat)^2 / D_ref^2), fp32 central differences with the
#          FP step table h = p(t), once per optimizer step on image 0 of the last micro-batch, t ~ U{reg_tmin..reg_tmax},
#          exact gradient by manual recompute (4 evaluations). D_ref(t) = RMS of d at the starting checkpoint (table).
# WEIGHT (pre-stated): lambda is calibrated ONCE at the starting checkpoint so that median |lambda g_reg| = rho *
#   median |g_base| over 24 optimizer-step-sized groups (own random stream, no update). Stored and reused on resume.
# CAP (pre-stated, from the S3 diagnosis): at every step |lambda g_reg| is capped at 2 rho |g_base| before the single
#   global clip at 1.0 -> the regulariser can never dominate the update.
# Logged every optimizer step (jsonl): base / regulariser / combined norm, cap and clip factors, cos(base, reg), exact Adam
# update norm and the first-order predicted change of the base and (normalised) regulariser losses.
import os, sys, glob, time, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim

WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
sys.path.insert(0, WD)
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--reg", choices=["none", "sn", "div"], default="none")
ap.add_argument("--rho", type=float, default=0.25)
ap.add_argument("--reg_tmin", type=int, default=0)
ap.add_argument("--reg_tmax", type=int, default=999)
ap.add_argument("--calib_groups", type=int, default=24)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fp_seed", type=int, default=7919)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--p_uncond", type=float, default=0.15)
ap.add_argument("--budget_h", type=float, default=10.5)
ap.add_argument("--log_every", type=int, default=200)
ap.add_argument("--smoke_batches", type=int, default=0)
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
ap.add_argument("--snap_epochs", default="")          # fp16 weight snapshots: these epochs + always the final one (disk)
a = ap.parse_args()
TAG = f"[{a.arm}]"
def log(*m):
    print(TAG, *m, flush=True)
try:
    T0 = float(open("/tmp/pg_session_start").read())
except Exception:
    T0 = time.time()
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
os.makedirs(a.ckpt_dir, exist_ok=True)

def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)

def minmax_pm1(x):
    mn = x.amin(dim=(1, 2, 3), keepdim=True); mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return 2 * (x - mn) / (mx - mn + 1e-8) - 1

_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX = (_sob[None, None] / 8).to(dev); KY = (_sob.t()[None, None] / 8).to(dev)
def hf_loss(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean()

d = np.load("/tmp/train_cache.npz")
X = torch.from_numpy(d["x"]); Y = torch.from_numpy(d["y"]).long(); N = X.shape[0]
Xg, Yg = X.to(dev), Y.to(dev)
log(f"data {N} images {tuple(X.shape[1:])} | reg={a.reg} rho={a.rho} t in [{a.reg_tmin},{a.reg_tmax}] | {torch.cuda.get_device_name(0)}")

def augment(img, gg):
    B = img.shape[0]
    do = (torch.rand(B, generator=gg, device=dev) < 0.9).float()
    u = lambda: (torch.rand(B, generator=gg, device=dev) * 2 - 1)
    sx, sy, shx, shy = 1 + 0.05 * u() * do, 1 + 0.05 * u() * do, 0.1 * u() * do, 0.1 * u() * do
    zero = torch.zeros(B, device=dev)
    theta = torch.stack([torch.stack([sx, shx, zero], 1), torch.stack([shy, sy, zero], 1)], 1)
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    flip = torch.rand(B, generator=gg, device=dev) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), out.flip(-1), out)

from model_architectures import MonaiDDPM
_m = MonaiDDPM(lr=a.lr, T=1000)
unet = _m.unet.to(dev)
ac = _m.scheduler.alphas_cumprod.detach().float().to(dev)
del _m
T = ac.numel()
opt = torch.optim.Adam(unet.parameters(), lr=a.lr)
scaler = torch.amp.GradScaler("cuda")
params = [p for p in unet.parameters()]

# ---------------- resume this arm if a checkpoint exists, else warm start ----------------
own = [os.path.join(a.ckpt_dir, f"{a.arm}_last.pt")]
if not a.smoke_batches:
    own += glob.glob(f"/kaggle/input/**/{a.arm}_last.pt", recursive=True)
cands = []
for p in own:
    if os.path.exists(p):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            cands.append((int(ck["step"]), p)); del ck
        except Exception as e:
            log("unreadable checkpoint (skipped):", p, repr(e)[:150])
src = max(cands)[1] if cands else a.init
ck = torch.load(src, map_location="cpu", weights_only=False)
if cands:
    for k in ("lr", "bs", "accum", "seed", "p_uncond"):
        if ck["args"].get(k) != getattr(a, k):
            log(f"!! WARNING {k} differs: checkpoint={ck['args'].get(k)} now={getattr(a, k)}")
unet.load_state_dict(ck["unet"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
start_epoch, step, opt_steps = int(ck["epoch"]), int(ck["step"]), int(ck["opt_steps"])
history = ck.get("history", []) if cands else []
init_epoch = int(ck.get("init_epoch", ck["epoch"])) if cands else int(ck["epoch"])
lam = ck.get("lam_cal") if cands else None
dref = ck.get("dref") if cands else None
del ck; torch.cuda.empty_cache()
n_opt = sum(1 for s in opt.state.values() if s)
chk = float(sum(p.detach().float().sum() for p in params[:5]))
log(f"START epoch={start_epoch} step={step} opt_steps={opt_steps} opt_states={n_opt} checksum={chk:.6f} init_epoch={init_epoch} from={src}")
if n_opt == 0:
    log("!! WARNING: optimizer state is EMPTY -> this is not a true continuation of the baseline")

def sqnorm(gs):
    return sum(float((g.float() ** 2).sum()) for g in gs if g is not None)

def dotl(xs, ys):
    return sum(float((x * y).sum()) for x, y in zip(xs, ys) if x is not None and y is not None)

@torch.no_grad()
def adam_update_stats(g_base, g_reg):
    """exact torch.optim.Adam update for the current (clipped) .grad: |update|, g_base.update, g_reg.update"""
    gr = opt.param_groups[0]; b1, b2 = gr["betas"]; lr, eps = gr["lr"], gr["eps"]
    nrm2, d_base, d_reg = 0.0, 0.0, 0.0
    for j, p in enumerate(params):
        g = p.grad
        if g is None:
            continue
        st = opt.state.get(p, {})
        k = float(st["step"]) + 1 if "step" in st else 1.0
        m = b1 * st["exp_avg"] + (1 - b1) * g if "exp_avg" in st else (1 - b1) * g
        v = b2 * st["exp_avg_sq"] + (1 - b2) * g * g if "exp_avg_sq" in st else (1 - b2) * g * g
        upd = -(lr / (1 - b1 ** k)) * m / (v.sqrt() / math.sqrt(1 - b2 ** k) + eps)
        nrm2 += float((upd * upd).sum())
        if g_base[j] is not None:
            d_base += float((g_base[j] * upd).sum())
        if g_reg is not None and g_reg[j] is not None:
            d_reg += float((g_reg[j] * upd).sum())
    return math.sqrt(nrm2), d_base, d_reg

def eps_fn(xb, tt, lab):
    c = torch.full((xb.shape[0], 1, 1), float(lab), device=dev)
    tb = torch.full((xb.shape[0],), int(tt), device=dev, dtype=torch.long)
    return unet(xb, timesteps=tb, context=c)

def div_terms(x, tt, v1, v2, lab, w, backward=True):
    """d_j = v_j.(eps(x+h v_j) - eps(x-h v_j))/2h; L = w d1 d2; exact gradient accumulated into .grad (manual recompute)"""
    h = fc.p_of_t(tt)
    Xs = torch.cat([x + h * v1, x - h * v1, x + h * v2, x - h * v2])
    with torch.no_grad():
        E = torch.cat([eps_fn(Xs[i:i + 2], tt, lab) for i in range(0, 4, 2)])
        d1 = (v1 * (E[0:1] - E[1:2])).sum() / (2 * h)
        d2 = (v2 * (E[2:3] - E[3:4])).sum() / (2 * h)
        coef = torch.cat([w * d2 * v1, -w * d2 * v1, w * d1 * v2, -w * d1 * v2]) / (2 * h)
    if backward:
        for i in range(0, 4, 2):
            (coef[i:i + 2] * eps_fn(Xs[i:i + 2], tt, lab)).sum().backward()
    return float(w * d1 * d2), float(d1), float(d2)

def micro_batch(perm, i, gg):
    idx = perm[i * a.bs:(i + 1) * a.bs].to(dev)
    x0 = minmax_pm1(augment(Xg[idx].float().unsqueeze(1), gg)); y = Yg[idx]
    t = torch.randint(0, T, (a.bs,), generator=gg, device=dev)
    eps = torch.randn(x0.shape, generator=gg, device=dev)
    drop = torch.rand(a.bs, generator=gg, device=dev) < a.p_uncond
    cond = torch.where(drop, torch.full_like(y, -1), y).float().view(a.bs, 1, 1)
    ab = ac[t].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    with torch.autocast("cuda", dtype=torch.float16):
        eps_hat = unet(xt, timesteps=t, context=cond)
    eps_hat = eps_hat.float()
    mse = F.mse_loss(eps_hat, eps)
    x0_hat = ((xt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()).clamp(-1, 1)
    ms = 1.0 - msssim((x0_hat + 1) / 2, (x0 + 1) / 2, data_range=2.0, gaussian_kernel=False, normalize="relu")
    hf = hf_loss(x0_hat, x0)
    loss = 0.7 * mse + 0.2 * ms + 0.05 * hf
    return x0, t, cond, eps_hat, loss, mse, ms, hf

def sn_value(t, eps_hat):
    msk = ((t >= a.reg_tmin) & (t <= a.reg_tmax)).float()
    return (msk * eps_hat.pow(2).mean((1, 2, 3))).sum() / a.bs, int(msk.sum())

def draw_div(x0img, gen, rng):
    tt = int(rng.integers(max(a.reg_tmin, 1), a.reg_tmax + 1))
    abf = float(ac[tt])
    x = math.sqrt(abf) * x0img + math.sqrt(1 - abf) * torch.randn(x0img.shape, generator=gen, device=dev)
    v1 = (torch.randint(0, 2, x.shape, generator=gen, device=dev) * 2 - 1).float()
    v2 = (torch.randint(0, 2, x.shape, generator=gen, device=dev) * 2 - 1).float()
    return tt, x, v1, v2

# ---------------- D_ref table (div) and lambda calibration at the starting checkpoint (fresh arms only) ----------------
DREF_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 900, 999]
if a.reg == "div" and dref is None:
    gd = torch.Generator(device=dev).manual_seed(4242); rd = np.random.default_rng(4242)
    sel = rd.choice(N, 2 if a.smoke_batches else 8, replace=False)
    unet.eval(); dref = {}
    for tt in DREF_GRID:
        vals = []
        for j in sel:
            x0i = minmax_pm1(Xg[int(j):int(j) + 1].float().unsqueeze(1))
            abf = float(ac[tt]); x = math.sqrt(abf) * x0i + math.sqrt(1 - abf) * torch.randn(x0i.shape, generator=gd, device=dev)
            v = (torch.randint(0, 2, x.shape, generator=gd, device=dev) * 2 - 1).float()
            _, d1, _ = div_terms(x, tt, v, v, int(Yg[int(j)]), 1.0, backward=False)
            vals.append(d1)
        dref[str(tt)] = float(np.sqrt(np.mean(np.square(vals))))
    unet.train()
    log("D_ref(t) (RMS of v.J_eps v at the start): " + ", ".join(f"{k}: {v:.3g}" for k, v in dref.items()))

if a.reg != "none" and lam is None:
    gcal = torch.Generator(device=dev).manual_seed(999); rcal = np.random.default_rng(999)
    pcal = torch.randperm(N, generator=torch.Generator().manual_seed(999))
    M = 2 if a.smoke_batches else a.calib_groups
    SC = 1024.0
    nb_list, nr_list = [], []
    unet.train()
    for mgrp in range(M):
        opt.zero_grad(set_to_none=True)
        rb = [torch.zeros_like(p) for p in params] if a.reg == "sn" else None
        for k in range(a.accum):
            x0, t, cond, eps_hat, loss, *_ = micro_batch(pcal, mgrp * a.accum + k, gcal)
            if a.reg == "sn":
                P, nm = sn_value(t, eps_hat)
                if nm:
                    gr = torch.autograd.grad(P * SC / a.accum, params, retain_graph=True, allow_unused=True)
                    for b_, g_ in zip(rb, gr):
                        if g_ is not None:
                            b_ += g_
            (loss * SC / a.accum).backward()
        gb = [p.grad / SC if p.grad is not None else None for p in params]
        nb_list.append(math.sqrt(sqnorm(gb)))
        if a.reg == "sn":
            nr = math.sqrt(sqnorm(rb)) / SC
            del rb
        else:
            opt.zero_grad(set_to_none=True)
            tt, x, v1, v2 = draw_div(x0[0:1].detach(), gcal, rcal)
            unet.eval()
            div_terms(x, tt, v1, v2, float(cond[0].item()), 1.0 / fc.sref_interp(dref, tt) ** 2)
            unet.train()
            nr = math.sqrt(sqnorm([p.grad for p in params]))
        if nr > 0 and math.isfinite(nr):
            nr_list.append(nr)
        del gb
    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    lam = a.rho * float(np.median(nb_list)) / float(np.median(nr_list)) if nr_list else 0.0
    log(f"CALIBRATION: median |g_base| {np.median(nb_list):.4g} | median |g_reg(lam=1)| "
        f"{np.median(nr_list) if nr_list else float('nan'):.4g} over {len(nr_list)}/{M} groups -> lambda = {lam:.6g} (rho {a.rho})")
lam = float(lam or 0.0)
CAP = 2.0 * a.rho

spe = N // a.bs
if a.smoke_batches:
    spe = min(spe, a.smoke_batches)
durs = [h["minutes"] * 60 for h in history[-3:]]
rlog = open(os.path.join(a.ckpt_dir, f"{a.arm}_reg_log.jsonl"), "a")
reason = "max epochs reached"
bad = 0
regbuf = [torch.zeros_like(p) for p in params] if a.reg == "sn" else None
for epoch in range(start_epoch, a.epochs):
    used_h = (time.time() - T0) / 3600
    if durs and used_h + 1.15 * max(durs[-3:]) / 3600 + 0.2 > a.budget_h:
        reason = f"time budget ({used_h:.2f} h used of {a.budget_h} h)"; break
    gcpu = torch.Generator().manual_seed(a.seed * 1000 + epoch)
    perm = torch.randperm(N, generator=gcpu)
    gg = torch.Generator(device=dev).manual_seed(a.seed * 1000 + epoch)
    gfp = torch.Generator(device=dev).manual_seed(a.fp_seed * 1000 + epoch)
    rfp = np.random.default_rng(a.fp_seed * 1000 + epoch)
    unet.train(); opt.zero_grad(set_to_none=True)
    t_ep, t_win = time.time(), time.time()
    S = dict(loss=0.0, mse=0.0, ms=0.0, hf=0.0, reg=0.0, n_reg=0, reg_skipped=0, clipped=0, capped=0, n_steps=0)
    n_ok = 0
    for i in range(spe):
        x0, t, cond, eps_hat, loss, mse, ms, hf = micro_batch(perm, i, gg)
        if not torch.isfinite(loss):                                 # same handling as train_arm2.py
            bad += 1
            if bad > 50:
                raise RuntimeError("50 consecutive non-finite losses - stopping")
            continue
        bad = 0
        if a.reg == "sn" and lam > 0:
            P, nm = sn_value(t, eps_hat)
            if nm:
                gr = torch.autograd.grad(scaler.scale(P * (lam / a.accum)), params, retain_graph=True, allow_unused=True)
                for b_, g_ in zip(regbuf, gr):
                    if g_ is not None:
                        b_ += g_
                del gr
                S["reg"] += float(P.detach()) * a.bs; S["n_reg"] += nm          # mean per penalised sample
        scaler.scale(loss / a.accum).backward()
        step += 1
        if (i + 1) % a.accum == 0 or i == spe - 1:
            scale = float(scaler.get_scale())
            scaler.unscale_(opt)
            rec = {"epoch": epoch + 1, "opt_step": opt_steps + 1}
            g_base = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            nb = math.sqrt(sqnorm(g_base)); rec["base_norm"] = nb
            g_reg = None
            if a.reg == "sn" and lam > 0:
                g_reg = [b_ / scale for b_ in regbuf]
                for b_ in regbuf:
                    b_.zero_()
            elif a.reg == "div" and lam > 0:
                tt, x, v1, v2 = draw_div(x0[0:1].detach(), gfp, rfp)
                for p in params:
                    p.grad = None
                unet.eval()
                Lw, d1, d2 = div_terms(x, tt, v1, v2, float(cond[0].item()), lam / fc.sref_interp(dref, tt) ** 2)
                unet.train()
                g_reg = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                rec.update({"t": tt, "L_reg_norm": Lw / lam, "d1": d1, "d2": d2})
                if math.isfinite(Lw):
                    S["reg"] += Lw / lam; S["n_reg"] += 1
            if g_reg is not None and sqnorm(g_reg) == 0.0:          # no penalised sample in this step (e.g. small-t arm)
                g_reg = None; rec["reg_active"] = False
                for p, gb in zip(params, g_base):                    # make sure .grad holds the base gradient
                    p.grad = gb.clone() if gb is not None else None
            if g_reg is not None:
                nr = math.sqrt(sqnorm(g_reg))
                ok = math.isfinite(nr)
                rec["reg_active"] = True
                cap_f = min(1.0, CAP * nb / (nr + 1e-30)) if ok else 0.0
                if ok and cap_f < 1.0:
                    for g_ in g_reg:
                        if g_ is not None:
                            g_.mul_(cap_f)
                    S["capped"] += 1
                rec.update({"reg_norm_weighted": nr, "cap_factor": cap_f, "reg_ok": ok,
                            "cos_base_reg": dotl(g_base, g_reg) / (nb * nr * cap_f + 1e-30) if ok else None})
                for p, gb, gr_ in zip(params, g_base, g_reg):
                    use_r = ok and gr_ is not None
                    p.grad = None if (gb is None and not use_r) else \
                        (gb + gr_ if (gb is not None and use_r) else (gb.clone() if gb is not None else gr_.clone()))
                if not ok:
                    S["reg_skipped"] += 1; g_reg = None
            comb = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            rec["combined_norm"] = comb; rec["clip_factor"] = min(1.0, 1.0 / (comb + 1e-6))
            S["clipped"] += int(comb > 1.0); S["n_steps"] += 1
            un, d_base, d_reg = adam_update_stats(g_base, g_reg)
            rec["update_norm"] = un; rec["pred_dL_base"] = d_base
            if g_reg is not None and lam > 0:
                rec["pred_dL_reg_norm"] = d_reg / lam
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            opt_steps += 1
            rlog.write(json.dumps(rec) + "\n")
            if opt_steps % 25 == 0:
                rlog.flush()
            del g_base, g_reg
        n_ok += 1
        for k_, v_ in (("loss", loss), ("mse", mse), ("ms", ms), ("hf", hf)):
            S[k_] += float(v_.detach())
        if (i + 1) % a.log_every == 0:
            ips = a.log_every / (time.time() - t_win); t_win = time.time()
            m = {k: S[k] / n_ok for k in ("loss", "mse", "ms", "hf")}
            log(f"ep {epoch + 1} it {i + 1}/{spe} step {step} loss {m['loss']:.4f} mse {m['mse']:.4f} ms {m['ms']:.4f} "
                f"hf {m['hf']:.4f} | clipped {S['clipped']}/{S['n_steps']} capped {S['capped']}"
                + (f" | reg {S['reg'] / max(S['n_reg'], 1):.4f}" if a.reg != "none" else "") + f" | {ips:.2f} it/s")
    dur = time.time() - t_ep; durs.append(dur)
    m = {k: S[k] / max(n_ok, 1) for k in ("loss", "mse", "ms", "hf")}
    m["reg_mean"] = S["reg"] / S["n_reg"] if S["n_reg"] else None
    m.update({"reg_samples": S["n_reg"], "reg_skipped": S["reg_skipped"], "clipped_share": S["clipped"] / max(S["n_steps"], 1),
              "capped_share": S["capped"] / max(S["n_steps"], 1)})
    history.append({"epoch": epoch + 1, "step": step, "opt_steps": opt_steps, **m, "minutes": dur / 60})
    save_atomic({"unet": unet.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                 "step": step, "opt_steps": opt_steps, "args": vars(a), "history": history, "init_epoch": init_epoch,
                 "lam_cal": lam, "dref": dref}, os.path.join(a.ckpt_dir, f"{a.arm}_last.pt"))
    snap = {"unet": {k: v.detach().half().cpu() for k, v in unet.state_dict().items()}, "epoch": epoch + 1, "step": step,
            "arm": a.arm, "lam": lam, "reg": a.reg, "rho": a.rho, "init_epoch": init_epoch}
    if epoch + 1 == a.epochs or str(epoch + 1) in a.snap_epochs.split(","):
        save_atomic(snap, os.path.join(a.ckpt_dir, f"{a.arm}_w_ep{epoch + 1:03d}.pt"))
    del snap
    with open(os.path.join(a.ckpt_dir, f"{a.arm}_history.json"), "w") as fh:
        json.dump(history, fh, indent=1)
    rlog.flush()
    log(f"SAVED epoch={epoch + 1} step={step} opt_steps={opt_steps} loss={m['loss']:.4f} mse={m['mse']:.4f} "
        + (f"reg={m['reg_mean']:.4f} " if m["reg_mean"] is not None else "")
        + f"clipped {m['clipped_share']:.2f} capped {m['capped_share']:.2f} | {dur / 60:.1f} min/epoch | "
          f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
rlog.close()
log(f"FINISHED epochs_completed={history[-1]['epoch'] if history else start_epoch} step={step} lambda={lam:.6g} reason={reason}")
