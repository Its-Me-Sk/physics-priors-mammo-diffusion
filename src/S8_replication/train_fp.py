# train_fp.py - S3 (and later Phase C): continue training a clean baseline WITH (lam > 0) or WITHOUT (lam = 0) the
# Fokker-Planck consistency term. The BASE recipe is EXACTLY train_arm2.py (Experiment 2): 0.7 MSE + 0.2 (1-MS-SSIM)
# + 0.05 HF on x0_hat, per-image min-max to [-1,1], shear/scale/flip augmentation, 15 % label dropout (-1),
# batch 2 x grad-accum 4, Adam 1e-5, fp16 autocast + GradScaler, global grad-norm clip 1.0, no EMA, same data order
# (seed*1000+epoch). The physics-NLF hinge of Experiment 2 is NOT used (lam there = 0 for the baseline anyway).
# FP term (only if lam > 0), once per optimizer step, on image 0 of the LAST micro-batch of the accumulation group,
# with that image's training label (or -1 if it was dropped):
#   t ~ U{tmin..tmax}, x_t = sqrt(abar) x0 + sqrt(1-abar) eps, one unit direction u, two Rademacher probes v1, v2,
#   L_FP = r(u,v1) r(u,v2) / S_ref(t)^2 (fp_core: fp32 finite differences, fixed step table p(t)),
#   exact gradient by manual recompute (13 network evaluations). The network is in eval() mode for the FP evaluations
#   (no dropout / batch-norm in this UNet, so this only guarantees the function is deterministic).
#   Order: base grads unscaled (GradScaler.unscale_) -> + lam * grad L_FP -> ONE global clip at 1.0 -> Adam step.
#   All FP random draws come from their own generators (fp_seed*1000+epoch), identical in every arm, so the base
#   data / noise / dropout sequence is identical in every arm of a pair (paired design).
# Logged at every optimizer step (jsonl): base / weighted-FP / combined gradient norm before clipping, clip factor,
# cos(base, FP), the norm of the exact Adam update, and the first-order predicted change of the normalised FP loss
# and of the base (batch) loss along that update - computed from Adam's own update formula (no model copy).
import os, sys, glob, time, json, math, argparse
def _walk_find(name, root="/kaggle/input"):
    """S8: like glob('/kaggle/input/**/name') but skips the huge image folders (CMMD DICOM tree, CBIS jpeg)"""
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim

WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
sys.path.insert(0, WD)
import fp_core as fc

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--epochs", type=int, required=True)     # TOTAL epoch number to reach (warm start continues the numbering)
ap.add_argument("--init", required=True)                 # <baseline>_last.pt (weights + optimizer + scaler) for a fresh arm
ap.add_argument("--sref_json", required=True)            # A8_fp_clean.json from S2 (S_ref table)
ap.add_argument("--seed", type=int, default=42)          # data order / noise seed - identical for all arms of a pair
ap.add_argument("--fp_seed", type=int, default=7919)
ap.add_argument("--tmin", type=int, default=10)
ap.add_argument("--tmax", type=int, default=300)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--p_uncond", type=float, default=0.15)
ap.add_argument("--budget_h", type=float, default=10.5)
ap.add_argument("--log_every", type=int, default=200)
ap.add_argument("--smoke_batches", type=int, default=0)
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
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

d = np.load(os.path.join(os.environ.get("PA_TMP", "/tmp"), "train_cache.npz"))   # S8: dataset switch (default = CBIS-DDSM)
X = torch.from_numpy(d["x"]); Y = torch.from_numpy(d["y"]).long(); N = X.shape[0]
Xg, Yg = X.to(dev), Y.to(dev)
log(f"data {N} images {tuple(X.shape[1:])} | lam={a.lam} | FP t in [{a.tmin},{a.tmax}], once per optimizer step | "
    f"{torch.cuda.get_device_name(0)}")

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
LOGAB = torch.log(_m.scheduler.alphas_cumprod.detach().double().cpu())
del _m
T = ac.numel()
opt = torch.optim.Adam(unet.parameters(), lr=a.lr)
scaler = torch.amp.GradScaler("cuda")
params = [p for p in unet.parameters()]
SREF = json.load(open(a.sref_json))["S_ref_table"]
log("S_ref table: " + ", ".join(f"{k}: {float(v):.3g}" for k, v in SREF.items()))

# ---------------- resume this arm if a checkpoint exists, else warm start from the baseline ----------------
own = [os.path.join(a.ckpt_dir, f"{a.arm}_last.pt")]
if not a.smoke_batches:
    own += _walk_find(f"{a.arm}_last.pt")
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
    for k in ("lam", "lr", "bs", "accum", "seed", "p_uncond", "fp_seed", "tmin", "tmax"):
        if ck["args"].get(k) != getattr(a, k):
            log(f"!! WARNING {k} differs: checkpoint={ck['args'].get(k)} now={getattr(a, k)}")
unet.load_state_dict(ck["unet"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
start_epoch, step, opt_steps = int(ck["epoch"]), int(ck["step"]), int(ck["opt_steps"])
history = ck.get("history", []) if cands else []
init_epoch = int(ck.get("init_epoch", ck["epoch"])) if cands else int(ck["epoch"])
del ck; torch.cuda.empty_cache()
n_opt = sum(1 for s in opt.state.values() if s)
chk = float(sum(p.detach().float().sum() for p in params[:5]))
log(f"START epoch={start_epoch} step={step} opt_steps={opt_steps} opt_states={n_opt} lam={a.lam} checksum={chk:.6f} "
    f"init_epoch={init_epoch} from={src}")
if n_opt == 0:
    log("!! WARNING: optimizer state is EMPTY -> this is not a true continuation of the baseline")

def sqnorm(gs):
    return sum(float((g.float() ** 2).sum()) for g in gs if g is not None)

@torch.no_grad()
def adam_update_stats(g_base, g_fp):
    """exact torch.optim.Adam update (no weight decay / amsgrad) for the CURRENT (clipped) .grad, tensor by tensor:
    returns |update|, g_base . update, g_fp . update (first-order predicted change of each loss)"""
    gr = opt.param_groups[0]; b1, b2 = gr["betas"]; lr, eps = gr["lr"], gr["eps"]
    nrm2, d_base, d_fp = 0.0, 0.0, 0.0
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
        if g_fp is not None and g_fp[j] is not None:
            d_fp += float((g_fp[j] * upd).sum())
    return math.sqrt(nrm2), d_base, d_fp

spe = N // a.bs
if a.smoke_batches:
    spe = min(spe, a.smoke_batches)
durs = [h["minutes"] * 60 for h in history[-3:]]
fplog = open(os.path.join(a.ckpt_dir, f"{a.arm}_fp_log.jsonl"), "a")
reason = "max epochs reached"
bad = 0
for epoch in range(start_epoch, a.epochs):
    used_h = (time.time() - T0) / 3600
    if durs and used_h + 1.15 * max(durs[-3:]) / 3600 + 0.2 > a.budget_h:
        reason = f"time budget ({used_h:.2f} h used of {a.budget_h} h)"; break
    gcpu = torch.Generator().manual_seed(a.seed * 1000 + epoch)
    perm = torch.randperm(N, generator=gcpu)
    gg = torch.Generator(device=dev).manual_seed(a.seed * 1000 + epoch)
    gfp = torch.Generator(device=dev).manual_seed(a.fp_seed * 1000 + epoch)          # FP draws: same in every arm
    rfp = np.random.default_rng(a.fp_seed * 1000 + epoch)
    unet.train(); opt.zero_grad(set_to_none=True)
    t_ep, t_win = time.time(), time.time()
    S = dict(loss=0.0, mse=0.0, ms=0.0, hf=0.0, fp=0.0, n_fp=0, fp_skipped=0, clipped=0, n_steps=0)
    n_ok = 0
    for i in range(spe):
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
        loss = 0.7 * mse + 0.2 * ms + 0.05 * hf                     # exact original loss
        if not torch.isfinite(loss):                                 # same handling as train_arm2.py
            bad += 1
            if bad > 50:
                raise RuntimeError("50 consecutive non-finite losses - stopping")
            continue
        bad = 0
        scaler.scale(loss / a.accum).backward()
        step += 1
        if (i + 1) % a.accum == 0 or i == spe - 1:
            scaler.unscale_(opt)
            rec = {"epoch": epoch + 1, "opt_step": opt_steps + 1}
            g_base = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            nb = math.sqrt(sqnorm(g_base)); rec["base_norm"] = nb
            g_fp = None
            if a.lam > 0:
                # ---- one FP sample: image 0 of this micro-batch, its own random draws ----
                tf = int(rfp.integers(a.tmin, a.tmax + 1))
                abf = float(ac[tf])
                xf0 = x0[0:1].detach()
                xf = math.sqrt(abf) * xf0 + math.sqrt(1 - abf) * torch.randn(xf0.shape, generator=gfp, device=dev)
                u = torch.randn(xf.shape, generator=gfp, device=dev); u = u / u.norm()
                v1 = (torch.randint(0, 2, xf.shape, generator=gfp, device=dev) * 2 - 1).float()
                v2 = (torch.randint(0, 2, xf.shape, generator=gfp, device=dev) * 2 - 1).float()
                lab = float(cond[0].item())
                for p in params:
                    p.grad = None
                w = a.lam / fc.sref_interp(SREF, tf) ** 2
                unet.eval()
                Lw, r1, r2, A = fc.fp_loss_and_grad(fc.make_score_fn(unet, ac, lab), xf, tf, u, v1, v2,
                                                    fc.p_of_t(tf), 1, fc.beta_eff(LOGAB, tf), w=w)
                unet.train()
                g_fp = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                nf = math.sqrt(sqnorm(g_fp))
                ok = math.isfinite(Lw) and math.isfinite(nf)
                for p, gb, gf in zip(params, g_base, g_fp):
                    use_f = ok and gf is not None
                    p.grad = None if (gb is None and not use_f) else \
                        (gb + gf if (gb is not None and use_f) else (gb.clone() if gb is not None else gf.clone()))
                rec.update({"t": tf, "label": lab, "L_fp_norm": Lw / a.lam, "r1": r1, "r2": r2, "lhs": A,
                            "fp_norm_weighted": nf, "fp_ok": ok,
                            "cos_base_fp": (sum(float((x_ * y_).sum()) for x_, y_ in zip(g_base, g_fp)
                                                if x_ is not None and y_ is not None) / (nb * nf + 1e-30)) if ok else None})
                if ok:
                    S["fp"] += Lw / a.lam; S["n_fp"] += 1
                else:
                    S["fp_skipped"] += 1; g_fp = None
            comb = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            rec["combined_norm"] = comb; rec["clip_factor"] = min(1.0, 1.0 / (comb + 1e-6))
            S["clipped"] += int(comb > 1.0); S["n_steps"] += 1
            un, d_base, d_fp = adam_update_stats(g_base, g_fp)
            rec["update_norm"] = un
            rec["pred_dL_base"] = d_base                                   # base (batch) loss gradient . update
            if g_fp is not None:
                rec["pred_dL_fp_norm"] = d_fp / a.lam                      # normalised FP loss gradient . update
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            opt_steps += 1
            fplog.write(json.dumps(rec) + "\n")
            if opt_steps % 25 == 0:
                fplog.flush()
            del g_base, g_fp
        n_ok += 1
        for k_, v_ in (("loss", loss), ("mse", mse), ("ms", ms), ("hf", hf)):
            S[k_] += float(v_.detach())
        if (i + 1) % a.log_every == 0:
            ips = a.log_every / (time.time() - t_win); t_win = time.time()
            m = {k: S[k] / n_ok for k in ("loss", "mse", "ms", "hf")}
            log(f"ep {epoch + 1} it {i + 1}/{spe} step {step} loss {m['loss']:.4f} mse {m['mse']:.4f} ms {m['ms']:.4f} "
                f"hf {m['hf']:.4f} | clipped {S['clipped']}/{S['n_steps']} steps"
                + (f" | FP {S['fp'] / max(S['n_fp'], 1):+.4f} ({S['n_fp']} samples, {S['fp_skipped']} skipped)" if a.lam > 0 else "")
                + f" | {ips:.2f} it/s")
    dur = time.time() - t_ep; durs.append(dur)
    m = {k: S[k] / max(n_ok, 1) for k in ("loss", "mse", "ms", "hf")}
    m["fp_norm_mean"] = S["fp"] / S["n_fp"] if S["n_fp"] else None
    m.update({"fp_samples": S["n_fp"], "fp_skipped": S["fp_skipped"], "clipped_share": S["clipped"] / max(S["n_steps"], 1)})
    history.append({"epoch": epoch + 1, "step": step, "opt_steps": opt_steps, **m, "minutes": dur / 60})
    save_atomic({"unet": unet.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                 "step": step, "opt_steps": opt_steps, "args": vars(a), "history": history, "init_epoch": init_epoch},
                os.path.join(a.ckpt_dir, f"{a.arm}_last.pt"))
    snap = {"unet": {k: v.detach().half().cpu() for k, v in unet.state_dict().items()}, "epoch": epoch + 1, "step": step,
            "arm": a.arm, "lam": a.lam, "init_epoch": init_epoch}
    save_atomic(snap, os.path.join(a.ckpt_dir, f"{a.arm}_w_ep{epoch + 1:03d}.pt")); del snap
    with open(os.path.join(a.ckpt_dir, f"{a.arm}_history.json"), "w") as fh:
        json.dump(history, fh, indent=1)
    fplog.flush()
    log(f"SAVED epoch={epoch + 1} step={step} opt_steps={opt_steps} loss={m['loss']:.4f} mse={m['mse']:.4f} "
        + (f"FP={m['fp_norm_mean']:+.4f} " if m["fp_norm_mean"] is not None else "")
        + f"clipped {m['clipped_share']:.2f} | {dur / 60:.1f} min/epoch | peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
fplog.close()
log(f"FINISHED epochs_completed={history[-1]['epoch'] if history else start_epoch} step={step} reason={reason}")
