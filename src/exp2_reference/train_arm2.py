# train_arm2.py - Experiment 2: PINN ablation on the ORIGINAL training recipe (monai_ddpm.py / train.py).
# Same as the original: 0.7*MSE + 0.2*(1-MS-SSIM) + 0.05*HF loss on x0_hat, per-image min-max to [-1,1],
# shear/scale/flip augmentation, 15% label dropout (-1), batch 2 x grad-accum 4, Adam, clip 1.0, fp16.
# The ONLY difference between arms: PINN adds lam * physics-consistency hinge term.
import os, sys, glob, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim

WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
sys.path.insert(0, WD)
from pinn_physics import PhysicsNLF

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)                  # base2 | pinn2
ap.add_argument("--lam", type=float, default=0.0)        # physics weight, 0 = baseline
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--p_uncond", type=float, default=0.15)
ap.add_argument("--tau_q", type=float, default=0.9)      # hinge threshold = this quantile of REAL residuals
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--budget_h", type=float, default=10.9)
ap.add_argument("--snap_every", type=int, default=5)
ap.add_argument("--smoke_batches", type=int, default=0)
ap.add_argument("--log_every", type=int, default=200)
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

def minmax_pm1(x):                                   # per-image min-max -> [-1,1] (= original z-score + ScaleIntensity)
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return 2 * (x - mn) / (mx - mn + 1e-8) - 1

# ---------------- original HighFreqLoss (copied from monai_ddpm.py) ----------------
_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX = (_sob[None, None] / 8).to(dev)
KY = (_sob.t()[None, None] / 8).to(dev)
def hf_loss(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean()

# ---------------- data (uint8 cache, kept on GPU) ----------------
d = np.load("/tmp/train_cache.npz")
X = torch.from_numpy(d["x"])
Y = torch.from_numpy(d["y"]).long()
N = X.shape[0]
Xg, Yg = X.to(dev), Y.to(dev)
log(f"data {N} images {tuple(X.shape[1:])} benign={int((Y == 0).sum())} malignant={int((Y == 1).sum())} "
    f"on {torch.cuda.get_device_name(0)}")

def augment(img, gg):
    # img: (B,1,H,W) float in 0..255. Effective original aug: shear +-0.1, scale +-5% (p=0.9), h-flip p=0.5
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

# ---------------- model: the repo's exact DiffusionModelUNet + noise schedule ----------------
from model_architectures import MonaiDDPM
torch.manual_seed(a.seed)                       # identical initial weights for both arms
_m = MonaiDDPM(lr=a.lr, T=1000)
unet = _m.unet.to(dev)
ac = _m.scheduler.alphas_cumprod.detach().float().to(dev)
del _m
T = ac.numel()
opt = torch.optim.Adam(unet.parameters(), lr=a.lr)
scaler = torch.amp.GradScaler("cuda")
nlf = PhysicsNLF(dev)
tau = None

# ---------------- resume ----------------
paths = [os.path.join(a.ckpt_dir, f"{a.arm}_last.pt")]
if not a.smoke_batches:
    paths += glob.glob(f"/kaggle/input/**/{a.arm}_last.pt", recursive=True)
cands = []
for p in paths:
    if os.path.exists(p):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            cands.append((int(ck["step"]), p))
            del ck
        except Exception as e:
            log("unreadable checkpoint (skipped):", p, repr(e)[:150])
start_epoch, step, opt_steps, history, resumed = 0, 0, 0, [], None
if cands:
    resumed = max(cands)[1]
    ck = torch.load(resumed, map_location="cpu", weights_only=False)
    for k in ("lam", "lr", "bs", "accum", "seed", "p_uncond", "tau_q"):
        if ck["args"].get(k) != getattr(a, k):
            log(f"!! WARNING {k} differs: checkpoint={ck['args'].get(k)} now={getattr(a, k)}")
    unet.load_state_dict(ck["unet"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    start_epoch, step, opt_steps, history = int(ck["epoch"]), int(ck["step"]), int(ck["opt_steps"]), ck["history"]
    nlf.load(ck["nlf"])
    tau = float(ck["tau"])
    del ck
    torch.cuda.empty_cache()

# ---------------- physics calibration on REAL images, in the training intensity domain ----------------
Xf = X.float()
mn = Xf.amin(dim=(1, 2), keepdim=True); mx = Xf.amax(dim=(1, 2), keepdim=True)
Xmm = ((Xf - mn) / (mx - mn + 1e-8) * 255).round().clamp(0, 255).to(torch.uint8)
del Xf
if nlf.p is None:
    rep = nlf.fit(Xmm)
    log(f"NLF fit: sigma^2 = {rep['b']:.3e} + {rep['a']:.3e}*mu + {rep['c']:.3e}*mu^2 | "
        f"R2(var)={rep['R2_var']:.3f} R2(logvar)={rep['R2_logvar']:.3f}")
with torch.no_grad():
    res_real = torch.cat([nlf.residual(Xmm[i:i + 32].to(dev).float().div(255).unsqueeze(1)).cpu()
                          for i in range(0, N, 32)])
if tau is None:
    tau = float(torch.quantile(res_real, a.tau_q))
    with open(os.path.join(a.ckpt_dir, f"nlf_fit_{a.arm}.json"), "w") as fh:
        json.dump({**rep, "tau": tau, "tau_q": a.tau_q, "real_mean_residual": float(res_real.mean())}, fh, indent=2)
log(f"physics residual of REAL images: mean={float(res_real.mean()):.4f} | hinge threshold tau(q={a.tau_q})={tau:.4f}")

n_opt = sum(1 for s in opt.state.values() if s)
chk = float(sum(p.detach().float().sum() for p in list(unet.parameters())[:5]))
log(f"START epoch={start_epoch} step={step} opt_states={n_opt} lam={a.lam} checksum={chk:.6f} "
    f"resumed={resumed or 'scratch'}")
if resumed and n_opt == 0:
    log("!! WARNING: checkpoint loaded but optimizer state is EMPTY -> not a true resume")

spe = N // a.bs
if a.smoke_batches:
    spe = min(spe, a.smoke_batches)
durs = [h["minutes"] * 60 for h in history[-3:]]
reason = "max epochs reached"
bad = 0
for epoch in range(start_epoch, a.epochs):
    used_h = (time.time() - T0) / 3600
    if durs and used_h + 1.15 * max(durs[-3:]) / 3600 + 0.2 > a.budget_h:
        reason = f"time budget ({used_h:.2f} h used of {a.budget_h} h)"
        break
    gcpu = torch.Generator().manual_seed(a.seed * 1000 + epoch)
    perm = torch.randperm(N, generator=gcpu)
    gg = torch.Generator(device=dev).manual_seed(a.seed * 1000 + epoch)
    unet.train()
    opt.zero_grad(set_to_none=True)
    t_ep, t_win = time.time(), time.time()
    S = dict(loss=0.0, comp=0.0, mse=0.0, ms=0.0, hf=0.0, phys=0.0, hinge=0.0)
    n_ok = 0
    for i in range(spe):
        idx = perm[i * a.bs:(i + 1) * a.bs].to(dev)
        x0 = minmax_pm1(augment(Xg[idx].float().unsqueeze(1), gg))
        y = Yg[idx]
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
        comp = 0.7 * mse + 0.2 * ms + 0.05 * hf                     # exact original loss
        w = ab.view(-1)
        if a.lam > 0:
            r = nlf.residual((x0_hat + 1) / 2)
            hinge = (w * F.relu(r - tau)).mean()
            loss = comp + a.lam * hinge
        else:
            with torch.no_grad():                                   # baseline: tracked, never trained on
                r = nlf.residual((x0_hat + 1) / 2)
                hinge = (w * F.relu(r - tau)).mean()
            loss = comp
        phys = (w * r).mean().detach()
        if not torch.isfinite(loss):
            bad += 1
            if bad > 50:
                raise RuntimeError("50 consecutive non-finite losses - stopping")
            continue
        bad = 0
        scaler.scale(loss / a.accum).backward()
        step += 1
        if (i + 1) % a.accum == 0 or i == spe - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1
        n_ok += 1
        for k, v in (("loss", loss), ("comp", comp), ("mse", mse), ("ms", ms), ("hf", hf), ("phys", phys), ("hinge", hinge)):
            S[k] += float(v.detach())
        if (i + 1) % a.log_every == 0:
            ips = a.log_every / (time.time() - t_win)
            t_win = time.time()
            m = {k: v / n_ok for k, v in S.items()}
            log(f"ep {epoch + 1} it {i + 1}/{spe} step {step} loss {m['loss']:.4f} comp {m['comp']:.4f} "
                f"mse {m['mse']:.4f} ms {m['ms']:.4f} hf {m['hf']:.4f} phys {m['phys']:.4f} "
                f"hinge {m['hinge']:.4f} | {ips:.2f} it/s")
    dur = time.time() - t_ep
    durs.append(dur)
    m = {k: v / max(n_ok, 1) for k, v in S.items()}
    history.append({"epoch": epoch + 1, "step": step, "opt_steps": opt_steps, **m, "minutes": dur / 60})
    save_atomic({"unet": unet.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "step": step, "opt_steps": opt_steps, "args": vars(a), "nlf": nlf.state(),
                 "tau": tau, "history": history}, os.path.join(a.ckpt_dir, f"{a.arm}_last.pt"))
    snap = {"unet": {k: v.detach().half().cpu() for k, v in unet.state_dict().items()},
            "epoch": epoch + 1, "step": step, "arm": a.arm, "lam": a.lam, "nlf": nlf.state(), "tau": tau}
    save_atomic(snap, os.path.join(a.ckpt_dir, f"{a.arm}_w_latest.pt"))
    if (epoch + 1) % a.snap_every == 0:
        save_atomic(snap, os.path.join(a.ckpt_dir, f"{a.arm}_w_ep{epoch + 1:03d}.pt"))
    del snap
    with open(os.path.join(a.ckpt_dir, f"{a.arm}_history.json"), "w") as fh:
        json.dump(history, fh, indent=1)
    left = a.epochs - epoch - 1
    log(f"SAVED epoch={epoch + 1} step={step} opt_steps={opt_steps} loss={m['loss']:.4f} comp={m['comp']:.4f} "
        f"mse={m['mse']:.4f} ms={m['ms']:.4f} hf={m['hf']:.4f} phys={m['phys']:.4f} hinge={m['hinge']:.4f} | "
        f"{dur / 60:.1f} min/epoch | peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB | "
        f"~{left * dur / 3600:.1f} h for {left} more")
log(f"FINISHED epochs_completed={history[-1]['epoch'] if history else start_epoch} step={step} reason={reason}")
