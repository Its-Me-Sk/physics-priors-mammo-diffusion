# ===================== CELL 2 — data cache + Experiment-2 code + GPU check (no training) =====================
import os, sys, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)

Path("/tmp/prepare.py").write_text(r'''
import os, subprocess
import numpy as np, pandas as pd
from PIL import Image
if os.path.exists("/tmp/train_cache.npz") and os.path.exists("/tmp/test_cache.npz"):
    print("caches already built"); raise SystemExit(0)
hit = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True,
                     capture_output=True, text=True).stdout.split()
if not hit:
    raise SystemExit("CBIS-DDSM not attached: Add Input -> CBIS-DDSM Breast Cancer Image Dataset")
RAW = os.path.dirname(os.path.dirname(hit[0]))
info = pd.read_csv(os.path.join(RAW, "csv/dicom_info.csv"))
lookup = info[info["SeriesDescription"] == "full mammogram images"].set_index("PatientName")["image_path"].to_dict()
for split, csvs in [("train", ["csv/calc_case_description_train_set.csv", "csv/mass_case_description_train_set.csv"]),
                    ("test",  ["csv/calc_case_description_test_set.csv",  "csv/mass_case_description_test_set.csv"])]:
    xs, skipped = {0: [], 1: []}, 0
    for c in csvs:
        for _, row in pd.read_csv(os.path.join(RAW, c)).iterrows():
            lab = 1 if row["pathology"] == "MALIGNANT" else 0
            rel = lookup.get(row["image file path"].split("/")[0])
            if rel is None:
                skipped += 1; continue
            p = os.path.join(RAW, "jpeg", rel.replace("CBIS-DDSM/jpeg/", ""))
            if not os.path.exists(p):
                skipped += 1; continue
            xs[lab].append(np.array(Image.open(p).convert("L").resize((256, 256)), dtype=np.uint8))
    if split == "train" and len(xs[0]) > 1300:          # same balancing as before: 1300 benign, all malignant
        keep = np.sort(np.random.default_rng(42).choice(len(xs[0]), 1300, replace=False))
        xs[0] = [xs[0][i] for i in keep]
    x = np.stack(xs[0] + xs[1]); y = np.array([0] * len(xs[0]) + [1] * len(xs[1]), dtype=np.int64)
    np.savez(f"/tmp/{split}_cache.npz", x=x, y=y)
    print(f"{split}: {x.shape} benign={int((y == 0).sum())} malignant={int((y == 1).sum())} skipped={skipped}")
''')
subprocess.run([sys.executable, "/tmp/prepare.py"], check=True)

Path("pinn_physics.py").write_text(r'''# pinn_physics.py - physics-informed soft constraint (PINN-style residual) for mammogram DDPM.
# Physical model: signal-dependent noise-level function (NLF) of the X-ray film/digitiser chain
#   sigma^2(mu) = b + a*mu + c*mu^2
#   b: Gaussian read/digitiser noise, a*mu: Poisson (quantum) noise, c*mu^2: multiplicative film-grain/gain term
# The coefficients are CALIBRATED on real training images, then the residual between the NLF of a
# generated image and the physical NLF is added to the diffusion loss (lambda * residual).
import torch
import torch.nn.functional as F


class PhysicsNLF:
    def __init__(self, device, k=7, K=16, lo=0.08, hi=0.95, mask_lo=0.04):
        ax = torch.arange(7, device=device, dtype=torch.float32) - 3
        g = torch.exp(-ax ** 2 / 2.0)
        self.g = g / g.sum()
        self.k, self.K, self.lo, self.hi, self.mask_lo = k, K, lo, hi, mask_lo
        self.c = torch.linspace(lo, hi, K, device=device)
        self.bw = float(hi - lo) / (K - 1) * 0.6
        self.p = None                     # (b, a, c) of the calibrated NLF

    def blur(self, I):
        g = self.g
        I = F.conv2d(F.pad(I, (3, 3, 0, 0), mode="reflect"), g.view(1, 1, 1, 7))
        return F.conv2d(F.pad(I, (0, 0, 3, 3), mode="reflect"), g.view(1, 1, 7, 1))

    def stats(self, I):
        # I: (B,1,H,W) intensities in [0,1]. Returns per-image binned high-frequency variance vs local mean.
        h = I - self.blur(I)
        mu = F.avg_pool2d(I, self.k, 1, self.k // 2, count_include_pad=False).flatten(1)
        v = F.avg_pool2d(h * h, self.k, 1, self.k // 2, count_include_pad=False).flatten(1)
        W = torch.exp(-(mu.unsqueeze(-1) - self.c) ** 2 / (2 * self.bw ** 2))
        W = W * (mu > self.mask_lo).unsqueeze(-1).float()        # ignore black background (air)
        mass = W.sum(1)                                           # (B,K)
        vbin = torch.einsum("bpk,bp->bk", W, v) / (mass + 1e-6)   # (B,K)
        return vbin, mass

    def model_var(self):
        b, a, c = self.p
        return (b + a * self.c + c * self.c ** 2).clamp_min(1e-7)

    def residual(self, I):
        # per-image physics residual: mass-weighted squared log-ratio of measured vs physical NLF
        vbin, mass = self.stats(I)
        r = (torch.log(vbin + 1e-5) - torch.log(self.model_var() + 1e-5)) ** 2
        w = mass / (mass.sum(1, keepdim=True) + 1e-6)
        return (w * r).sum(1)

    @torch.no_grad()
    def fit(self, X_u8, bs=32):
        dev = self.c.device
        tv = torch.zeros(self.K, device=dev, dtype=torch.float64)
        tm = torch.zeros(self.K, device=dev, dtype=torch.float64)
        for i in range(0, X_u8.shape[0], bs):
            I = X_u8[i:i + bs].to(dev).float().div(255).unsqueeze(1)
            vb, m = self.stats(I)
            tv += (vb * m).sum(0).double()
            tm += m.sum(0).double()
        V = tv / tm.clamp_min(1e-6)
        wt = tm / tm.sum()
        c = self.c.double()

        def wls(cols):
            A = torch.stack(cols, 1)
            th = torch.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * V))
            pred = (A @ th).clamp_min(1e-9)
            ss_res = (wt * (V - pred) ** 2).sum()
            ss_tot = (wt * (V - (wt * V).sum()) ** 2).sum()
            lr = torch.log(V) - torch.log(pred)
            lres = (wt * lr ** 2).sum()
            ltot = (wt * (torch.log(V) - (wt * torch.log(V)).sum()) ** 2).sum()
            return th, float(1 - ss_res / ss_tot), float(1 - lres / ltot)

        th_lin, r2_lin, lr2_lin = wls([torch.ones_like(c), c])
        th_q, r2_q, lr2_q = wls([torch.ones_like(c), c, c ** 2])
        self.p = (float(th_q[0]), float(th_q[1]), float(th_q[2]))
        return {"model": "sigma^2 = b + a*mu + c*mu^2 (Poisson-Gaussian + multiplicative)",
                "b": self.p[0], "a": self.p[1], "c": self.p[2],
                "R2_var": r2_q, "R2_logvar": lr2_q,
                "linear_poisson_gaussian_only": {"b": float(th_lin[0]), "a": float(th_lin[1]),
                                                  "R2_var": r2_lin, "R2_logvar": lr2_lin},
                "bin_centers": [float(x) for x in c], "measured_var": [float(x) for x in V],
                "bin_weight": [float(x) for x in wt]}

    @torch.no_grad()
    def mean_residual(self, X_u8, bs=32):
        dev = self.c.device
        tot, n = 0.0, 0
        for i in range(0, X_u8.shape[0], bs):
            I = X_u8[i:i + bs].to(dev).float().div(255).unsqueeze(1)
            tot += float(self.residual(I).sum())
            n += I.shape[0]
        return tot / max(n, 1)

    def state(self):
        return {"p": list(self.p), "k": self.k, "K": self.K, "lo": self.lo, "hi": self.hi, "mask_lo": self.mask_lo}

    def load(self, s):
        self.p = tuple(s["p"])
''')
Path("train_arm2.py").write_text(r'''# train_arm2.py - Experiment 2: PINN ablation on the ORIGINAL training recipe (monai_ddpm.py / train.py).
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
''')
Path("eval_arm.py").write_text(r'''# eval_arm.py - sample one arm with DDIM + classifier-free guidance and score it.
# Same seeds for both arms -> sample i of baseline and sample i of pinn start from the SAME noise (paired test).
import os, sys, glob, json, time, argparse
import numpy as np
import torch

WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
sys.path.insert(0, WD)
from pinn_physics import PhysicsNLF

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--epoch", type=int, default=0)          # 0 = latest snapshot
ap.add_argument("--n_per_class", type=int, default=200)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--gs", type=float, default=3.0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
ap.add_argument("--out", default="/kaggle/working/eval")
a = ap.parse_args()
TAG = f"[eval-{a.arm}]"
def log(*m):
    print(TAG, *m, flush=True)
dev = torch.device("cuda")
os.makedirs(a.out, exist_ok=True)

name = f"{a.arm}_w_latest.pt" if a.epoch == 0 else f"{a.arm}_w_ep{a.epoch:03d}.pt"
found = glob.glob(os.path.join(a.ckpt_dir, name)) + glob.glob(f"/kaggle/input/**/{name}", recursive=True)
if not found:
    raise SystemExit(f"{TAG} snapshot {name} not found in {a.ckpt_dir} or /kaggle/input")
snap = torch.load(found[0], map_location="cpu", weights_only=False)
log(f"snapshot {found[0]} | epoch={snap['epoch']} step={snap['step']} lam={snap['lam']}")

from model_architectures import MonaiDDPM
_m = MonaiDDPM(lr=1e-5, T=1000)
unet = _m.unet
ac = _m.scheduler.alphas_cumprod.detach().float().to(dev)
del _m
unet.load_state_dict({k: v.float() for k, v in snap["unet"].items()})
unet = unet.half().to(dev).eval()
T = ac.numel()
nlf = PhysicsNLF(dev)
nlf.load(snap["nlf"])

@torch.no_grad()
def ddim(label, n, gen):
    x = torch.randn(n, 1, 256, 256, generator=gen, device=dev)
    ts = torch.linspace(T - 1, 0, a.steps, device=dev).round().long()
    cond = torch.full((n, 1, 1), float(label), device=dev)
    ctx = torch.cat([-torch.ones_like(cond), cond]).half()
    for i, t in enumerate(ts):
        ab_t = ac[t]
        ab_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        tb = torch.full((2 * n,), int(t), device=dev, dtype=torch.long)
        out = unet(torch.cat([x, x]).half(), timesteps=tb, context=ctx).float()
        eu, ec = out.chunk(2)
        eps = eu + a.gs * (ec - eu)
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return ((x + 1) / 2).clamp(0, 1)

# ---------------- generate ----------------
t0 = time.time()
imgs, labels = [], []
for label in (0, 1):
    done, b = 0, 0
    while done < a.n_per_class:
        n = min(a.bs, a.n_per_class - done)
        gen = torch.Generator(device=dev).manual_seed(a.seed + 100000 * label + b)
        imgs.append(ddim(label, n, gen).cpu())
        labels += [label] * n
        done += n
        b += 1
    log(f"class {label}: {done} samples ({(time.time() - t0) / 60:.1f} min)")
G = torch.cat(imgs)                                    # (M,1,256,256) in [0,1]
L = torch.tensor(labels)

# ---------------- real test images ----------------
d = np.load("/tmp/test_cache.npz")
R = torch.from_numpy(d["x"]).float().div(255).unsqueeze(1)
RL = torch.from_numpy(d["y"]).long()

# ---------------- oracle classifier ----------------
root = "/kaggle/working/U24_synthesis"
sys.path.insert(0, root)
cwd = os.getcwd(); os.chdir(root)
from models import get_model
os.chdir(cwd)
op = glob.glob("/kaggle/input/**/oracle-checkpoint-final/last.pt", recursive=True)
res = {"arm": a.arm, "epoch": snap["epoch"], "step": snap["step"], "lam": snap["lam"],
       "n_per_class": a.n_per_class, "ddim_steps": a.steps, "guidance": a.gs}
correct = np.zeros(len(L), dtype=bool)
if op:
    orc = get_model("convnext_tiny", pretrained=False)
    orc.load_state_dict(torch.load(op[0], map_location="cpu", weights_only=False)["model_state"])
    orc = orc.half().to(dev).eval()
    def oracle_pred(I01):
        out = []
        for i in range(0, len(I01), 32):
            x = I01[i:i + 32].to(dev)
            mn = x.amin(dim=(1, 2, 3), keepdim=True); mx = x.amax(dim=(1, 2, 3), keepdim=True)
            x = ((x - mn) / (mx - mn + 1e-8) - 0.5) / 0.5
            with torch.no_grad():
                out.append(orc(x.half()).argmax(1).cpu())
        return torch.cat(out)
    pr = oracle_pred(R)
    res["oracle_real_acc"] = {c: float((pr[RL == k] == k).float().mean()) for k, c in enumerate(["benign", "malignant"])}
    pg = oracle_pred(G)
    correct = (pg == L).numpy()
    res["oracle_acc"] = {c: float(correct[L.numpy() == k].mean()) for k, c in enumerate(["benign", "malignant"])}
    res["oracle_acc"]["mean"] = float(correct.mean())
    log(f"oracle on REAL test: {res['oracle_real_acc']} | on generated: {res['oracle_acc']}")
else:
    log("oracle checkpoint not found (attach oracle-checkpoint-final) - skipping oracle accuracy")

# ---------------- physics residual (generated vs real) ----------------
def mean_res(I):
    return float(torch.cat([nlf.residual(I[i:i + 32].to(dev)).cpu() for i in range(0, len(I), 32)]).mean())
res["physics_residual_generated"] = mean_res(G)
res["physics_residual_real_test"] = mean_res(R)
log(f"physics residual: generated={res['physics_residual_generated']:.4f} real={res['physics_residual_real_test']:.4f}")

# ---------------- KID / FID ----------------
def rgb8(I):
    return (I.clamp(0, 1) * 255).round().to(torch.uint8).repeat(1, 3, 1, 1)
try:
    from torchmetrics.image.kid import KernelInceptionDistance
    kid = KernelInceptionDistance(subset_size=min(100, len(G), len(R))).to(dev)
    for i in range(0, len(R), 32):
        kid.update(rgb8(R[i:i + 32]).to(dev), real=True)
    for i in range(0, len(G), 32):
        kid.update(rgb8(G[i:i + 32]).to(dev), real=False)
    km, ks = kid.compute()
    res["KID_mean"], res["KID_std"] = float(km), float(ks)
    log(f"KID = {float(km):.4f} +- {float(ks):.4f}")
except Exception as e:
    log("KID skipped:", repr(e)[:200])
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(dev)
    for i in range(0, len(R), 32):
        fid.update(rgb8(R[i:i + 32]).to(dev), real=True)
    for i in range(0, len(G), 32):
        fid.update(rgb8(G[i:i + 32]).to(dev), real=False)
    res["FID"] = float(fid.compute())
    log(f"FID = {res['FID']:.2f}")
except Exception as e:
    log("FID skipped:", repr(e)[:200])

# ---------------- save ----------------
from PIL import Image
tag = f"{a.arm}_ep{snap['epoch']:03d}"
k = min(8, a.n_per_class)
rows = [np.concatenate([(G[j + c * a.n_per_class, 0].numpy() * 255).astype(np.uint8) for j in range(k)], axis=1)
        for c in (0, 1)]
Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(a.out, f"grid_{tag}.png"))
np.savez_compressed(os.path.join(a.out, f"samples_{tag}.npz"),
                    x=(G[:, 0].numpy() * 255).astype(np.uint8), y=L.numpy(), correct=correct)
with open(os.path.join(a.out, f"results_{tag}.json"), "w") as fh:
    json.dump(res, fh, indent=2)
log(f"EVAL DONE -> {a.out}/results_{tag}.json ({(time.time() - t0) / 60:.1f} min)")
''')
print("wrote pinn_physics.py, train_arm2.py, eval_arm.py")

print(subprocess.run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv",
                     shell=True, capture_output=True, text=True).stdout)
print(subprocess.run([sys.executable, "-c", "import torch; print('torch', torch.__version__, '| GPUs visible:', torch.cuda.device_count())"],
                     capture_output=True, text=True).stdout)

def run_parallel(jobs, poll_min=20):
    """jobs = [(gpu_index, [script, args...]), ...] -> runs them at the same time, one per GPU."""
    procs, outs = [], []
    for gpu, argv in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(p); outs.append([])
    def pump(p, buf, gpu):
        for line in p.stdout:
            buf.append(line)
            print(line if line.startswith("[") else f"[gpu{gpu}] {line}", end="", flush=True)
    def poll():
        while any(p.poll() is None for p in procs):
            time.sleep(poll_min * 60)
            if any(p.poll() is None for p in procs):
                q = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
                                   shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
                print(f"[nvidia-smi] {q}", flush=True)
    ths = [threading.Thread(target=pump, args=(p, b, g)) for p, b, (g, _) in zip(procs, outs, jobs)]
    for th in ths: th.start()
    threading.Thread(target=poll, daemon=True).start()
    for th in ths: th.join()
    return [p.wait() for p in procs], ["".join(b) for b in outs]
print("Cell 2 done.")
