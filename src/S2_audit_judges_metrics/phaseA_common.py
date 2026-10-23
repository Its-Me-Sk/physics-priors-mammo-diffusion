# phaseA_common.py - shared helpers for the Phase A diagnostics (no training happens in Phase A).
# Model, schedule, data domain and conditioning are taken from the Experiment-2 code (train_arm2.py / eval_arm.py):
#   * model: model_architectures.MonaiDDPM().unet (MONAI-Generative DiffusionModelUNet, flash attention off)
#   * schedule: MonaiDDPM().scheduler.alphas_cumprod (linear beta 1e-4..0.02, T=1000)
#   * images: per-image min-max to [-1,1]; physics statistic on (x+1)/2
#   * conditioning: context = label as float (B,1,1); -1 = null label (classifier-free guidance)
import os, sys, glob, json, math, time
import numpy as np
import torch

WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
if WD not in sys.path:
    sys.path.insert(0, WD)
OUT = os.environ.get("PA_OUT", "/kaggle/working/phaseA")
os.makedirs(OUT, exist_ok=True)


def log(tag, *m):
    print(f"[{tag}]", *m, flush=True)


_INDEX = None


def find(name, roots=("/kaggle/input", "/kaggle/working")):
    """all files with this exact name in the attached inputs (one directory walk per process; image folders skipped)"""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for r in roots:
            for dp, dns, fns in os.walk(r):
                dns[:] = [d for d in dns if d not in ("jpeg", "U24_synthesis", ".git", "phaseA")]
                for fn in fns:
                    _INDEX.setdefault(fn, []).append(os.path.join(dp, fn))
    return sorted(set(_INDEX.get(name, [])))


def main_ckpt():
    """full-data Experiment-2 baseline, epoch 40 (the main Phase A checkpoint)"""
    h = find("base2_w_ep040.pt")
    if not h:
        raise SystemExit("base2_w_ep040.pt not found: Add Input -> Your Work -> the Experiment 2 notebook "
                         "(notebook64a7f79592), then run again")
    return h[0]


def clean_ckpt(seed, epoch=40):
    """clean-data baseline (step 2 of the timeline): base_clean_s<seed>_w_ep<epoch>.pt"""
    h = find(f"base_clean_s{seed}_w_ep{epoch:03d}.pt")
    if not h:
        raise SystemExit(f"base_clean_s{seed}_w_ep{epoch:03d}.pt not found: Add Input -> Your Work -> the clean-baseline "
                         "notebook (its latest output version), then run again")
    return h[0]


def optional_25_ckpt():
    """any 25 %-data baseline at epoch 45 from Experiment 3 or 4 (optional)"""
    for s in (42, 43, 44, 45, 46, 47, 48):
        h = find(f"base3_s{s}_w_ep045.pt")
        if h:
            return h[0]
    return None


def schedule(dev):
    from model_architectures import MonaiDDPM
    m = MonaiDDPM(lr=1e-5, T=1000)
    ac = m.scheduler.alphas_cumprod.detach().double().to(dev)
    unet = m.unet
    del m
    return unet, ac


def load_model(path, dev, dtype=torch.float32):
    """returns (unet in eval mode, alphas_cumprod float64, snapshot meta)"""
    snap = torch.load(path, map_location="cpu", weights_only=False)
    unet, ac = schedule(dev)
    sd = snap["unet"] if "unet" in snap else snap["models"]["net"]
    unet.load_state_dict({k: v.float() for k, v in sd.items()})
    unet = unet.to(dev).to(dtype).eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    meta = {k: snap[k] for k in ("epoch", "step", "lam", "tau") if k in snap}
    meta["nlf"] = snap.get("nlf")
    meta["path"] = path
    del snap
    return unet, ac, meta


def load_cache(split):
    d = np.load(f"/tmp/{split}_cache.npz")
    return d["x"], d["y"]


def load_clean_meta(split):
    """clean cache WITH patient ids and md5 - same order as /tmp/<split>_cache.npz (written by prepare_clean.py)"""
    d = np.load(f"/kaggle/working/clean_data/{split}_cache.npz")
    return d["x"], d["y"], d["patient"].astype(str), d["md5"].astype(str)


def val_mask(patients, frac=0.15, seed=0):
    """patient-level VALIDATION split of the clean TRAIN set (classifiers, gates, guidance choice only).
    The diffusion model itself was trained on all of train. Fixed seed -> identical in every script/session."""
    ups = sorted(set(patients))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(ups, int(round(frac * len(ups))), replace=False).tolist())
    return np.array([p in val for p in patients])


def minmax_pm1(x):
    """x: torch (B,1,H,W) any range -> per-image min-max to [-1,1] (training domain)"""
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return 2 * (x - mn) / (mx - mn + 1e-8) - 1


def to_model_domain(x_u8, dev):
    return minmax_pm1(torch.from_numpy(x_u8).float().unsqueeze(1).to(dev))


def ctx_of(labels, dev, dtype=torch.float32):
    """labels: list/array of 0, 1 or -1 (null) -> context tensor (B,1,1)"""
    return torch.as_tensor(np.asarray(labels, dtype=np.float32), device=dev).view(-1, 1, 1).to(dtype)


def eps_pred(unet, x, t_int, ctx, amp=False, chunk=8):
    """epsilon prediction. amp=True: exactly as in training (fp32 weights + fp16 autocast); amp=False: the weights'
    own dtype. Large batches are split into chunks (the non-flash attention at 64x64 needs ~0.5 GB per image)."""
    if chunk and x.shape[0] > chunk:
        if ctx.shape[0] == 1:
            ctx = ctx.expand(x.shape[0], *ctx.shape[1:])
        return torch.cat([eps_pred(unet, x[i:i + chunk], t_int, ctx[i:i + chunk], amp, None)
                          for i in range(0, x.shape[0], chunk)])
    tb = torch.full((x.shape[0],), int(t_int), device=x.device, dtype=torch.long)
    if amp:
        with torch.autocast("cuda", dtype=torch.float16):
            out = unet(x, timesteps=tb, context=ctx)
        return out.float()
    return unet(x, timesteps=tb, context=ctx)


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
