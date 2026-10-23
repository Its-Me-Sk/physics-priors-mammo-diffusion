# s6_generate.py - S6 sample generation (= s5_generate.py + optional dynamic thresholding) (no training). Same DDIM-50 + classifier-free guidance sampler and the SAME
# noise bank as s3_eval / s4_eval (noise seed 2025 + 100000*label + batch, batch 16, benign first), so image i of every
# set starts from the same noise and label (paired comparisons between checkpoints / sampler settings).
#   --ckpts a.pt[,b.pt,...]  one snapshot, or several -> their weights are averaged (uniform checkpoint averaging)
#   --eps_scale s            Epsilon Scaling (Ning et al., ICLR 2024): the guided noise prediction is divided by s at
#                            every step (s = 1: unchanged sampler)
#   --dt p                   dynamic thresholding (Saharia et al. 2022, Imagen): at every step the predicted clean image
#                            x0 is clipped to [-s, s] and divided by s, s = max(1, p-th percentile of |x0| per image);
#                            p = 0 -> the original fixed clamp to [-1, 1]
import os, sys, time, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, schedule, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ckpts", required=True)
ap.add_argument("--n_per_class", type=int, default=100)
ap.add_argument("--gs", type=float, default=2.0)
ap.add_argument("--eps_scale", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--noise_seed", type=int, default=2025)
ap.add_argument("--dt", type=float, default=0.0)
a = ap.parse_args()
TAG = f"S6-gen-{a.name}"
dev = torch.device("cuda")
paths = a.ckpts.split(",")
unet, ac = schedule(dev)
avg = None
for p in paths:
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd["unet"] if "unet" in sd else sd["models"]["net"]
    if avg is None:
        avg = {k: v.float().clone() for k, v in sd.items()}
    else:
        for k in avg:
            avg[k] += sd[k].float()
    del sd
for k in avg:
    avg[k] /= len(paths)
unet.load_state_dict(avg); del avg
unet = unet.to(dev).half().eval()
for q in unet.parameters():
    q.requires_grad_(False)
acf = ac.float(); T = acf.numel()
log(TAG, f"{len(paths)} checkpoint(s) {'(weights averaged)' if len(paths) > 1 else ''}: {paths} | {a.n_per_class}/class | gs {a.gs} | "
         f"eps_scale {a.eps_scale} | dynamic threshold {a.dt or 'off'} | noise seed {a.noise_seed} | {a.steps} DDIM steps")

@torch.no_grad()
def ddim(label, n, gen):
    x = torch.randn(n, 1, 256, 256, generator=gen, device=dev)
    ts = torch.linspace(T - 1, 0, a.steps, device=dev).round().long()
    cond = torch.full((n, 1, 1), float(label), device=dev)
    for i, t in enumerate(ts):
        ab_t = acf[t]
        ab_prev = acf[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        tb = torch.full((2 * n,), int(t), device=dev, dtype=torch.long)
        ctx = torch.cat([-torch.ones_like(cond), cond]).half()
        eu, ec = unet(torch.cat([x, x]).half(), timesteps=tb, context=ctx).float().chunk(2)
        eps = (eu + a.gs * (ec - eu)) / a.eps_scale
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if a.dt > 0:
            sdt = torch.quantile(x0.abs().flatten(1), a.dt / 100.0, dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
            x0 = torch.maximum(torch.minimum(x0, sdt), -sdt) / sdt
        else:
            x0 = x0.clamp(-1, 1)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return ((x + 1) / 2).clamp(0, 1)

t0 = time.time(); imgs, labels = [], []
for label in (0, 1):
    done, b = 0, 0
    while done < a.n_per_class:
        k = min(16, a.n_per_class - done)
        out = ddim(label, k, torch.Generator(device=dev).manual_seed(a.noise_seed + 100000 * label + b))[:, 0].cpu()
        imgs.append((out.numpy() * 255).round().astype(np.uint8)); labels += [label] * k; done += k; b += 1
    log(TAG, f"label {label}: {done} images | {(time.time() - t0) / 60:.1f} min")
G = np.concatenate(imgs); L = np.array(labels, dtype=np.int64)
np.savez_compressed(os.path.join(OUT, f"gen_{a.name}.npz"), x=G, y=L)
from PIL import Image
kk = min(8, a.n_per_class)
Image.fromarray(np.concatenate([np.concatenate([G[j + c * a.n_per_class] for j in range(kk)], 1) for c in (0, 1)], 0)).save(
    os.path.join(OUT, f"grid_{a.name}.png"))
save_json({"name": a.name, "ckpts": paths, "n_per_class": a.n_per_class, "gs": a.gs, "eps_scale": a.eps_scale, "dt": a.dt, "noise_seed": a.noise_seed, "steps": a.steps,
           "minutes": (time.time() - t0) / 60}, f"gen_{a.name}.json")
log(TAG, f"saved gen_{a.name}.npz ({len(G)} images) in {(time.time() - t0) / 60:.1f} min")
