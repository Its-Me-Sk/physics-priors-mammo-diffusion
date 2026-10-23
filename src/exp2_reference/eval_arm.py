# eval_arm.py - sample one arm with DDIM + classifier-free guidance and score it.
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
