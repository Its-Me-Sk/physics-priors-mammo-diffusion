# a7_generate.py - A7: generate samples from a clean baseline (DDIM 50 steps, same sampler as eval_arm.py).
# Spec "gs:n_per_class[s]", e.g. "4:200,1:200,0:50s":
#   guidance gs, n images per class; eps = eu + gs (ec - eu), null label -1 (as in training); gs = 1 -> one conditional pass.
#   suffix "s" = SAME starting noise for both labels + pixel-identity test (guidance 0 must ignore the label:
#   benign and malignant outputs must be identical). Otherwise each class has its own noise (seed + 100000*label + batch);
#   the same noise bank is used for every guidance value and every model (paired).
# Writes gen_<tag>_gs<g>.npz (x uint8, y) after EACH guidance value, a grid png, and the speed. --max_min stops early.
import os, sys, time, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_model, clean_ckpt, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--seed_model", type=int, required=True)
ap.add_argument("--epoch", type=int, default=40)
ap.add_argument("--spec", default="4:200,1:200")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--noise_seed", type=int, default=2025)
ap.add_argument("--max_min", type=float, default=1e9)
ap.add_argument("--tag", default="")
a = ap.parse_args()
TAG = f"A7-gen-s{a.seed_model}{a.tag}"
dev = torch.device("cuda")
path = clean_ckpt(a.seed_model, a.epoch)
unet, ac, meta = load_model(path, dev)
unet = unet.half()
acf = ac.float(); T = acf.numel()
name = f"s{a.seed_model}_ep{a.epoch:03d}"
log(TAG, f"checkpoint {path} | epoch {meta.get('epoch')} | spec {a.spec}")


@torch.no_grad()
def ddim(label, n, gen, gs):
    x = torch.randn(n, 1, 256, 256, generator=gen, device=dev)
    ts = torch.linspace(T - 1, 0, a.steps, device=dev).round().long()
    cond = torch.full((n, 1, 1), float(label), device=dev)
    for i, t in enumerate(ts):
        ab_t = acf[t]
        ab_prev = acf[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        if gs == 1.0:
            tb = torch.full((n,), int(t), device=dev, dtype=torch.long)
            eps = unet(x.half(), timesteps=tb, context=cond.half()).float()
        else:
            tb = torch.full((2 * n,), int(t), device=dev, dtype=torch.long)
            ctx = torch.cat([-torch.ones_like(cond), cond]).half()
            eu, ec = unet(torch.cat([x, x]).half(), timesteps=tb, context=ctx).float().chunk(2)
            eps = eu + gs * (ec - eu)
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return ((x + 1) / 2).clamp(0, 1)


t_start = time.time()
info = {"checkpoint": path, "epoch": meta.get("epoch"), "steps": a.steps,
        "guidance_equation": "eps = eps_uncond + gs * (eps_cond - eps_uncond), null label -1", "runs": {}}
for item in a.spec.split(","):
    g, n = item.split(":"); same = n.endswith("s"); n = int(n.rstrip("s")); gs = float(g)
    if (time.time() - t_start) / 60 > a.max_min:
        log(TAG, f"time limit ({a.max_min} min) reached -> guidance {gs:g} skipped"); continue
    t0 = time.time(); imgs, labels, floats = [], [], {0: [], 1: []}
    for label in (0, 1):
        done, b = 0, 0
        while done < n:
            k = min(a.bs, n - done)
            seed = a.noise_seed + 7777 + b if same else a.noise_seed + 100000 * label + b
            out = ddim(label, k, torch.Generator(device=dev).manual_seed(seed), gs)[:, 0].cpu()
            if same:
                floats[label].append(out)
            imgs.append((out.numpy() * 255).round().astype(np.uint8)); labels += [label] * k; done += k; b += 1
    G = np.concatenate(imgs); L = np.array(labels, dtype=np.int64)
    sec = time.time() - t0
    key = f"gs{gs:g}" + ("_samenoise" if same else "")
    run = {"images": int(len(G)), "per_class": n, "same_noise": same, "minutes": sec / 60, "s_per_image": sec / len(G)}
    if same:
        d = (torch.cat(floats[0]) - torch.cat(floats[1])).abs()
        run["identity_test"] = {"max_abs_diff_float": float(d.max()), "mean_abs_diff_float": float(d.mean()),
                                "max_abs_diff_uint8": int(np.abs(G[L == 0].astype(int) - G[L == 1].astype(int)).max()),
                                "label_ignored": bool(float(d.max()) < 1e-3)}
        log(TAG, f"guidance {gs:g} same-noise identity test: max |benign - malignant| = {run['identity_test']['max_abs_diff_float']:.2e} "
                 f"-> label ignored: {run['identity_test']['label_ignored']}")
    info["runs"][key] = run
    np.savez_compressed(os.path.join(OUT, f"gen_{name}_{key}.npz"), x=G, y=L)
    from PIL import Image
    kk = min(8, n)
    Image.fromarray(np.concatenate([np.concatenate([G[j + c * n] for j in range(kk)], 1) for c in (0, 1)], 0)).save(
        os.path.join(OUT, f"grid_{name}_{key}.png"))
    log(TAG, f"guidance {gs:g}: {len(G)} images in {sec / 60:.1f} min ({sec / len(G):.2f} s/image)")
    save_json(info, f"A7_gen_{name}{a.tag}.json")
