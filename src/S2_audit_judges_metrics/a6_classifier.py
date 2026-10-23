# a6_classifier.py - A6: real-data benign/malignant classifier on the CLEAN split (+ shared training helpers).
# Used as (i) the "real-only" reference of the later downstream test, (ii) the judge of class conditioning (A7/A9),
# (iii) the "mammography feature" extractor for KID-mammo.
# PRE-STATED RECIPE (no tuning, no epoch selection): torchvision ConvNeXt-Tiny with ImageNet-1k weights,
# 256 px grayscale -> 3 channels, per-image min-max, ImageNet normalisation, AdamW lr 1e-4 wd 0.05, batch 32,
# 15 epochs, cosine schedule with 1 warm-up epoch, horizontal flip + small affine, class-balanced cross-entropy,
# fp16 autocast. FINAL epoch is used. Seeds 0, 1, 2. Trained on clean TRAIN minus the patient-level VALIDATION split.
# PRE-STATED RULE (validation only - the test set is never used for a decision):
#   usable as judge if the mean VALIDATION AUC over the 3 seeds is >= 0.65.  Test AUC is reported, not used.
import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache, load_clean_meta, val_mask, OUT

TAG = "A6"
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(x01):
    """(B,1,H,W) any range -> per-image min-max -> 3ch ImageNet-normalised"""
    mn = x01.amin(dim=(1, 2, 3), keepdim=True); mx = x01.amax(dim=(1, 2, 3), keepdim=True)
    x = (x01 - mn) / (mx - mn + 1e-8)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD


def build():
    import torchvision
    m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


def auc(scores, labels):
    """Mann-Whitney AUC with tie handling (no sklearn dependency)"""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def predict(m, X_u8, bs=64):
    m.eval(); out = []
    for i in range(0, len(X_u8), bs):
        xb = torch.as_tensor(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(m(prep(xb)).float(), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def augment(x, g):
    B = x.shape[0]
    u = lambda: torch.rand(B, generator=g, device=dev) * 2 - 1
    sx, sy, shx, shy = 1 + 0.05 * u(), 1 + 0.05 * u(), 0.05 * u(), 0.05 * u()
    z = torch.zeros(B, device=dev)
    th = torch.stack([torch.stack([sx, shx, z], 1), torch.stack([shy, sy, z], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), padding_mode="zeros", align_corners=False)
    flip = torch.rand(B, generator=g, device=dev) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


def train_clf(X_u8, y, seed, epochs=15, bs=32, tag=TAG):
    """the fixed recipe above; returns the model after the FINAL epoch"""
    t0 = time.time()
    Xg, Yg = torch.from_numpy(X_u8).to(dev), torch.from_numpy(np.asarray(y)).long().to(dev)
    w = torch.tensor([len(y) / (2 * max(1, (np.asarray(y) == c).sum())) for c in (0, 1)], device=dev, dtype=torch.float32)
    torch.manual_seed(seed)
    m = build().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    spe = math.ceil(len(y) / bs); total = spe * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed * 100 + ep)).to(dev)
        tl = 0.0
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(Xg[idx].float().div(255).unsqueeze(1), g)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(m(prep(xb)).float(), Yg[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(tag, f"seed {seed} epoch {ep + 1}/{epochs} train loss {tl / spe:.4f} | {(time.time() - t0) / 60:.1f} min")
    del Xg, Yg
    return m


def boot_auc(p, y, n=1000, seed=0):
    r = np.random.default_rng(seed); vals = []
    i0, i1 = np.where(np.asarray(y) == 0)[0], np.where(np.asarray(y) == 1)[0]
    for _ in range(n):
        b = np.concatenate([r.choice(i0, len(i0)), r.choice(i1, len(i1))])
        vals.append(auc(p[b], np.asarray(y)[b]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    x, y, pat, _ = load_clean_meta("train"); xte, yte = load_cache("test")
    vm = val_mask(pat)
    res = {"recipe": "convnext_tiny IMAGENET1K_V1, AdamW 1e-4 wd .05, bs 32, 15 ep cosine (1 warm-up), flip+affine, "
                     "balanced CE, final epoch; trained on train minus validation patients",
           "n_train_fit": int((~vm).sum()), "n_val": int(vm.sum()), "n_val_patients": len(set(pat[vm])),
           "n_test": int(len(yte)), "seeds": {}}
    log(TAG, f"fit on {res['n_train_fit']} images | validation {res['n_val']} images ({res['n_val_patients']} patients)")
    pv_sum = np.zeros(int(vm.sum()))
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        m = train_clf(x[~vm], y[~vm], seed, a.epochs)
        pv, pt = predict(m, x[vm]), predict(m, xte)
        pv_sum += pv
        res["seeds"][str(seed)] = {"val_auc": auc(pv, y[vm]), "test_auc_reported_only": auc(pt, yte),
                                   "val_balanced_acc": float(np.mean([((pv >= 0.5) == (y[vm] == 1))[y[vm] == c].mean() for c in (0, 1)])),
                                   "minutes": (time.time() - t0) / 60}
        torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(OUT, f"a6_clf_s{seed}.pt"))
        log(TAG, f"seed {seed}: VALIDATION AUC {res['seeds'][str(seed)]['val_auc']:.3f} | test AUC (reported only) "
                 f"{res['seeds'][str(seed)]['test_auc_reported_only']:.3f} | {(time.time() - t0) / 60:.1f} min")
        del m; torch.cuda.empty_cache()
    va = [v["val_auc"] for v in res["seeds"].values()]
    res["mean_val_auc"] = float(np.mean(va)); res["sd_val_auc"] = float(np.std(va, ddof=1)) if len(va) > 1 else 0.0
    pe = pv_sum / len(va)
    res["ensemble_val_auc"] = auc(pe, y[vm]); res["ensemble_val_auc_ci95"] = boot_auc(pe, y[vm])
    res["mean_test_auc_reported_only"] = float(np.mean([v["test_auc_reported_only"] for v in res["seeds"].values()]))
    res["usable_as_judge"] = bool(res["mean_val_auc"] >= 0.65)
    save_json(res, "A6_classifier.json")
    log(TAG, f"mean VALIDATION AUC {res['mean_val_auc']:.3f} +- {res['sd_val_auc']:.3f} (ensemble {res['ensemble_val_auc']:.3f}) "
             f"-> usable as judge: {res['usable_as_judge']}")
