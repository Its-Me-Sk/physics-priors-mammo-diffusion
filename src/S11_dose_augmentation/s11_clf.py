# s11_clf.py - S11: physics-based dose augmentation for a benign/malignant classifier (rules: S11_preregistration.txt).
# One call = one training dataset, a list of arms and seeds. Trains the fixed S2 classifier recipe with
#   R0 standard augmentation | R1 + white noise of matched power | R2 + physics dose-reduction noise,
# then saves predicted probabilities on every test condition (clean, dose50, dose25 on CMMD; clean on CBIS).
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a6_classifier import build, prep, augment, predict, auc, dev
from pinn_physics import PhysicsNLF

ap = argparse.ArgumentParser()
ap.add_argument("--train_ds", choices=["cmmd", "cbis"], required=True)
ap.add_argument("--arms", default="R0,R1,R2")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--data_cmmd", required=True)      # folder with train_cache.npz / test_cache.npz (with patient ids)
ap.add_argument("--data_cbis", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
TAG = f"S11-{a.train_ds}"
def log(*m): print(f"[{TAG}]", *m, flush=True)

def val_mask(patients, frac=0.15, seed=0):            # identical to phaseA_common.val_mask
    ups = sorted(set(patients)); rng = np.random.default_rng(seed)
    val = set(rng.choice(ups, int(round(frac * len(ups))), replace=False).tolist())
    return np.array([p in val for p in patients])
def load(folder, split):
    d = np.load(os.path.join(folder, f"{split}_cache.npz")); return d["x"], d["y"].astype(np.int64), d["patient"].astype(str)

xtr, ytr, ptr = load(a.data_cmmd if a.train_ds == "cmmd" else a.data_cbis, "train")
vm = val_mask(ptr); XF, YF, XV, YV = xtr[~vm], ytr[~vm], xtr[vm], ytr[vm]
xc, yc, _ = load(a.data_cmmd, "test"); xb_, yb_, _ = load(a.data_cbis, "test")
if a.smoke:
    XF, YF, a.epochs = XF[:64], YF[:64], 1
    xc, yc, xb_, yb_ = xc[:40], yc[:40], xb_[:40], yb_[:40]

def mm01(x):                                          # (B,1,H,W) -> per-image min-max [0,1]
    mn = x.amin(dim=(1, 2, 3), keepdim=True); mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-8)

# ---------------- noise model: linear Poisson-Gaussian NLF of THIS training set (PhysicsNLF, as in S7/S8) ----------
def fit_nlf(X_u8):
    X = torch.from_numpy(X_u8).float(); mn = X.amin(dim=(1, 2), keepdim=True); mx = X.amax(dim=(1, 2), keepdim=True)
    U = ((X - mn) / (mx - mn + 1e-8) * 255).round().clamp(0, 255).to(torch.uint8)
    rep = PhysicsNLF(dev).fit(U); lin = rep["linear_poisson_gaussian_only"]
    return float(lin["a"]), float(lin["b"]), float(lin["R2_var"])
A_, B_, R2_ = fit_nlf(XF)
g1 = torch.exp(-(torch.arange(7.0) - 3) ** 2 / 2); g1 = g1 / g1.sum(); G = torch.outer(g1, g1)
delta = torch.zeros(7, 7); delta[3, 3] = 1
K_HP = float(((delta - G) ** 2).sum())               # E[highpass^2] = K_HP * white-noise variance
log(f"train {len(YF)} (benign {int((YF == 0).sum())}) | val {len(YV)} | NLF sigma^2 = {A_:.3e} I + {B_:.3e} (R2 {R2_:.2f}) | k_hp {K_HP:.4f}")

def phys_var(I, alpha):
    al = alpha.view(-1, 1, 1, 1)
    return ((A_ * I * (1 / al - 1) + B_ * (1 / al ** 2 - 1)) / K_HP).clamp_min(0)
def degrade(I, alpha, g):
    return (I + phys_var(I, alpha).sqrt() * torch.randn(I.shape, generator=g, device=I.device)).clamp(0, 1)

# matched power for R1: mean added R2 variance over 200 training images (Monte Carlo, fixed seed)
with torch.no_grad():
    gm = torch.Generator(device=dev).manual_seed(99)
    I = mm01(torch.from_numpy(XF[:200]).to(dev).float().unsqueeze(1))
    al = 0.25 + 0.75 * torch.rand(I.shape[0], generator=gm, device=dev)
    V_GEN = float(phys_var(I, al).mean())
log(f"R1 generic white-noise variance (matched to R2 mean) = {V_GEN:.3e} (std {math.sqrt(V_GEN) * 255:.1f} grey levels)")

def noise_aug(x01, arm, g):
    if arm == "R0":
        return x01
    B = x01.shape[0]; I = mm01(x01)
    use = (torch.rand(B, generator=g, device=dev) < 0.8).view(-1, 1, 1, 1)
    if arm == "R1":
        out = (I + math.sqrt(V_GEN) * torch.randn(I.shape, generator=g, device=dev)).clamp(0, 1)
    else:
        out = degrade(I, 0.25 + 0.75 * torch.rand(B, generator=g, device=dev), g)
    return torch.where(use, out, x01)

def train(arm, seed, epochs, bs=32):                  # = a6_classifier.train_clf + the noise hook
    Xg, Yg = torch.from_numpy(XF).to(dev), torch.from_numpy(YF).long().to(dev)
    w = torch.tensor([len(YF) / (2 * max(1, (YF == c).sum())) for c in (0, 1)], device=dev, dtype=torch.float32)
    torch.manual_seed(seed); m = build().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    spe = math.ceil(len(YF) / bs); total = spe * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda"); g = torch.Generator(device=dev).manual_seed(seed)
    gn = torch.Generator(device=dev).manual_seed(seed * 7 + 1)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(YF), generator=torch.Generator().manual_seed(seed * 100 + ep)).to(dev); tl = 0.0
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = noise_aug(augment(Xg[idx].float().div(255).unsqueeze(1), g), arm, gn)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(m(prep(xb)).float(), Yg[idx], weight=w)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss.detach())
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"{arm} seed {seed} epoch {ep + 1}/{epochs} loss {tl / spe:.4f}")
    return m

# fixed degraded CMMD test sets (alpha 0.5 and 0.25, seed 4242) - built once, identical for every arm and seed
with torch.no_grad():
    gt = torch.Generator(device=dev).manual_seed(4242); TESTS = {"cmmd_clean": xc, "cbis_clean": xb_}
    for al in (0.5, 0.25):
        D = []
        for i in range(0, len(xc), 64):
            I = mm01(torch.from_numpy(xc[i:i + 64]).to(dev).float().unsqueeze(1))
            D.append((degrade(I, torch.full((I.shape[0],), al, device=dev), gt) * 255).round().byte().squeeze(1).cpu().numpy())
        TESTS[f"cmmd_dose{int(al * 100)}"] = np.concatenate(D)
LAB = {"cmmd_clean": yc, "cbis_clean": yb_, "cmmd_dose50": yc, "cmmd_dose25": yc}
json.dump({"train_ds": a.train_ds, "nlf_a": A_, "nlf_b": B_, "nlf_R2": R2_, "k_hp": K_HP, "v_generic": V_GEN,
           "n_train": int(len(YF)), "n_val": int(len(YV))}, open(os.path.join(a.out, f"s11_{a.train_ds}_meta.json"), "w"), indent=1)
for arm in a.arms.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        f = os.path.join(a.out, f"pred_{a.train_ds}_{arm}_s{seed}.npz")
        if os.path.exists(f):
            log(f"{f} exists -> skipped"); continue
        t0 = time.time(); m = train(arm, seed, a.epochs).eval()
        P = {k: predict(m, v) for k, v in TESTS.items()}; P["val"] = predict(m, XV)
        np.savez(f, **P, **{f"y_{k}": v for k, v in LAB.items()}, y_val=YV)
        log(f"{arm} seed {seed}: val AUC {auc(P['val'], YV):.3f} | " + " | ".join(f"{k} {auc(P[k], LAB[k]):.3f}" for k in TESTS)
            + f" | {(time.time() - t0) / 60:.1f} min")
        del m; torch.cuda.empty_cache()
