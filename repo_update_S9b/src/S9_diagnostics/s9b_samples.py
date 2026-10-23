# s9b_samples.py - sample-image figures for Paper 1 (CPU only, no training, no new generation).
# Uses ONLY images that already exist: the S8 held-out TEST generated sets and the real test caches.
# No cherry-picking: the first K images of each class are shown (generated) and K images per class drawn with a fixed
# seed (real). Because all arms share the same test noise bank, column j of every generated row starts from the SAME
# initial noise, so differences between rows are caused by the model, not by the noise.
# usage: python s9b_samples.py <S8_dir> <cmmd_data_dir> <clean_data_dir> <out_dir> [K]
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S8, DC, DB, OUT = sys.argv[1:5]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 3
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})

PANELS = {
    "CMMD": {"real": os.path.join(DC, "test_cache.npz"),
             "rows": [("tc40", "Baseline (epoch 40)"), ("tcm0", "λ = 0 control (C1)"), ("tcfp", "FP residual (C1)"),
                      ("tc40_nl", "Acquisition-noise layer (C4)")]},
    "CBIS-DDSM": {"real": os.path.join(DB, "test_cache.npz"),
                  "rows": [("tbm0", "λ = 0 control, epoch 92 (C2)"), ("tbfp", "FP residual, epoch 92 (C2)"),
                           ("tsm0", "λ = 0 control (C5)"), ("tsnlf", "NLF loss (C5)")]},
}
LAB = {0: "benign", 1: "malignant"}
info = {"K_per_class": K, "rule": "generated: first K images of each class (shared test noise bank); real: K per class, rng seed 0"}

def load(path):
    d = np.load(path)
    return d["x"], d["y"].astype(int)

def pick_gen(y):
    return np.concatenate([np.where(y == c)[0][:K] for c in (0, 1)])

def pick_real(y):
    rng = np.random.default_rng(0)
    return np.concatenate([np.sort(rng.choice(np.where(y == c)[0], K, replace=False)) for c in (0, 1)])

def show(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

for ds, P in PANELS.items():
    xr, yr = load(P["real"]); ir = pick_real(yr)
    rows = [("Real (test set)", xr[ir], yr[ir])]
    used = {"real": {"file": os.path.basename(P["real"]), "indices": ir.tolist()}}
    for name, title in P["rows"]:
        f = os.path.join(S8, f"gen_{name}.npz")
        if not os.path.exists(f):
            print(f"!! {f} missing - row skipped"); continue
        xg, yg = load(f); ig = pick_gen(yg)
        rows.append((title, xg[ig], yg[ig])); used[name] = {"file": os.path.basename(f), "indices": ig.tolist(), "n_total": int(len(yg))}
    info[ds] = used
    n = 2 * K
    # ---------- full images ----------
    fig, axs = plt.subplots(len(rows), n, figsize=(1.45 * n + 1.9, 1.5 * len(rows)), squeeze=False)
    for r, (title, X, Y) in enumerate(rows):
        for c in range(n):
            show(axs[r, c], X[c])
            if r == 0:
                axs[r, c].set_title(LAB[int(Y[c])], fontsize=8)
        axs[r, 0].set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    tag = "cmmd" if ds == "CMMD" else "cbis"
    fig.savefig(os.path.join(OUT, f"fig_samples_{tag}.png")); plt.close(fig)
    # ---------- 64 x 64 centre-of-breast crops (texture), same images ----------
    fig, axs = plt.subplots(len(rows), n, figsize=(1.45 * n + 1.9, 1.5 * len(rows)), squeeze=False)
    for r, (title, X, Y) in enumerate(rows):
        for c in range(n):
            img = X[c].astype(float)
            m = img > 0.04 * 255                                   # breast mask, same threshold as the saturation metric
            ys, xs = np.nonzero(m) if m.any() else (np.array([128]), np.array([128]))
            cy, cx = int(np.median(ys)), int(np.median(xs))
            cy, cx = min(max(cy, 32), 224), min(max(cx, 32), 224)
            show(axs[r, c], X[c][cy - 32:cy + 32, cx - 32:cx + 32])
            if r == 0:
                axs[r, c].set_title(LAB[int(Y[c])], fontsize=8)
        axs[r, 0].set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(os.path.join(OUT, f"fig_samples_{tag}_crops.png")); plt.close(fig)
    print(f"{ds}: {len(rows)} rows x {n} images -> fig_samples_{tag}.png, fig_samples_{tag}_crops.png")

json.dump(info, open(os.path.join(OUT, "samples_used.json"), "w"), indent=1)
print("samples_used.json written (exact files and indices shown)")
