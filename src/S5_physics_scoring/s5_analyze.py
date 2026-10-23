# s5_analyze.py - S5 analysis (validation / train-fit only; the TEST SET is never used). Everything below is fixed BEFORE
# any S5 result is seen (2026-10-21, ~09:55 Asia/Dhaka):
# PER-IMAGE SCORES - direction fixed in advance: LOWER = BETTER for every score.
#   F1 physics (score-FPE): H = mean over t in {10, 50} of r1 r2 / S_ref(t)^2 (s5_score.py)
#   F2 physics (film/digitiser noise): Poisson-Gaussian noise-level-function residual of the image vs the NLF fitted on the
#      real train-fit images (pinn_physics.PhysicsNLF, per-image min-max to [0,1] for every image)
#   F3 non-physics (model confidence): mean over t in {50, 200, 500} of eps-MSE(t) / median over REAL validation images
#   F4 trivial: Euclidean distance of (breast brightness, saturation share, high-frequency energy, breast area), each
#      z-scored with the real validation images of the same class, to the real class mean
#   F5 REFERENCE ONLY (never a pass criterion: it uses the same feature space as KID): mean distance to the 5 nearest
#      train-fit images in mammography-classifier feature space
# TESTS
#   T1 reliability of F1: Spearman(H first scoring, H independent re-scoring) on the first 80 pool images; F1 is declared
#      UNRELIABLE if rho < 0.5 (its filter result is then reported but cannot pass)
#   T2 filter (each of F1..F4): keep the lowest-score 50 % of the 600-image pool within each class (150 / class);
#      KID-mammo (unbiased MMD^2, cubic kernel, ALL train-fit images as reference) of the kept set vs 1000 random
#      class-balanced 50 % subsets. PASS if KID_kept <= 0.90 x median(random) AND KID_kept < 2.5th percentile(random).
#   T3 realism detector (each of F1..F4): AUC = P(score of a generated image > score of a real validation image), 1000
#      bootstrap CI. PASS if AUC >= 0.65 and the CI lower bound > 0.5 (direction fixed: generated = less consistent).
#   E2 per-t version of T3 for F1 (where in t generated images are least FP-consistent)
#   E3 descriptive: KID-mammo of checkpoints 35 / 40 / 41 / 42 / 44 and of the uniform weight average of 40/41/42/44
#      (same noise bank, 100 / class, paired bootstrap): how much of the apparent quality variability is attributable to
#      checkpoint selection / weight averaging. No pass rule (not assumed to improve anything).
#   E4 Epsilon Scaling at sampling (s = 1.005 / 1.01 / 1.02, epoch 40) vs s = 1: "helps" if KID <= 0.90 x and the paired
#      bootstrap CI of the difference lies below 0.
import os, sys, glob, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, find, OUT
from a6_classifier import build, prep
from pinn_physics import PhysicsNLF
from s5_stats import analyze

ap = argparse.ArgumentParser()
ap.add_argument("--n_random", type=int, default=1000)
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
TAG = "S5-analyze"
dev = torch.device("cuda")
xtr, ytr, pat, _ = load_clean_meta("train"); vm = val_mask(pat)
XV, YV, XF = xtr[vm], ytr[vm], xtr[~vm]
J = lambda f: dict(np.load(os.path.join(OUT, f))) if os.path.exists(os.path.join(OUT, f)) else None

mam = build(); mam.load_state_dict({k: v.float() for k, v in torch.load(find("a6_clf_s0.pt")[0], map_location="cpu").items()})
mam = mam.to(dev).eval()
@torch.no_grad()
def feats(X_u8, bs=64):
    out = []
    for i in range(0, len(X_u8), bs):
        xb = torch.from_numpy(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            h = mam.classifier[1](mam.classifier[0](mam.avgpool(mam.features(prep(xb)))))
        out.append(h.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)

def istats_img(X_u8):
    """per-image version of the a9 istats: breast brightness, saturation share, HF energy, breast area"""
    I = torch.from_numpy(X_u8).float().div(255).unsqueeze(1)
    mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True); I = (I - mn) / (mx - mn + 1e-8)
    mu = F.avg_pool2d(I, 7, 1, 3, count_include_pad=False); m = (mu > 0.04).float(); n = m.sum((1, 2, 3)).clamp_min(1)
    hp = I - F.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    return np.stack([((I * m).sum((1, 2, 3)) / n).numpy(), ((((I > 0.95).float() * m).sum((1, 2, 3))) / n).numpy(),
                     ((hp * hp * m).sum((1, 2, 3)) / n).numpy(), m.mean((1, 2, 3)).numpy()], 1)

def mm_u8(X):
    Xf = torch.from_numpy(X).float(); mn = Xf.amin(dim=(1, 2), keepdim=True); mx = Xf.amax(dim=(1, 2), keepdim=True)
    return ((Xf - mn) / (mx - mn + 1e-8) * 255).round().clamp(0, 255).to(torch.uint8)

D = {"feat_ref": feats(XF), "real_y": YV, "istats_real": istats_img(XV)}
log(TAG, f"reference features: {D['feat_ref'].shape}")
pool, sp, sr_, st = J("gen_pool_ep040.npz"), J("score_pool.npz"), J("score_real.npz"), J("score_retest.npz")
if pool is not None:
    D.update({"pool_y": pool["y"], "feat_pool": feats(pool["x"]), "istats_pool": istats_img(pool["x"])})
    nlf = PhysicsNLF(dev); fitrep = nlf.fit(mm_u8(XF))
    log(TAG, f"noise-level function fitted on train-fit: b {fitrep['b']:.3e} a {fitrep['a']:.3e} c {fitrep['c']:.3e} | R2 {fitrep['R2_var']:.3f}")
    def nlf_res(X):
        U = mm_u8(X); out = []
        with torch.no_grad():
            for i in range(0, len(U), 32):
                out.append(nlf.residual(U[i:i + 32].to(dev).float().div(255).unsqueeze(1)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    D["nlf_pool"], D["nlf_real"] = nlf_res(pool["x"]), nlf_res(XV)
for nm, s_ in (("pool", sp), ("real", sr_), ("retest", st)):
    if s_ is not None:
        D[f"score_{nm}_H"], D[f"score_{nm}_P"], D[f"score_{nm}_MSE"] = s_["H"], s_["P"], s_["MSE"]
        D["score_t"] = s_["t"]
for f in sorted(glob.glob(os.path.join(OUT, "gen_e[34]_*.npz"))):
    k = os.path.basename(f)[4:-4]; d = np.load(f)
    D["feat_" + k] = feats(d["x"]); D["istats_" + k] = istats_img(d["x"])
np.savez(os.path.join(OUT, "S5_data.npz"), **D)                 # everything needed to re-run s5_stats.py on a CPU
log(TAG, f"S5_data.npz written ({len(D)} arrays)")
L, rep = analyze(D, a.n_random, a.n_boot)
txt = "\n".join(L)
open(os.path.join(OUT, "S5_report.txt"), "w").write(txt)
save_json(rep, "S5_results.json")
print(txt)
