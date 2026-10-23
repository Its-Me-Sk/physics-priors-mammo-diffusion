# s5_stats.py - S5 statistics (pure numpy; called by s5_analyze.py, re-runnable on CPU from S5_data.npz) (validation / train-fit only; the TEST SET is never used). Everything below is fixed BEFORE
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
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

kern = lambda A_, B_: (A_ @ B_.T / A_.shape[1] + 1.0) ** 3

def auc(pos, neg):
    """Mann-Whitney AUC = P(pos > neg) + 0.5 P(tie), average ranks for ties"""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def auc_ci(pos, neg, n=1000, seed=0):
    rg = np.random.default_rng(seed)
    v = [auc(pos[rg.integers(0, len(pos), len(pos))], neg[rg.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def analyze(D, n_random=1000, n_boot=1000):
    """D: dict of numpy arrays (see s5_analyze.py). Returns (report lines, results dict)."""
    L = []; P_ = lambda *m: L.append(" ".join(str(x) for x in m)); rep = {}
    FR = D["feat_ref"]
    KRR = kern(FR, FR); kRR = (KRR.sum() - np.trace(KRR)) / (len(FR) * (len(FR) - 1)); del KRR
    def parts(G):
        return kern(G, G), kern(G, FR).mean(1)
    def mmd(KGG, h, c):
        num = c @ KGG @ c - (c * c * np.diag(KGG)).sum(); den = c.sum() ** 2 - (c * c).sum()
        return float(num / den - 2 * (c * h).sum() / c.sum() + kRR)
    P_("=" * 110); P_("S5 REPORT - physics-consistency scoring of synthetic mammograms + failure investigation (validation / train-fit only)"); P_("=" * 110)
    if "pool_y" in D and "score_pool_H" in D and "score_real_H" in D:
        yp, yv = D["pool_y"], D["real_y"]; Gp = D["feat_pool"]; ip, iv = D["istats_pool"], D["istats_real"]
        S, SR = {}, {}
        S["F1_fp"], SR["F1_fp"] = D["score_pool_H"], D["score_real_H"]
        S["F2_noise_physics"], SR["F2_noise_physics"] = D["nlf_pool"], D["nlf_real"]
        med = np.median(D["score_real_MSE"], 0)
        S["F3_denoising_error"], SR["F3_denoising_error"] = (D["score_pool_MSE"] / med).mean(1), (D["score_real_MSE"] / med).mean(1)
        def f4(st_, y):
            out = np.zeros(len(y))
            for c in (0, 1):
                mu_, sd_ = iv[yv == c].mean(0), iv[yv == c].std(0) + 1e-9
                out[y == c] = np.sqrt((((st_[y == c] - mu_) / sd_) ** 2).sum(1))
            return out
        S["F4_trivial_intensity"], SR["F4_trivial_intensity"] = f4(ip, yp), f4(iv, yv)
        S["F5_reference_knn"] = cKDTree(FR).query(Gp, k=min(5, len(FR)))[0].reshape(len(Gp), -1).mean(1)
        # ---- T1 ----
        P_("\n[T1] RELIABILITY OF THE FP SCORE (rule: Spearman >= 0.5 between two independent scorings)")
        if "score_retest_H" in D:
            k = len(D["score_retest_H"]); rel = spearman(D["score_pool_H"][:k], D["score_retest_H"])
            per_t = [spearman(D["score_pool_P"][:k, j], D["score_retest_P"][:, j]) for j in range(D["score_retest_P"].shape[1])]
            rep["T1"] = {"n": k, "spearman_H": rel, "spearman_per_t": per_t, "reliable": bool(rel >= 0.5)}
            P_(f"  {k} images: Spearman H {rel:.3f} | per t " + " ".join(f"{int(t)}:{r:.3f}" for t, r in zip(D["score_t"], per_t))
               + f" -> F1 {'RELIABLE' if rel >= 0.5 else 'UNRELIABLE'}")
        else:
            rep["T1"] = {"reliable": False}; P_("  retest missing -> F1 treated as UNRELIABLE")
        f1_ok = bool(rep["T1"]["reliable"])
        # ---- T2 ----
        KGG, hh = parts(Gp)
        rng = np.random.default_rng(0)
        idx_c = {c: np.where(yp == c)[0] for c in (0, 1)}
        keep_n = {c: len(idx_c[c]) // 2 for c in (0, 1)}
        rand = []
        for _ in range(n_random):
            cvec = np.zeros(len(yp))
            for c in (0, 1):
                cvec[rng.choice(idx_c[c], keep_n[c], replace=False)] = 1
            rand.append(mmd(KGG, hh, cvec))
        rand = np.array(rand); med_r, q025 = float(np.median(rand)), float(np.quantile(rand, 0.025))
        full = mmd(KGG, hh, np.ones(len(yp)))
        P_(f"\n[T2] FILTERS: keep the lowest-score 50 % per class ({keep_n[0]}+{keep_n[1]} of {len(yp)}); KID-mammo vs train-fit")
        P_(f"  whole pool {full:.4f} | random 50 % subsets: median {med_r:.4f}, 2.5th pct {q025:.4f}, 97.5th pct {np.quantile(rand, .975):.4f}")
        rep["T2"] = {"pool_kid": full, "random_median": med_r, "random_q025": q025, "filters": {}}
        for f, sc in S.items():
            cvec = np.zeros(len(yp))
            for c in (0, 1):
                ii = idx_c[c]; cvec[ii[np.argsort(sc[ii], kind="mergesort")[:keep_n[c]]]] = 1
            kk = mmd(KGG, hh, cvec); pct = float((rand < kk).mean() * 100); kept = cvec.astype(bool)
            ref_only = f.startswith("F5")
            passed = (not ref_only) and kk <= 0.90 * med_r and kk < q025 and (f1_ok or not f.startswith("F1"))
            rep["T2"]["filters"][f] = {"kid_kept": kk, "ratio_to_random_median": kk / med_r, "percentile_among_random": pct,
                                       "saturation_kept": float(ip[kept, 1].mean()), "hf_energy_kept": float(ip[kept, 2].mean()),
                                       "passed": bool(passed), "reference_only": ref_only}
            P_(f"  {f:22s}: KID {kk:.4f} = x{kk / med_r:.3f} of random median | percentile among random {pct:5.1f} | kept saturation "
               f"{ip[kept, 1].mean():.4f} HF {ip[kept, 2].mean():.4f} -> " + ("REFERENCE ONLY" if ref_only else ("PASS" if passed else "fail"))
               + ("" if (f1_ok or not f.startswith("F1")) else " (F1 unreliable)"))
        P_(f"  all pool: saturation {ip[:, 1].mean():.4f} HF {ip[:, 2].mean():.4f} | real validation: saturation {iv[:, 1].mean():.4f} HF {iv[:, 2].mean():.4f}")
        # ---- T3 / E2 ----
        P_("\n[T3] REAL vs GENERATED (AUC = P(generated scores worse than real); rule AUC >= 0.65 with CI lower > 0.5)")
        P_("  caveat: the real validation images were part of the diffusion model's training set (it was trained on all clean train")
        P_("  images), so a difference can also mean 'seen in training' - the filter test T2 does not have this problem")
        rep["T3"] = {}
        for f in ("F1_fp", "F2_noise_physics", "F3_denoising_error", "F4_trivial_intensity"):
            A0 = auc(S[f], SR[f]); ci = auc_ci(S[f], SR[f], n_boot)
            passed = A0 >= 0.65 and ci[0] > 0.5 and (f1_ok or not f.startswith("F1"))
            rep["T3"][f] = {"auc": A0, "ci95": ci, "passed": bool(passed), "median_generated": float(np.median(S[f])),
                            "median_real": float(np.median(SR[f]))}
            P_(f"  {f:22s}: AUC {A0:.3f} CI {ci[0]:.3f}-{ci[1]:.3f} | median generated {np.median(S[f]):.4g} vs real {np.median(SR[f]):.4g} -> "
               + ("PASS" if passed else "fail"))
        P_("\n[E2] FP CONSISTENCY PER t (generated vs real)")
        rep["E2"] = {}
        for j, t in enumerate(D["score_t"]):
            pg, pr = D["score_pool_P"][:, j], D["score_real_P"][:, j]
            A0 = auc(pg, pr); ci = auc_ci(pg, pr, n_boot)
            rep["E2"][str(int(t))] = {"auc": A0, "ci95": ci, "mean_generated": float(pg.mean()), "mean_real": float(pr.mean())}
            P_(f"  t={int(t):4d}: mean P generated {pg.mean():.4f} vs real {pr.mean():.4f} | AUC {A0:.3f} CI {ci[0]:.3f}-{ci[1]:.3f}")
        P_("\n[CORR] Spearman correlation between the scores on the generated pool")
        names = list(S); M = np.array([[spearman(S[p], S[q]) for q in names] for p in names])
        rep["score_correlations"] = {"names": names, "matrix": M.tolist()}
        P_("         " + " ".join(f"{n[:6]:>8s}" for n in names))
        for p, row in zip(names, M):
            P_(f"  {p[:6]:6s} " + " ".join(f"{v:+8.2f}" for v in row))
    else:
        P_("\n!! pool or scores missing -> filter tests skipped")
    # ---- E3 / E4 ----
    sets = sorted(k[5:] for k in D if k.startswith("feat_e3_") or k.startswith("feat_e4_"))
    if sets:
        n = min(len(D["feat_" + k]) for k in sets)
        FE = {k: parts(D["feat_" + k][:n]) for k in sets}
        C = np.random.default_rng(1).multinomial(n, np.ones(n) / n, size=n_boot).astype(float)
        ones = np.ones(n)
        kid = {k: mmd(*FE[k], ones) for k in FE}
        boot = {k: np.array([mmd(*FE[k], c) for c in C]) for k in FE}
        e3 = sorted(k for k in kid if k.startswith("e3_"))
        P_(f"\n[E3] CHECKPOINT SELECTION / WEIGHT AVERAGING (KID-mammo, {n} images, same noise; bootstrap 95 % CI; descriptive)")
        rep["E3"] = {}
        for k in e3:
            ci = [float(np.quantile(boot[k], .025)), float(np.quantile(boot[k], .975))]
            rep["E3"][k] = {"kid": kid[k], "ci95": ci}
            P_(f"  {k:12s}: KID {kid[k]:.4f} (CI {ci[0]:.4f}-{ci[1]:.4f})")
        singles = [k for k in e3 if k != "e3_avg"]
        if len(singles) > 1:
            vals = np.array([kid[k] for k in singles]); within = float(np.mean([boot[k].std() for k in singles]))
            rep["E3"]["summary"] = {"between_checkpoint_sd": float(vals.std(ddof=1)), "mean_within_bootstrap_sd": within,
                                    "range": [float(vals.min()), float(vals.max())]}
            P_(f"  single checkpoints: KID range {vals.min():.4f}-{vals.max():.4f} | between-checkpoint SD {vals.std(ddof=1):.4f} vs "
               f"sampling (bootstrap) SD {within:.4f} -> ratio {vals.std(ddof=1) / (within + 1e-12):.2f}")
            if "e3_avg" in kid:
                med_k = sorted(singles, key=lambda k: kid[k])[len(singles) // 2]
                d = boot["e3_avg"] - boot[med_k]
                rep["E3"]["avg_vs_median_checkpoint"] = {"median_checkpoint": med_k,
                                                         "diff_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
                P_(f"  weight average vs the median single checkpoint ({med_k}): diff CI {np.quantile(d, .025):+.4f}..{np.quantile(d, .975):+.4f}")
        if "e3_ep040" in kid:
            P_("\n[E4] EPSILON SCALING at sampling (epoch 40) vs s = 1 (rule: KID <= 0.90x with paired CI < 0)")
            rep["E4"] = {}
            for k in sorted(x for x in kid if x.startswith("e4_")):
                d = boot[k] - boot["e3_ep040"]; ci = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
                ok = kid[k] <= 0.90 * kid["e3_ep040"] and ci[1] < 0
                rep["E4"][k] = {"kid": kid[k], "ratio": kid[k] / kid["e3_ep040"], "diff_ci95": ci, "helps": bool(ok)}
                P_(f"  {k:12s}: KID {kid[k]:.4f} = x{kid[k] / kid['e3_ep040']:.3f} (diff CI {ci[0]:+.4f}..{ci[1]:+.4f}) -> {'HELPS' if ok else 'no'}")
    P_("\n" + "=" * 110)
    P_("Send S5_report.txt (or S5_results.zip) and the grid_*.png images back.")
    return L, rep
