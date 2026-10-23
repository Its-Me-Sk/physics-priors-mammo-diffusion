# s6_stats.py - S6 statistics (pure numpy). Pre-stated rule (fixed 2026-10-21 ~17:10 Asia/Dhaka, before any S6 result):
#   For every arm vs the baseline arm (same noise bank -> generated image i is paired across arms):
#     KID-mammo  = unbiased MMD^2 (cubic kernel) between the generated set and the reference set in the A6 mammography-
#                  classifier feature space; KID-dino = the same in DINOv2 ViT-B/14 feature space (independent "eye")
#     precision / recall (Kynkaenniemi et al. 2019, k = 3) in DINOv2 space: realism / coverage (diversity)
#   PASS if ALL: KID-mammo <= 0.90 x baseline with the paired-bootstrap 95 % CI of the difference < 0;
#               KID-dino <= 1.00 x baseline (no loss in the independent feature space);
#               recall >= baseline recall - 0.05 (no loss of diversity).
#   Validation stage: reference = train-fit images; WINNER = passing arm with the lowest KID-mammo (none -> no winner).
#   Test stage (run ONCE, frozen winner vs baseline, fresh noise seeds): reference = the untouched TEST images; the same
#   rule -> CONFIRMED / NOT CONFIRMED.
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

kern = lambda A_, B_: (A_ @ B_.T / A_.shape[1] + 1.0) ** 3

def _kRR(R):
    K = kern(R, R); return (K.sum() - np.trace(K)) / (len(R) * (len(R) - 1))

def _mmd(KGG, h, kRR, c):
    num = c @ KGG @ c - (c * c * np.diag(KGG)).sum(); den = c.sum() ** 2 - (c * c).sum()
    return float(num / den - 2 * (c * h).sum() / c.sum() + kRR)

def prec_recall(real, fake, k=3):
    """improved precision / recall (Kynkaenniemi et al. 2019): k-NN balls around real (fake) points"""
    rr = cKDTree(real).query(real, k=k + 1)[0][:, -1]; rf = cKDTree(fake).query(fake, k=k + 1)[0][:, -1]
    D = cdist(fake, real)
    return float((D <= rr[None, :]).any(1).mean()), float((D.T <= rf[None, :]).any(1).mean())

def evaluate(F, base, n_boot=1000, seed=0):
    """F: {"ref_mammo", "ref_dino", "<arm>_mammo", "<arm>_dino", "<arm>_istats" (optional)}; returns (lines, results)"""
    arms = sorted({k[:-6] for k in F if k.endswith("_mammo") and not k.startswith("ref")})
    assert base in arms, f"baseline {base} missing"
    n = min(len(F[f"{a}_mammo"]) for a in arms)
    C = np.random.default_rng(seed).multinomial(n, np.ones(n) / n, size=n_boot).astype(float); ones = np.ones(n)
    res = {}
    for space in ("mammo", "dino"):
        R = F[f"ref_{space}"]; kRR = _kRR(R)
        for a in arms:
            G = F[f"{a}_{space}"][:n]; KGG, h = kern(G, G), kern(G, R).mean(1)
            res.setdefault(a, {})[f"kid_{space}"] = _mmd(KGG, h, kRR, ones)
            res[a][f"boot_{space}"] = np.array([_mmd(KGG, h, kRR, c) for c in C])
    for a in arms:
        res[a]["precision"], res[a]["recall"] = prec_recall(F["ref_dino"], F[f"{a}_dino"][:n])
        if f"{a}_istats" in F:
            st = F[f"{a}_istats"][:n]; res[a]["saturation"], res[a]["hf_energy"] = float(st[:, 1].mean()), float(st[:, 2].mean())
    b = res[base]; L = []; out = {}
    L.append(f"  baseline {base}: KID-mammo {b['kid_mammo']:.4f} | KID-dino {b['kid_dino']:.4f} | precision {b['precision']:.3f} "
             f"recall {b['recall']:.3f}" + (f" | saturation {b['saturation']:.4f} HF {b['hf_energy']:.4f}" if "saturation" in b else ""))
    for a in arms:
        r = res[a]
        if a == base:
            out[a] = {k: v for k, v in r.items() if not k.startswith("boot")}; continue
        dm = r["boot_mammo"] - b["boot_mammo"]; dd = r["boot_dino"] - b["boot_dino"]
        ci_m = [float(np.quantile(dm, .025)), float(np.quantile(dm, .975))]; ci_d = [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]
        ok = {"KIDm<=0.90x": bool(r["kid_mammo"] <= 0.90 * b["kid_mammo"]), "KIDm_CI<0": bool(ci_m[1] < 0),
              "KIDd<=1.00x": bool(r["kid_dino"] <= b["kid_dino"]), "recall_ok": bool(r["recall"] >= b["recall"] - 0.05)}
        out[a] = {"kid_mammo": r["kid_mammo"], "kid_mammo_ratio": r["kid_mammo"] / b["kid_mammo"], "kid_mammo_diff_ci95": ci_m,
                  "kid_dino": r["kid_dino"], "kid_dino_ratio": r["kid_dino"] / b["kid_dino"], "kid_dino_diff_ci95": ci_d,
                  "precision": r["precision"], "recall": r["recall"], "saturation": r.get("saturation"), "hf_energy": r.get("hf_energy"),
                  "checks": ok, "passed": all(ok.values())}
        L.append(f"  {a:16s}: KID-mammo x{out[a]['kid_mammo_ratio']:.3f} (diff CI {ci_m[0]:+.4f}..{ci_m[1]:+.4f}) | KID-dino "
                 f"x{out[a]['kid_dino_ratio']:.3f} (CI {ci_d[0]:+.4f}..{ci_d[1]:+.4f}) | precision {r['precision']:.3f} recall {r['recall']:.3f}"
                 + (f" | saturation {r['saturation']:.4f} HF {r['hf_energy']:.4f}" if "saturation" in r else "")
                 + f" -> {'PASS' if out[a]['passed'] else 'fail'} {ok}")
    passed = sorted([(v["kid_mammo_ratio"], a) for a, v in out.items() if v.get("passed")])
    return L, out, (passed[0][1] if passed else None)

def _auc(p, y):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def downstream_summary(D, ref="real_only", n_boot=2000, seed=0):
    """D: {arm: {"y": y, "p0": ..., "p1": ..., ...}} -> lines (secondary, descriptive)"""
    L = []; rng = np.random.default_rng(seed)
    y = D[ref]["y"]; i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    B = [np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]) for _ in range(n_boot)]
    ens = {a: np.mean([v for k, v in d.items() if k.startswith("p")], 0) for a, d in D.items()}
    for a, d in D.items():
        aucs = [_auc(v, y) for k, v in d.items() if k.startswith("p")]
        line = f"  {a:16s}: TEST AUC {np.mean(aucs):.4f} +- {np.std(aucs, ddof=1) if len(aucs) > 1 else 0:.4f} (seeds) | ensemble {_auc(ens[a], y):.4f}"
        if a != ref:
            diff = [_auc(ens[a][b], y[b]) - _auc(ens[ref][b], y[b]) for b in B]
            line += f" | vs {ref}: {_auc(ens[a], y) - _auc(ens[ref], y):+.4f} (CI {np.quantile(diff, .025):+.4f}..{np.quantile(diff, .975):+.4f})"
        L.append(line)
    return L
