# Offline analyses on results already in hand (no training, no new GPU time).
# Addresses reviewer weaknesses: single seed, effect sizes, FP mechanism, metric validity, memorisation, noise physics.
import json, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

SP = "/tmp/claude-0/-home-claude/3f2b8356-4646-5718-a62c-c0e93328cc98/scratchpad"
sys.path.insert(0, "/home/claude/phaseA/src6")
from s6_stats import _kRR, _mmd, kern
OUT = f"{SP}/analysis/out"; os.makedirs(OUT, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6",
     "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surf": "#fcfcfb"}
plt.rcParams.update({"figure.facecolor": C["surf"], "axes.facecolor": C["surf"], "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight"})
REP = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); REP.append(s)

# ---------------------------------------------------------------- data from earlier stages
rows = []
for f in sorted(glob.glob(f"{SP}/s3res/eval_*.json") + glob.glob(f"{SP}/s4res/eval_*.json")):
    d = json.load(open(f))
    if "quality" in d:
        q = d["quality"]
        rows.append({"arm": d["arm"], "ep": d["epoch"], "H": d["H_mean"], "mse": d["val_mse_mean"],
                     "kid": q["kid_mammo_vs_trainfit"][0], "kid_sd": q["kid_mammo_vs_trainfit"][1],
                     "floor": q["kid_mammo_real_val_vs_trainfit_floor"][0]})
E3 = json.load(open(f"{SP}/s5res/S5_results.json"))["E3"]
base40 = 0.4932
# pre-registered comparisons: (label, stage, family, ratio, diff CI lo, hi, reference KID)
CMP = [
 ("FP lambda 0.016", "S3", "FP", 0.7928 / 0.5139, None, None, 0.5139),
 ("FP lambda 0.054", "S3", "FP", 1.1576 / 0.5139, None, None, 0.5139),
 ("FP lambda 0.107", "S3", "FP", 1.3270 / 0.5139, None, None, 0.5139),
 ("Spectral norm r1a", "S4a", "FP-derived", 1.315, 0.1617, 0.2228, 0.6079),
 ("Spectral norm r1b", "S4a", "FP-derived", 1.151, 0.0642, 0.1206, 0.6079),
 ("Spectral norm r2a", "S4a", "FP-derived", 1.321, 0.1641, 0.2249, 0.6079),
 ("Spectral norm r2b", "S4a", "FP-derived", 1.040, -0.0094, 0.0597, 0.6079),
 ("Divergence r3", "S4a", "FP-derived", 1.363, 0.1884, 0.2505, 0.6079),
 ("Weight averaging A", "S6", "inference", 0.914, -0.0673, -0.0169, base40),
 ("Weight averaging B", "S6", "inference", 0.998, -0.0260, 0.0257, base40),
 ("Best-of-2 (F3)", "S6", "inference", 0.911, -0.0776, -0.0114, base40),
 ("Dynamic threshold", "S6", "inference", 1.346, 0.1273, 0.2103, base40),
 ("Avg A + dyn. thr.", "S6", "inference", 1.496, 0.1953, 0.2925, base40),
 ("Avg A + dyn. thr. + bo2", "S6", "inference", 1.277, 0.0890, 0.1843, base40),
 ("Acquisition noise layer", "S7", "acquisition", 1.415, 0.1802, 0.2290, base40),
]
FAMC = {"FP": C["red"], "FP-derived": C["orange"], "inference": C["blue"], "acquisition": C["violet"]}

# ---------------------------------------------------------------- A. effect-size table
say("=" * 100); say("A. EFFECT SIZES of every comparison with rules fixed before its run (KID-mammo, validation, vs its own control)")
say(f"{'comparison':26s} {'stage':5s} {'ratio':>6s} {'change':>8s} {'diff (abs)':>11s} {'95% CI of diff':>22s}  passes 0.90 bar?")
for lab, st, fam, r, lo, hi, ref in CMP:
    ci = f"{lo:+.3f} .. {hi:+.3f}" if lo is not None else "(not computed in S3)"
    say(f"{lab:26s} {st:5s} {r:6.3f} {100 * (r - 1):+7.1f}% {ref * (r - 1):+11.3f} {ci:>22s}  {'yes' if (r <= 0.90 and hi is not None and hi < 0) else 'no'}")

# ---------------------------------------------------------------- B. single-seed: how big is training variability?
ck = {k: v["kid"] for k, v in E3.items() if k.startswith("e3_ep")}
near = [ck[k] for k in ("e3_ep040", "e3_ep041", "e3_ep042", "e3_ep044")]
sd_ck, mu_ck = float(np.std(near, ddof=1)), float(np.mean(near))
boot_sd = E3["summary"]["mean_within_bootstrap_sd"]
say("=" * 100); say("B. SINGLE-SEED CAVEAT: checkpoint-to-checkpoint KID variation WITHIN ONE training trajectory (not a substitute for repeated seeds)")
say(f"  KID at epochs 40/41/42/44 (same run, 1-4 epochs apart): {', '.join(f'{v:.3f}' for v in near)} | mean {mu_ck:.3f} SD {sd_ck:.3f} "
    f"-> relative SD {100 * sd_ck / mu_ck:.1f}% (epoch 35: {ck['e3_ep035']:.3f})")
say(f"  sampling (bootstrap) SD of one KID: {boot_sd:.4f} -> checkpoint SD is {sd_ck / boot_sd:.1f}x the evaluation uncertainty")
say(f"  -> an effect is large relative to this variation only if |ratio - 1| clearly exceeds ~2 x {100 * sd_ck / mu_ck:.0f}% = {200 * sd_ck / mu_ck:.0f}%")
for lab, st, fam, r, lo, hi, ref in CMP:
    z = abs(r - 1) / (sd_ck / mu_ck)
    tag = "large relative to observed checkpoint variation" if z >= 2 else ("borderline" if z >= 1 else "within observed checkpoint variation")
    say(f"    {lab:26s} {100 * (r - 1):+6.1f}%  = {z:4.1f} checkpoint-SDs -> {tag}")
floor = rows[0]["floor"]
say(f"  empirical real-vs-real reference KID under this protocol (validation real vs train-fit): {floor:.4f}; baseline generated KID ~{base40:.3f} = {base40 / floor:.0f}x the floor")

# ---------------------------------------------------------------- C. FP residual vs KID vs validation MSE (metric validity)
H = np.array([r["H"] for r in rows]); M = np.array([r["mse"] for r in rows]); K = np.array([r["kid"] for r in rows])
rh, ph = spearmanr(H, K); rm, pm = spearmanr(M, K)
say("=" * 100); say(f"C. ACROSS {len(rows)} TRAINED MODELS (S3/S4, subset KID): Spearman(FP residual H, KID) = {rh:.2f} (p {ph:.2f}) | "
                    f"Spearman(validation MSE, KID) = {rm:.2f} (p {pm:.4f})")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, x, lab, rr, pp in ((ax[0], H, "FP residual H (lower = more FP-consistent)", rh, ph), (ax[1], M * 1e3, "validation MSE (x 1e-3)", rm, pm)):
    a.scatter(x, K, s=46, color=C["blue"], edgecolor=C["surf"], linewidth=1.5, zorder=3)
    a.set_xlabel(lab); a.set_title(f"Spearman {rr:.2f} (p = {pp:.3g})", fontsize=10, color=C["ink"], loc="left")
ax[0].set_ylabel("KID-mammo (lower = better)")
fig.suptitle(f"The FP residual does not track sample quality; the denoising error does ({len(rows)} models)", x=0.02, y=1.04, ha="left", fontsize=11)
fig.savefig(f"{OUT}/fig_C_residual_vs_kid.png"); plt.close(fig)

# ---------------------------------------------------------------- D. forest plot of all comparisons
fig, ax = plt.subplots(figsize=(8, 6.2))
for i, (lab, st, fam, r, lo, hi, ref) in enumerate(CMP[::-1]):
    y = i
    if lo is not None:
        ax.plot([1 + lo / ref, 1 + hi / ref], [y, y], color=FAMC[fam], lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([r], [y], s=60, color=FAMC[fam], edgecolor=C["surf"], linewidth=1.5, zorder=3, marker="o" if lo is not None else "D")
ax.axvline(1.0, color=C["ink2"], lw=1); ax.axvline(0.90, color=C["aqua"], lw=1.2, ls="--")
ax.axvspan(1 - 2 * sd_ck / mu_ck, 1 + 2 * sd_ck / mu_ck, color=C["grid"], alpha=0.6, zorder=0)
ax.text(0.90, len(CMP) - 0.4, " pre-registered bar (0.90)", color=C["aqua"], fontsize=8, ha="right")
ax.text(1 + 2 * sd_ck / mu_ck, -0.9, f"grey band: +/- 2 checkpoint SDs ({200 * sd_ck / mu_ck:.0f}%)", color=C["ink2"], fontsize=8, ha="right")
ax.set_yticks(range(len(CMP))); ax.set_yticklabels([f"{c[0]}  [{c[1]}]" for c in CMP[::-1]])
ax.set_xscale("log"); ax.set_xticks([0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]); ax.set_xticklabels(["0.8", "0.9", "1.0", "1.2", "1.5", "2.0", "2.5"])
ax.set_xlabel("KID-mammo ratio vs control (log scale; < 1 = better)  |  bars = 95 % paired-bootstrap CI (diamonds: CI not computed)")
for fam, col in FAMC.items():
    ax.scatter([], [], color=col, label=fam)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Every intervention with decision rules fixed before its run (CBIS-DDSM, validation stage)", loc="left", fontsize=11)
ax.set_ylim(-1.3, len(CMP) - 0.1)
fig.savefig(f"{OUT}/fig_D_forest_all_interventions.png"); plt.close(fig)

# ---------------------------------------------------------------- E. FP mechanism (S3 optimisation logs)
lam = [0, 0.016, 0.054, 0.107]
clip = [0.77, 0.89, 0.96, 0.97]; pull = [0, 0.408, 1.38, 2.74]; cos = [None, 0.005, 0.001, 0.002]
kidr = [1, 0.7928 / 0.5139, 1.1576 / 0.5139, 1.3270 / 0.5139]; mser = [1, 1.1721, 1.3796, 1.5248]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
ax[0].plot(lam, pull, "-o", color=C["red"], lw=2, ms=7); ax[0].axhline(1, color=C["ink2"], lw=1, ls=":")
ax[0].set_title("|lambda g_FP| / |g_base|\n(median over steps)", fontsize=9, loc="left")
ax[1].plot(lam, [100 * c for c in clip], "-o", color=C["orange"], lw=2, ms=7)
ax[1].set_title("steps clipped at norm 1.0 (%)\n(control lambda = 0: 77 %)", fontsize=9, loc="left")
ax[2].plot(lam, kidr, "-o", color=C["blue"], lw=2, ms=7, label="KID-mammo")
ax[2].plot(lam, mser, "-s", color=C["violet"], lw=2, ms=7, label="validation MSE")
ax[2].axhline(1, color=C["ink2"], lw=1); ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title("ratio to the lambda = 0 control\n(epoch 42, > 1 = worse)", fontsize=9, loc="left")
for a in ax: a.set_xlabel("FP weight lambda")
fig.suptitle("Fokker-Planck term: orthogonal to the denoising gradient (mean cos 0.001-0.005), dominates the update, degrades the model",
             x=0.02, y=1.10, ha="left", fontsize=10)
fig.savefig(f"{OUT}/fig_E_fp_mechanism.png"); plt.close(fig)

# ---------------------------------------------------------------- F. gradient SNR + residual profile (S4a)
B = json.load(open(f"{SP}/s4res/S4_B_gradsnr.json"))
prof = B["profile"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot([p["t"] for p in prof], [p["rho_rel"] for p in prof], "-o", color=C["red"], lw=2, ms=5)
ax[0].set_xscale("log"); ax[0].set_xlabel("diffusion step t"); ax[0].set_title("relative FP violation of the baseline\n(0 = perfectly consistent)", fontsize=9, loc="left")
snr_t = [2, 50, 150, 300]; snr = [0.040, 0.120, 0.081, 0.056]
ax[1].bar([str(t) for t in snr_t], snr, color=C["red"], width=0.55, label="FP term (one sample)")
ax[1].axhline(0.166, color=C["blue"], lw=2, label="denoising loss (one micro-batch)")
ax[1].set_xlabel("diffusion step t"); ax[1].legend(frameon=False, fontsize=8, loc="upper right"); ax[1].set_ylim(0, 0.24)
ax[1].set_title("gradient signal-to-noise ratio of the FP term\n(SNR 1 needs 69-620 samples, >= 35 at the upper CI)", fontsize=9, loc="left")
fig.savefig(f"{OUT}/fig_F_fp_snr_profile.png"); plt.close(fig)

# ---------------------------------------------------------------- G. saturation vs KID (fixing an artifact != realism)
S5 = np.load(f"{SP}/s5res/S5_data.npz")
real_sat = float(S5["istats_real"][:, 1].mean()); pts = []
for k in ("e3_ep035", "e3_ep040", "e3_ep041", "e3_ep042", "e3_ep044", "e3_avg"):
    pts.append((k.replace("e3_", ""), float(S5[f"istats_{k}"][:, 1].mean()), E3[k]["kid"] / base40, C["blue"]))
for lab, sat, r in (("dyn. thr.", 0.0080, 1.346), ("avg+dyn. thr.", 0.0030, 1.496), ("avg+dt+bo2", 0.0031, 1.277), ("noise layer", 0.0667, 1.415)):
    pts.append((lab, sat, r, C["orange"] if lab != "noise layer" else C["violet"]))
fig, ax = plt.subplots(figsize=(6.4, 4))
for lab, s, r, col in pts:
    ax.scatter([100 * s], [r], s=60, color=col, edgecolor=C["surf"], linewidth=1.5, zorder=3)
    ax.annotate(lab, (100 * s, r), xytext=(5, 3), textcoords="offset points", fontsize=8, color=C["ink2"])
ax.axvline(100 * real_sat, color=C["aqua"], lw=1.5, ls="--"); ax.text(100 * real_sat, ax.get_ylim()[1], " real images", color=C["aqua"], fontsize=8, va="top")
ax.axhline(1, color=C["ink2"], lw=1)
ax.set_xlabel("saturated breast pixels (%)"); ax.set_ylabel("KID-mammo ratio vs epoch-40 baseline")
ax.set_title("Moving saturation to the real value does not improve KID", loc="left", fontsize=11)
fig.savefig(f"{OUT}/fig_G_saturation_vs_kid.png"); plt.close(fig)
rs, ps = spearmanr([p[1] for p in pts], [p[2] for p in pts])
say("=" * 100); say(f"G. saturation vs KID over {len(pts)} sets: Spearman {rs:.2f} (p {ps:.2f}); real saturation {100 * real_sat:.2f} %")

# ---------------------------------------------------------------- G2. high-frequency energy vs KID (exploratory)
hf_pts = [(k.replace("e3_", ""), float(S5[f"istats_{k}"][:, 2].mean()), E3[k]["kid"] / base40) for k in ("e3_ep035", "e3_ep040", "e3_ep041", "e3_ep042", "e3_ep044", "e3_avg")]
hf_pts += [("avg B", 0.0024, 0.998), ("dyn. thr.", 0.0021, 1.346), ("avg+dyn. thr.", 0.0026, 1.496), ("best-of-2", 0.0016, 0.911),
           ("avg+dt+bo2", 0.0018, 1.277), ("noise layer", 0.0022, 1.415)]
real_hf = float(S5["istats_real"][:, 2].mean())
rhf, phf = spearmanr([p[1] for p in hf_pts], [p[2] for p in hf_pts])
say("=" * 100); say(f"G2. high-frequency energy vs KID over {len(hf_pts)} sets (exploratory): Spearman {rhf:.2f} (p {phf:.2f}); real HF {real_hf:.4f}, "
    f"generated range {min(p[1] for p in hf_pts):.4f}-{max(p[1] for p in hf_pts):.4f}")
fig, ax = plt.subplots(figsize=(6.4, 4))
for lab, h, r in hf_pts:
    ax.scatter([1e3 * h], [r], s=60, color=C["blue"], edgecolor=C["surf"], linewidth=1.5, zorder=3)
    ax.annotate(lab, (1e3 * h, r), xytext=(5, 3), textcoords="offset points", fontsize=8, color=C["ink2"])
ax.axvline(1e3 * real_hf, color=C["aqua"], lw=1.5, ls="--"); ax.text(1e3 * real_hf, 1.5, " real images", color=C["aqua"], fontsize=8, va="top")
ax.axhline(1, color=C["ink2"], lw=1)
ax.set_xlabel("high-frequency energy of breast pixels (x 1e-3)"); ax.set_ylabel("KID-mammo ratio vs epoch-40 baseline")
ax.set_title(f"More high-frequency energy goes with WORSE KID (Spearman {rhf:.2f}, p {phf:.2f})", loc="left", fontsize=11)
fig.savefig(f"{OUT}/fig_G2_hf_energy_vs_kid.png"); plt.close(fig)

# ---------------------------------------------------------------- H. noise-level function: CBIS measured vs fits; CMMD fit (S8 log)
nl = json.load(open(f"{SP}/s7res/S7_nlf_seed7007.json"))["fit"]
c = np.array(nl["bin_centers"]); V = np.array(nl["measured_var"]); w = np.array(nl["bin_weight"])
xx = np.linspace(0.08, 0.95, 100)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(c, V * 1e3, s=20 + 400 * w, color=C["blue"], edgecolor=C["surf"], linewidth=1.2, zorder=3, label="measured (size = pixel share)")
lin = nl["linear_poisson_gaussian_only"]
ax[0].plot(xx, (lin["b"] + lin["a"] * xx) * 1e3, color=C["orange"], lw=2, label=f"Poisson-Gaussian fit (R2 {lin['R2_var']:.2f})")
ax[0].plot(xx, (nl["b"] + nl["a"] * xx + nl["c"] * xx ** 2) * 1e3, color=C["ink2"], lw=1.5, ls="--", label=f"+ quadratic term (R2 {nl['R2_var']:.2f})")
ax[0].set_title("CBIS-DDSM (digitised film, 256 px)\nhighest in DARK regions, U-shaped: not photon-noise-like", fontsize=9, loc="left")
ax[1].plot(xx, (2.2634e-4 + 1.7299e-3 * xx) * 1e3, color=C["orange"], lw=2, label="Poisson-Gaussian fit (R2 0.76)")
ax[1].plot(xx, (-2.054e-4 + 4.888e-3 * xx - 4.313e-3 * xx ** 2) * 1e3, color=C["ink2"], lw=1.5, ls="--", label="+ quadratic term (R2 0.98)")
ax[1].set_title("CMMD (digital FFDM, 256 px) - fitted curves only (S8 log)\nrises with brightness up to mid intensity, then flattens/falls", fontsize=9, loc="left")
for a in ax:
    a.set_xlabel("local mean intensity (per-image min-max)"); a.set_ylabel("high-pass variance (x 1e-3)"); a.legend(frameon=False, fontsize=7.5)
fig.savefig(f"{OUT}/fig_H_noise_level_function_cbis_vs_cmmd.png"); plt.close(fig)

# ---------------------------------------------------------------- I. memorisation / nearest neighbours in feature space
F = np.load(f"{SP}/s7res/S7_val_features.npz")
say("=" * 100); say("I. MEMORISATION CHECK (nearest-neighbour distances in feature space; train-fit = images the DDPM was trained on)")
for sp in ("mammo", "dino"):
    R = F[f"ref_{sp}"]; G = F[f"v_base_{sp}"]
    rng = np.random.default_rng(0); sub = rng.choice(len(R), 400, replace=False)
    Drr = cdist(R[sub], R); Drr[np.arange(len(sub)), sub] = np.inf; loo = Drr.min(1)          # real -> nearest OTHER real
    dgr = cdist(G, R).min(1)                                                                       # generated -> nearest real
    q05 = np.quantile(loo, 0.05)
    say(f"  {sp:5s}: median NN distance real->real {np.median(loo):.2f} | generated->real {np.median(dgr):.2f} "
        f"(ratio {np.median(dgr) / np.median(loo):.2f}) | generated closer than the 5th percentile of real->real: "
        f"{100 * np.mean(dgr < q05):.1f} % (near-copies)")
say("  (S4a also measured pixel-level memorisation = 0.000 for every arm.)")

open(f"{OUT}/analysis_results.txt", "w").write("\n".join(REP) + "\n")
print("figures:", sorted(os.listdir(OUT)))
