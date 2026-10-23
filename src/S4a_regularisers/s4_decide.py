# s4_decide.py - S4a screening decision (VALIDATION / train-fit only; the TEST SET is never used). Pre-stated rule:
#   PRIMARY: KID-mammo = unbiased MMD^2 (cubic polynomial kernel, ConvNeXt mammography features of the A6 seed-0
#   classifier) between the 200 generated images (100 / class, guidance 2, the same starting noise and label for image i
#   in every arm) and ALL train-fit images. Paired bootstrap over generated images (2000 resamples, same resample for the
#   arm and the control; reference set fixed).
#   An arm PASSES if ALL of:
#     KID_arm <= 0.90 x KID_control  and  the 95 % CI of (KID_arm - KID_control) lies entirely below 0
#     validation MSE <= 1.02 x control  and  validation training loss <= 1.02 x control
#     saturation share (max over classes) <= max(1.25 x control, control + 0.005)
#     memorisation: share of generated images above the validation->train-fit pixel-correlation q99 <= 0.01
#   WINNER = the passing arm with the largest KID reduction (reported with all other arms).
# AMENDMENT 1 (2026-10-21 01:12 Asia/Dhaka, made while S4a was running and BEFORE any screening result was seen):
#   the validation-error guards (validation MSE and validation training loss) are loosened from <= 1.02x to <= 1.15x
#   control (--err_guard). Reason, stated in advance: the score-norm penalty shrinks the noise prediction and therefore
#   raises validation MSE BY CONSTRUCTION even when samples improve (cf. Epsilon Scaling, Ning et al. ICLR 2024); the
#   guard is meant to catch a broken denoiser (S3: +17..+52 %), which 1.15x still does. All other criteria unchanged.
#   The report shows the outcome under BOTH the original (1.02x) and the amended (1.15x) rule.
#   MECHANISM (reported, not part of the pass rule): FP residual change R = 1 - H_arm / H_control on the 32 probe images
#   (patient bootstrap); the winner is called "physics-consistent" if R >= 0.10 with CI lower bound > 0, otherwise
#   "generic regularisation" (the benefit is not explained by a lower Fokker-Planck residual).
import os, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/kaggle/working/S4")
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
ap.add_argument("--arms", required=True)          # "s3_m0:none:0,s4_r1a:sn:0.25,..." control first
ap.add_argument("--epoch", type=int, default=44)
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--err_guard", type=float, default=1.15)   # amendment 1 (original pre-stated value 1.02)
a = ap.parse_args()
ARMS = [tuple(s.split(":")) for s in a.arms.split(",")]
CTRL = ARMS[0][0]
L = []
P_ = lambda *m: L.append(" ".join(str(x) for x in m))
J = lambda f: json.load(open(os.path.join(a.out, f))) if os.path.exists(os.path.join(a.out, f)) else None
fmt = lambda v, n=4: "n/a" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

def load_eval(arm):
    n = f"eval_{arm}_ep{a.epoch:03d}"
    if not (os.path.exists(os.path.join(a.out, n + ".json")) and os.path.exists(os.path.join(a.out, n + ".npz"))):
        return None
    d = {"json": J(n + ".json"), "npz": dict(np.load(os.path.join(a.out, n + ".npz")))}
    fg = os.path.join(a.out, f"feat_{arm}_ep{a.epoch:03d}_gs2.npy"); fr = os.path.join(a.out, f"feat_trainfit_{arm}_ep{a.epoch:03d}.npy")
    d["feat"] = np.load(fg).astype(np.float64) if os.path.exists(fg) else None
    d["ref"] = np.load(fr).astype(np.float64) if os.path.exists(fr) else None
    return d

kern = lambda A, B: (A @ B.T / A.shape[1] + 1.0) ** 3
def mmd_parts(G, R):
    return kern(G, G), kern(G, R).mean(1)
def mmd_w(KGG, h, kRR, c):
    num = c @ KGG @ c - (c * c * np.diag(KGG)).sum(); den = c.sum() ** 2 - (c * c).sum()
    return num / den - 2 * (c * h).sum() / c.sum() + kRR

P_("=" * 110); P_("S4a REPORT - gradient signal test + cheap FP-derived regulariser screening (validation only; TEST SET NOT USED)"); P_("=" * 110)

# ---------------- B ----------------
b = J("S4_B_gradsnr.json")
P_("\n[B1] FOKKER-PLANCK RESIDUAL PROFILE of the epoch-40 baseline (32 validation probe images, 2 probes)")
if b is None:
    P_("  missing (see logs/B_gradsnr.log)")
else:
    P_("     t   sigma   mean r1r2 (z)          RMS(u.d_tau s)   relative violation")
    for r in b["profile"]:
        P_(f"  {r['t']:4d}  {r['sigma']:.4f}  {r['mean_r1r2']:+.3e} ({r['z']:6.2f})   {r['lhs_rms']:.3e}        {r['rho_rel']:.3f}")
    P_("\n[B2] EXACT-JVP CHECK OF THE FD STEP AT SMALL t (rule <= 0.10)")
    for r in b["jvp_small_t"]:
        P_(f"  {r}")
    bb = b["base_gradient"]
    P_("\n[B3] GRADIENT SIGNAL-TO-NOISE (SNR of ONE sample; 'needed' = samples to average for SNR 1)")
    P_(f"  base loss, per micro-batch of 2: SNR {bb['snr_single']:.3f} -> micro-batches needed {bb['K_needed']:.1f} "
       f"(one optimizer step averages 4)")
    for r in b["fp_snr"]:
        P_(f"  FP t={r['t']:3d}: SNR {r['snr_single']:.3f} -> samples needed {r['K_needed']:.1f} (at the upper 95 % of |mean|: "
           f"{r['K_needed_at_upper95_mu']:.1f}) | image-level/population |mean|^2 "
           f"{(r['image_level_mu_norm2'] / r['mu_norm2_unbiased']) if r['mu_norm2_unbiased'] > 0 else float('nan'):.1f} | "
           f"cos(mean FP, mean base) {fmt(r['cos_mean_fp_vs_mean_base'], 3)} | mean pull vs base at lam .054: "
           f"{fmt(r['mean_pull_ratio_lam0.054'], 3)}")

# ---------------- screening ----------------
ctrl = load_eval(CTRL)
res = {"control": CTRL, "epoch": a.epoch, "arms": {}}
P_(f"\n[S1] SCREENING at epoch {a.epoch} vs control {CTRL} (paired)")
if ctrl is None or ctrl["feat"] is None:
    P_("  !! control evaluation or features missing -> no decision"); res["decision"] = "NO_DECISION_MISSING_CONTROL"
else:
    R0 = ctrl["ref"]
    KRR = kern(R0, R0); kRR = (KRR.sum() - np.trace(KRR)) / (len(R0) * (len(R0) - 1)); del KRR
    n = len(ctrl["feat"])
    KGc, hc = mmd_parts(ctrl["feat"], R0)
    ones = np.ones(n)
    mm_c = mmd_w(KGc, hc, kRR, ones)
    rng = np.random.default_rng(0)
    C = rng.multinomial(n, np.ones(n) / n, size=a.n_boot).astype(float)
    boot_c = np.array([mmd_w(KGc, hc, kRR, c) for c in C])
    qc, vc, mc = ctrl["json"].get("quality", {}), ctrl["npz"]["val_mse"].mean(1), ctrl["npz"]["val_comp"].mean(1)
    Hc = ctrl["npz"]["H"]
    P_(f"  control: KID-mammo (full reference) {mm_c:.4f} | subset KID {qc.get('kid_mammo_vs_trainfit')} | val MSE {vc.mean():.5f} | "
       f"saturation {fmt(qc.get('saturation_max_class'))} | H {Hc.mean():.4f}")
    sat_c = qc.get("saturation_max_class")
    for arm, reg, rho in ARMS[1:]:
        d = load_eval(arm)
        if d is None or d["feat"] is None:
            P_(f"  {arm}: missing"); res["arms"][arm] = {"missing": True}; continue
        q = d["json"].get("quality", {})
        KG, h = mmd_parts(d["feat"], R0)
        mm = mmd_w(KG, h, kRR, ones)
        boot = np.array([mmd_w(KG, h, kRR, c) for c in C])
        diff = boot - boot_c
        ci = [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
        ratio = float(mm / mm_c)
        va, ma = d["npz"]["val_mse"].mean(1), d["npz"]["val_comp"].mean(1)
        r_mse, r_comp = float(va.mean() / vc.mean()), float(ma.mean() / mc.mean())
        Ha = d["npz"]["H"]
        R = float(1 - Ha.mean() / Hc.mean())
        idx = rng.integers(0, len(Ha), (a.n_boot, len(Ha)))
        Rb = 1 - Ha[idx].mean(1) / Hc[idx].mean(1)
        R_ci = [float(np.quantile(Rb, 0.025)), float(np.quantile(Rb, 0.975))]
        sat = q.get("saturation_max_class"); mem = q.get("memorisation", {}).get("share_above_q99")
        ok = {"KID<=0.90x": bool(ratio <= 0.90), "KID_CI<0": bool(ci[1] < 0),
              f"val_mse<={a.err_guard:g}": bool(r_mse <= a.err_guard), f"val_loss<={a.err_guard:g}": bool(r_comp <= a.err_guard),
              "saturation": bool(sat is not None and sat_c is not None and sat <= max(1.25 * sat_c, sat_c + 0.005)),
              "memorisation<=0.01": bool(mem is not None and mem <= 0.01)}
        c = {"reg": reg, "rho": float(rho), "lam": d["json"].get("lam"), "kid_full": float(mm), "kid_ratio": ratio,
             "kid_diff_ci95": ci, "kid_subset": q.get("kid_mammo_vs_trainfit"), "val_mse_ratio": r_mse, "val_loss_ratio": r_comp,
             "saturation": sat, "memorisation_share": mem, "R_fp": R, "R_fp_ci95": R_ci,
             "intensity": q.get("intensity"), "checks": ok, "passed": all(ok.values())}
        c["passed_original_rule_1.02"] = bool(c["passed"] and r_mse <= 1.02 and r_comp <= 1.02)
        res["arms"][arm] = c
        P_(f"  {arm} ({reg}, rho {rho}, lambda {fmt(c['lam'], 5)}): KID {mm:.4f} = x{ratio:.3f} of control "
           f"(diff CI {ci[0]:+.4f}..{ci[1]:+.4f}) | val MSE x{r_mse:.4f} loss x{r_comp:.4f} | saturation {fmt(sat)} | "
           f"memorisation {fmt(mem, 3)} | FP residual R {R:+.3f} (CI {R_ci[0]:+.3f}..{R_ci[1]:+.3f})")
        if q.get("intensity") and qc.get("intensity"):
            P_("      HF energy b/m: " + f"{q['intensity']['benign']['hf_energy']:.4f}/{q['intensity']['malignant']['hf_energy']:.4f} (control "
               f"{qc['intensity']['benign']['hf_energy']:.4f}/{qc['intensity']['malignant']['hf_energy']:.4f}, real "
               f"{qc['intensity_real_val']['benign']['hf_energy']:.4f}/{qc['intensity_real_val']['malignant']['hf_energy']:.4f}) | "
               f"brightness b/m {q['intensity']['benign']['mean_breast']:.3f}/{q['intensity']['malignant']['mean_breast']:.3f}")
        P_(f"      checks {ok} -> {'PASS' if c['passed'] else 'fail'} (amended rule) | original 1.02x rule: "
           f"{'PASS' if c['passed_original_rule_1.02'] else 'fail'}")
    passed = sorted([(c["kid_ratio"], arm) for arm, c in res["arms"].items() if c.get("passed")])
    passed_orig = sorted([(c["kid_ratio"], arm) for arm, c in res["arms"].items() if c.get("passed_original_rule_1.02")])
    res["winner_under_original_rule"] = passed_orig[0][1] if passed_orig else None
    P_(f"\n[S2] DECISION (amended rule: validation-error guards <= {a.err_guard:g}x; under the original 1.02x rule the winner "
       f"would be {res['winner_under_original_rule'] or 'NONE'})")
    if passed:
        w = passed[0][1]; cw = res["arms"][w]
        kind = "physics-consistent" if (cw["R_fp"] >= 0.10 and cw["R_fp_ci95"][0] > 0) else "generic regularisation"
        res.update({"decision": "WINNER", "winner": w, "winner_kind": kind})
        P_(f"  WINNER {w} ({cw['reg']}, rho {cw['rho']}): KID x{cw['kid_ratio']:.3f} of control -> goes to the LOCK / Phase C | "
           f"mechanism: {kind} (FP residual R {cw['R_fp']:+.3f})")
        if len(passed) > 1:
            P_(f"  also passed: {[p[1] for p in passed[1:]]}")
    else:
        res["decision"] = "NONE_PASSED"
        best = sorted([(c["kid_ratio"], arm) for arm, c in res["arms"].items() if not c.get("missing")])
        P_("  NONE PASSED" + (f" | closest: {best[0][1]} (KID x{best[0][0]:.3f})" if best else ""))

# ---------------- optimisation ----------------
P_("\n[S3] OPTIMISATION (training logs; norms before the clip at 1.0)")
for arm, reg, rho in ARMS:
    f = os.path.join(a.ckpt_dir, f"{arm}_reg_log.jsonl")
    if not os.path.exists(f):
        P_(f"  {arm}: no log"); continue
    R_ = [json.loads(l) for l in open(f) if l.strip()]
    g = lambda k: np.array([r[k] for r in R_ if r.get(k) is not None], float)
    line = (f"  {arm} ({reg}): {len(R_)} steps | median |g_base| {np.median(g('base_norm')):.3g} | clipped "
            f"{np.mean(g('combined_norm') > 1):.2f} | mean |update| {np.nanmean(g('update_norm')):.3g}")
    if len(g("reg_norm_weighted")):
        rn = np.array([r["reg_norm_weighted"] / r["base_norm"] for r in R_ if r.get("reg_norm_weighted") and r["base_norm"] > 0])
        line += (f" | |lam g_reg|/|g_base| median {np.median(rn):.3f} p90 {np.quantile(rn, .9):.3f} | capped "
                 f"{np.mean(g('cap_factor') < 1):.2f} | cos(base, reg) {np.nanmean(g('cos_base_reg')):+.3f} | penalty active in "
                 f"{np.mean([bool(r.get('reg_active')) for r in R_]):.2f} of steps")
        pr = g("pred_dL_reg_norm")
        if len(pr):
            line += f" | regulariser predicted to decrease in {np.mean(pr < 0):.2f} of steps"
    P_(line)
    hp = os.path.join(a.ckpt_dir, f"{arm}_history.json")
    if os.path.exists(hp):
        for h in json.load(open(hp))[-4:]:
            P_(f"      epoch {h['epoch']}: loss {h['loss']:.4f} mse {h['mse']:.4f} | {h['minutes']:.1f} min")
P_("\n" + "=" * 110)
P_("Send S4_report.txt (or S4_results.zip) and the grid_*.png images back.")
txt = "\n".join(L)
open(os.path.join(a.out, "S4_report.txt"), "w").write(txt)
json.dump(res, open(os.path.join(a.out, "S4_decision.json"), "w"), indent=1, default=float)
print(txt)
