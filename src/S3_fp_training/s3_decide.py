# s3_decide.py - S3 decision (pre-stated; VALIDATION images only, the test set is never used):
#   R(lam) = 1 - mean_i H_i(lam) / mean_i H_i(M0) at the final epoch, paired per probe image (same x_t, u, v), 95 % CI from
#   2000 bootstrap resamples of the 32 probe images (= 32 different patients, so image = patient bootstrap).
#   lam is ELIGIBLE if ALL of:
#     R >= 0.10 and CI lower bound > 0                          (the FP term really lowers the FP residual)
#     R_jvp > 0                                                 (the drop is also seen with exact JVP derivatives,
#                                                                i.e. training did not just exploit the FD error)
#     validation training loss <= 1.02 x M0 and validation MSE <= 1.02 x M0
#     saturation share (max over classes, guidance 2) <= 1.25 x M0 (absolute tolerance 0.005) and <= 0.10
#        (the 0.10 cap is dropped, with a flag, if M0 itself is above 0.10)
#     KID-mammo vs train-fit <= 1.5 x M0 (if KID_M0 <= 0: <= KID_M0 + its subset SD)
#   DECISION: the SMALLEST eligible lam -> LAMBDA_SELECTED (goes into the LOCK for Phase C)
#             else, if any lam has CI lower bound > 0 -> INCONCLUSIVE (S3b: extend those arms by 2 epochs)
#             else -> H1_FAIL (the FP term does not measurably lower the residual at these lam -> stop, report)
import os, sys, json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/kaggle/working/S3")
ap.add_argument("--ckpt_dir", default="/kaggle/working/ckpt")
ap.add_argument("--arms", required=True)                  # "s3_m0:0,s3_l016:0.016,..." control (lam = 0) first
ap.add_argument("--final_epoch", type=int, default=42)
ap.add_argument("--epochs", default="41,42")
ap.add_argument("--base_name", default="base")
a = ap.parse_args()
ARMS = [(s.split(":")[0], float(s.split(":")[1])) for s in a.arms.split(",")]
CTRL = ARMS[0][0]; EPOCHS = [int(v) for v in a.epochs.split(",")]
L = []
P_ = lambda *m: L.append(" ".join(str(x) for x in m))

def load(arm, ep):
    f = os.path.join(a.out, f"eval_{arm}_ep{ep:03d}")
    if not (os.path.exists(f + ".npz") and os.path.exists(f + ".json")):
        return None
    return dict(np.load(f + ".npz")), json.load(open(f + ".json"))

def boot_R(Hl, Hm, n=2000, seed=0):
    r = np.random.default_rng(seed); idx = r.integers(0, len(Hl), (n, len(Hl)))
    num, den = Hl[idx].mean(1), Hm[idx].mean(1)
    Rb = 1 - num / np.where(np.abs(den) > 0, den, np.nan)
    return [float(np.nanquantile(Rb, 0.025)), float(np.nanquantile(Rb, 0.975))]

def boot_ratio(xl, xm, n=2000, seed=1):
    r = np.random.default_rng(seed); idx = r.integers(0, len(xl), (n, len(xl)))
    q = xl[idx].mean(1) / xm[idx].mean(1)
    return [float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975))]

def fp_log(arm):
    f = os.path.join(a.ckpt_dir, f"{arm}_fp_log.jsonl")
    if not os.path.exists(f):
        return None
    recs = [json.loads(l) for l in open(f) if l.strip()]
    recs = [r for r in recs if r["epoch"] in EPOCHS]
    if not recs:
        return None
    g = lambda k: np.array([r[k] for r in recs if r.get(k) is not None], float)
    out = {"optimizer_steps": len(recs), "median_base_norm": float(np.median(g("base_norm"))),
           "median_combined_norm": float(np.median(g("combined_norm"))),
           "clipped_share": float((g("combined_norm") > 1.0).mean()), "median_clip_factor": float(np.median(g("clip_factor"))),
           "mean_update_norm": float(g("update_norm").mean()), "mean_pred_dL_base": float(g("pred_dL_base").mean())}
    if len(g("fp_norm_weighted")):
        fpn, bn = g("fp_norm_weighted"), np.array([r["base_norm"] for r in recs if r.get("fp_norm_weighted") is not None], float)
        out.update({"median_fp_norm_weighted": float(np.median(fpn)), "median_fp_over_base": float(np.median(fpn / bn)),
                    "mean_cos_base_fp": float(np.nanmean(g("cos_base_fp"))),
                    "mean_pred_dL_fp_norm": float(g("pred_dL_fp_norm").mean()),
                    "share_steps_FP_predicted_to_decrease": float((g("pred_dL_fp_norm") < 0).mean()),
                    "fp_skipped": int(sum(1 for r in recs if r.get("fp_ok") is False)),
                    "mean_L_fp_norm_per_epoch": {str(e): float(np.mean([r["L_fp_norm"] for r in recs if r["epoch"] == e and r.get("fp_ok")]))
                                                  for e in EPOCHS if any(r["epoch"] == e and r.get("fp_ok") for r in recs)}})
    return out

P_("=" * 110); P_("S3 REPORT - Fokker-Planck feasibility and lambda selection (validation images only; TEST SET NOT USED)"); P_("=" * 110)
P_(f"arms: {ARMS} | control {CTRL} | decision epoch {a.final_epoch}")
ctrl = load(CTRL, a.final_epoch)
res = {"arms": {}, "control": CTRL, "final_epoch": a.final_epoch}
base = load(a.base_name, 40)

# ---------------- residual trajectories ----------------
P_("\n[1] FP RESIDUAL ON THE PROBE SET  (H = mean over t of r1 r2 / S_ref^2; lower = more FP-consistent)")
if base is not None:
    P_(f"  epoch 40 (shared start): H {base[0]['H'].mean():.4f} | per t " + " ".join(f"{t}:{v:.3f}" for t, v in base[1]["P_mean_per_t"].items()))
for arm, lam in ARMS:
    row = []
    for ep in EPOCHS:
        d = load(arm, ep)
        row.append(f"ep{ep} H {d[0]['H'].mean():.4f}" if d else f"ep{ep} missing")
    P_(f"  {arm:10s} (lam {lam:g}): " + " | ".join(row))
if ctrl is None:
    P_("\n!! control arm missing at the decision epoch -> no decision possible (see logs)")
    res["decision"] = "NO_DECISION_MISSING_CONTROL"
else:
    Hm = ctrl[0]["H"]; qm = ctrl[1].get("quality", {}); jm = ctrl[1].get("jvp_check", {})
    P_(f"\n  control {CTRL} epoch {a.final_epoch}: H {Hm.mean():.4f} | per t " + " ".join(f"{t}:{v:.3f}" for t, v in ctrl[1]["P_mean_per_t"].items()))
    if Hm.mean() <= 0:
        P_("  !! control H is not positive -> the probe set cannot resolve the residual; R is undefined")
    sat_m = qm.get("saturation_max_class"); kid_m = qm.get("kid_mammo_vs_trainfit")
    pj_m = np.array(jm.get("P_jvp", []), float)
    flags = []
    if sat_m is not None and sat_m > 0.10:
        flags.append(f"control saturation {sat_m:.3f} > 0.10 -> absolute cap dropped")
    P_("\n[2] PER-ARM CRITERIA at the decision epoch (vs control, paired)")
    for arm, lam in ARMS[1:]:
        d = load(arm, a.final_epoch)
        if d is None:
            P_(f"  {arm}: missing at epoch {a.final_epoch}"); res["arms"][arm] = {"lam": lam, "missing": True}; continue
        Hl, J = d[0]["H"], d[1]
        R = float(1 - Hl.mean() / Hm.mean()) if Hm.mean() > 0 else float("nan")
        ci = boot_R(Hl, Hm) if Hm.mean() > 0 else [float("nan")] * 2
        per_t = {t: float(1 - J["P_mean_per_t"][t] / ctrl[1]["P_mean_per_t"][t]) if ctrl[1]["P_mean_per_t"][t] > 0 else None
                 for t in J["P_mean_per_t"]}
        vc = d[0]["val_comp"].mean(1); vcm = ctrl[0]["val_comp"].mean(1)
        vs = d[0]["val_mse"].mean(1); vsm = ctrl[0]["val_mse"].mean(1)
        r_comp, r_mse = float(vc.mean() / vcm.mean()), float(vs.mean() / vsm.mean())
        q = J.get("quality", {}); jl = J.get("jvp_check", {})
        pj_l = np.array(jl.get("P_jvp", []), float)
        R_jvp = float(1 - pj_l.mean() / pj_m.mean()) if len(pj_l) and len(pj_m) == len(pj_l) and pj_m.mean() > 0 else float("nan")
        sat = q.get("saturation_max_class"); kidl = q.get("kid_mammo_vs_trainfit")
        c = {"R": R, "R_ci95": ci, "R_per_t": per_t, "R_jvp": R_jvp, "fd_vs_jvp_rel_err": jl.get("rel_err_fd_vs_jvp"),
             "val_loss_ratio": r_comp, "val_loss_ratio_ci95": boot_ratio(vc, vcm), "val_mse_ratio": r_mse,
             "val_mse_ratio_ci95": boot_ratio(vs, vsm), "saturation": sat, "saturation_control": sat_m,
             "kid_mammo": kidl, "kid_mammo_control": kid_m, "memorisation": q.get("memorisation")}
        ok = {"R>=0.10": bool(R >= 0.10), "CI_lo>0": bool(ci[0] > 0), "R_jvp>0": bool(R_jvp > 0),
              "val_loss<=1.02": bool(r_comp <= 1.02), "val_mse<=1.02": bool(r_mse <= 1.02)}
        if sat is not None and sat_m is not None:
            ok["saturation"] = bool(sat <= max(1.25 * sat_m, sat_m + 0.005) and (sat <= 0.10 or sat_m > 0.10))
        else:
            ok["saturation"] = False
        if kidl is not None and kid_m is not None:
            ok["kid_mammo"] = bool(kidl[0] <= 1.5 * kid_m[0]) if kid_m[0] > 0 else bool(kidl[0] <= kid_m[0] + kid_m[1])
        else:
            ok["kid_mammo"] = False
        c["checks"] = ok; c["eligible"] = all(ok.values()); c["lam"] = lam
        c["optimisation"] = fp_log(arm)
        res["arms"][arm] = c
        P_(f"  {arm} (lam {lam:g}): R {R:+.3f} CI {ci[0]:+.3f}..{ci[1]:+.3f} | R_jvp {R_jvp:+.3f} (FD-vs-JVP err "
           f"{c['fd_vs_jvp_rel_err'] if c['fd_vs_jvp_rel_err'] is None else round(c['fd_vs_jvp_rel_err'], 4)}) | val loss x{r_comp:.4f} "
           f"mse x{r_mse:.4f} | saturation {sat if sat is None else round(sat, 4)} (control {sat_m if sat_m is None else round(sat_m, 4)}) | "
           f"KID-mammo {None if kidl is None else round(kidl[0], 4)} (control {None if kid_m is None else round(kid_m[0], 4)})")
        P_(f"      R per t: " + " ".join(f"{t}:{v:+.2f}" if v is not None else f"{t}:n/a" for t, v in per_t.items()))
        P_(f"      checks: {ok} -> ELIGIBLE {c['eligible']}")
    # ---------------- decision ----------------
    elig = sorted([(c["lam"], arm) for arm, c in res["arms"].items() if c.get("eligible")])
    sig = [arm for arm, c in res["arms"].items() if not c.get("missing") and c["R_ci95"][0] > 0]
    if elig:
        res["decision"], res["lambda"], res["selected_arm"] = "LAMBDA_SELECTED", elig[0][0], elig[0][1]
    elif sig:
        res["decision"], res["lambda"] = "INCONCLUSIVE", None
        res["s3b_extend_arms"] = sig
    else:
        res["decision"], res["lambda"] = "H1_FAIL", None
    res["flags"] = flags
    P_("\n[3] DECISION")
    P_(f"  {res['decision']}" + (f" -> lambda = {res['lambda']:g} ({res['selected_arm']}) for the LOCK / Phase C" if res.get("lambda") else "")
       + (f" -> S3b: extend {sig} by 2 epochs" if res["decision"] == "INCONCLUSIVE" else "")
       + (" -> the FP term did not lower the residual measurably at these lambda" if res["decision"] == "H1_FAIL" else ""))
    for f in flags:
        P_(f"  FLAG: {f}")

# ---------------- optimisation diagnostics ----------------
P_("\n[4] OPTIMISATION (training logs of the S3 epochs; norms before the clip at 1.0)")
for arm, lam in ARMS:
    o = fp_log(arm)
    res.setdefault("optimisation", {})[arm] = o
    if o is None:
        P_(f"  {arm}: no log"); continue
    P_(f"  {arm} (lam {lam:g}): {o['optimizer_steps']} steps | median |g_base| {o['median_base_norm']:.3g} | median |g_total| "
       f"{o['median_combined_norm']:.3g} | clipped {o['clipped_share']:.2f} (median factor {o['median_clip_factor']:.3f}) | "
       f"mean |update| {o['mean_update_norm']:.3g} | predicted dL_base per step {o['mean_pred_dL_base']:+.3e}")
    if "median_fp_norm_weighted" in o:
        P_(f"      FP: median |lam g_FP| {o['median_fp_norm_weighted']:.3g} (x{o['median_fp_over_base']:.3g} of base) | mean cos(base, FP) "
           f"{o['mean_cos_base_fp']:+.3f} | predicted dL_FP(norm) per step {o['mean_pred_dL_fp_norm']:+.3e} (decrease in "
           f"{o['share_steps_FP_predicted_to_decrease']:.2f} of steps) | training L_FP per epoch {o['mean_L_fp_norm_per_epoch']} | "
           f"skipped {o['fp_skipped']}")
    hp = os.path.join(a.ckpt_dir, f"{arm}_history.json")
    if os.path.exists(hp):
        for h in json.load(open(hp)):
            P_(f"      epoch {h['epoch']}: train loss {h['loss']:.4f} mse {h['mse']:.4f} | {h['minutes']:.1f} min")
P_("\n" + "=" * 110)
P_("Send S3_report.txt (or S3_results.zip) and the grid_*.png images back. The LOCK document is written after this.")
txt = "\n".join(L)
open(os.path.join(a.out, "S3_report.txt"), "w").write(txt)
json.dump(res, open(os.path.join(a.out, "S3_decision.json"), "w"), indent=1, default=float)
print(txt)
