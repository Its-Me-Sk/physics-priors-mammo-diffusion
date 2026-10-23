# report.py - collects the Phase A JSON files into one plain-text report (phaseA_report.txt).
# It only REPORTS the pre-stated rules of each script; the LOCK decisions are written afterwards, by hand, with the RA.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import OUT

L = []
def P(*m):
    L.append(" ".join(str(x) for x in m))
def J(name):
    try:
        return json.load(open(os.path.join(OUT, name)))
    except Exception:
        return None
f = lambda v, d=3: "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))

P("=" * 100); P("PHASE A REPORT (diagnostics only, no training)"); P("=" * 100)

# ---------------- A1 ----------------
r = J("A1_leakage.json")
P("\n[A1] LEAKAGE / DUPLICATES (hard gate)")
if r is None:
    P("  missing (A1 did not finish - see logs/A1.log)")
else:
    P(f"  VERDICT: {r['verdict']}")
    P(f"  patients train/test/both: {r['patients_train']} / {r['patients_test']} / {r['patients_in_both']}"
      f"   images in both splits: {r['images_in_both_splits']}   test-train near-duplicate pairs (r>0.995): "
      f"{r['test_train_pairs_corr_gt_0.995']} (exactly equal: {r['test_train_pairs_exact_equal']})")
    P(f"  cache sizes train/test: {r['cache_train_n']} / {r['cache_test_n']}   test->train max-corr quantiles: "
      f"{ {k: round(v, 4) for k, v in r['test_to_train_max_corr_quantiles'].items()} }")
    P(f"  train rows / unique images / repeats / label conflicts: {r['train_rows']} / {r['train_unique_images']} / "
      f"{r['train_rows_that_repeat_an_image']} / {r['train_images_with_conflicting_labels']}")
    P(f"  test  rows / unique images / repeats / label conflicts: {r['test_rows']} / {r['test_unique_images']} / "
      f"{r['test_rows_that_repeat_an_image']} / {r['test_images_with_conflicting_labels']}")
    P(f"  train-cache duplicate pairs (r>0.995): {r['train_cache_pairs_corr_gt_0.995']}  of which different labels: "
      f"{r['train_cache_dup_pairs_with_different_labels']}")
    for w in r.get("warnings", []):
        P("  WARNING:", w)

# ---------------- A2 ----------------
for which in ("main", "25"):
    r = J(f"A2_jensen_{which}.json")
    P(f"\n[A2-{which}] CONDITIONAL-MEAN (JENSEN) TEST on the real network" + ("" if which == "main" else " (25 % data)"))
    if r is None:
        P("  missing" + (" (no 25 %-data checkpoint attached, or A2-25 not run)" if which == "25" else " (see logs/A2-main.log)"))
        continue
    P(f"  checkpoint {os.path.basename(r['checkpoint'])} epoch {r['epoch']} | {r['n_img']} test images | real L4 band "
      f"[{r['real_band_L4'][0]:.3f}, {r['real_band_L4'][1]:.3f}] | real out-of-band share {r['f_out_real_all']:.2f}")
    P(f"  t* = {r['t_star']}   (rule: x0_hat out-of-band share <= {r['threshold']:.2f} for every grid t up to t*)")
    P("     t | out-of-band share: x0_hat  Bastek  samples  real16 | L4 gap vs real (SD): x0_hat  samples | "
      "Jensen gap S(x0_hat)-mean S(sample) (SD): L1   L2   L3   L4 | K")
    for w in r["rows"]:
        jg = w["jensen_S_of_mean_minus_mean_of_S_sd"]
        P(f"  {w['t']:4d} |   {w['out_band_xhat']:.2f}    {w['out_band_bastek']:.2f}    {w['out_band_samples']:.2f}    "
          f"{w['out_band_real_subset']:.2f} |   {w['gap_sd_xhat'][3]:+.2f}   {w['gap_sd_samples'][3]:+.2f} | "
          f"{jg[0]:+.2f} {jg[1]:+.2f} {jg[2]:+.2f} {jg[3]:+.2f} | {w.get('k_samples', '')}")

# ---------------- A3 ----------------
r = J("A3_texture.json")
P("\n[A3] NOISE vs TEXTURE + LESION SIZE (CPU)")
if r is None:
    P("  missing (see logs/A3.log)")
else:
    P(f"  native JPEG size (median WxH): {r['native_size_median_WxH']}  downsample factor to 256: "
      f"{[round(v, 1) for v in r['downsample_factor_to_256_median']]}")
    P(f"  resolution test ratios (median): { {k: round(v, 3) for k, v in r['resolution_ratios_median'].items()} }")
    P(f"  RESOLUTION VERDICT (flat 256/512; <=0.35 noise-like, >=0.70 texture): {r['resolution_verdict']}")
    P(f"  edge/flat high-pass variance (median over bins): {f(r['edge_over_flat_variance_median'])}")
    P(f"  calibration: high-pass attenuation expected {r['calibration_highpass_attenuation_expected']:.3f} | "
      f"recovered/expected {r['calibration_recovered_over_expected_median']:.3f} | "
      f"texture leak clean/known {r['calibration_texture_leak_clean_over_known_median']:.3f}")
    P(f"  lesion size at 256 px: {r['lesion_size_at_256']}")

# ---------------- A4 ----------------
r = J("A4_fp.json") or J("A4_fp_partial.json")
P("\n[A4] FOKKER-PLANCK RESIDUAL ESTIMATOR on the real network")
if r is None:
    P("  missing (see logs/A4.log)")
else:
    if r.get("partial"):
        P("  !! PARTIAL (A4 stopped early) - rows below are the finished t values only")
    P(f"  reference: {r['reference']}")
    if "verdict_estimator" in r:
        P(f"  ESTIMATOR VERDICT: {r['verdict_estimator']}")
        P(f"  FP residual measurable (z>3) at t = {r['t_with_z_gt_3']} | all t in [10,300]: "
          f"{r['fp_residual_measurable_all_t_10_300']}")
        for prec, c in r["precision_choice"].items():
            P(f"  {prec}: best {c['best_setting']} worst rel.err (t 10-300) {c['worst_rel_err_t10_300']:.3f} usable={c['usable']}"
              f" | all: { {k: round(v, 3) for k, v in c['all_settings_worst_rel_err'].items()} }")
        for c in r.get("fp64_crosscheck", []):
            P(f"  fp64 cross-check: {c}")
    P("     t |     z    rho | term RMS: LHS     u.s      J-term   D2-term | rel.err fp32 p=1e-3 3e-3 1e-2 | "
      "fp16 p=1e-3 3e-3 1e-2 | dtau k=2,5 | guided/unguided")
    for w in r["rows"]:
        m = w["methods"]
        e32 = " ".join(f"{m[k]['rel_err_r']:.3f}" for k in m if k.startswith("fp32"))
        e16 = " ".join(f"{m[k]['rel_err_r']:.3f}" for k in m if k.startswith("fp16"))
        tr = w["term_rms"]
        dk = " ".join(f"{v:.3f}" for v in w["dtau_step_rel_change_vs_k1"].values())
        P(f"  {w['t']:4d} | {w['z']:6.2f} {w['relative_residual_rho']:.3f} | {tr['L']:.2e} {tr['us']:.2e} {tr['Jt']:.2e} "
          f"{tr['D2']:.2e} | {e32} | {e16} | {dk} | {w['guided']['rms_ratio_guided_over_unguided']:.2f}")

# ---------------- A5 ----------------
r = J("A5_bench.json")
P("\n[A5] COMPUTE BENCHMARK")
if r is None:
    P("  missing (see logs/A5.log)")
else:
    P(f"  unit test manual-recompute vs autograd: rel diff {r['unit_test_rel_diff_manual_vs_autograd']:.1e} "
      f"-> {'PASS' if r['unit_test_pass'] else 'FAIL'}")
    P(f"  UNet {r['unet_parameters_M']:.1f} M parameters | baseline micro-step {r['baseline_micro_step_s']:.3f} s, "
      f"peak {r['baseline_peak_mem_GB']:.1f} GB, ~{r['baseline_epoch_min_est_1240_steps']:.1f} min/epoch")
    P(f"  FP setting used: {r['fp_setting_used']}")
    for k, v in r["fp_manual_recompute"].items():
        P(f"  FP term (1 image, 13 evaluations) manual recompute {k}: {v}")
    P(f"  FP term full-graph autograd fp16: {r['fp_full_graph_fp16']}")
    if "cost_ratio" in r:
        P(f"  cost ratio (FP on 1 image every k micro-steps): { {k: round(v, 2) for k, v in r['cost_ratio'].items()} }")
        P(f"  est. min/epoch: { {k: round(v, 1) for k, v in r['epoch_min_est'].items()} }")
    P(f"  SEED RULE (k=4): {r['seed_rule_k4']}")
    P(f"  Phase B warm-start file: {r['base2_last']}")

P("\n" + "=" * 100)
P("Send this whole file (or phaseA_results.zip) back to Claude. Nothing has been decided yet: the LOCK is written after.")
txt = "\n".join(L)
open(os.path.join(OUT, "phaseA_report.txt"), "w").write(txt)
print(txt)
