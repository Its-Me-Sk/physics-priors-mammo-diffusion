# report2.py - S2 baseline-audit report (applies only the pre-stated rules written in each script header)
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import OUT

L = []
P = lambda *m: L.append(" ".join(str(x) for x in m))
def J(n):
    try:
        return json.load(open(os.path.join(OUT, n)))
    except Exception:
        return None
P("=" * 100); P("S2 BASELINE AUDIT REPORT (clean baseline seed 42; the TEST SET is not used for any decision)"); P("=" * 100)

r = J("A10_metadata.json")
P("\n[A10] REAL DATA BY CLASS (view / laterality / density / digitiser)")
if r is None:
    P("  missing (see logs/A10.log)")
else:
    for g, v in r.items():
        P(f"  {g}: MLO share benign {v['benign']['share_MLO']:.3f} (n={v['benign']['n']}) | malignant {v['malignant']['share_MLO']:.3f} "
          f"(n={v['malignant']['n']}) | malignant-benign {v['MLO_share_malignant_minus_benign']:+.3f} "
          f"CI {v['MLO_diff_ci95_patient_bootstrap'][0]:+.3f}..{v['MLO_diff_ci95_patient_bootstrap'][1]:+.3f}")
        for f in ("density", "scanner", "abnormality"):
            P(f"     {f}: benign {v['benign'][f]} | malignant {v['malignant'][f]}")
        P(f"     LEFT share: benign {v['benign']['share_LEFT']:.3f} | malignant {v['malignant']['share_LEFT']:.3f}")

r = J("A6_classifier.json")
P("\n[A6] PATHOLOGY CLASSIFIER (rule: mean VALIDATION AUC >= 0.65)")
if r is None:
    P("  missing (see logs/A6.log)")
else:
    for s, v in r["seeds"].items():
        P(f"  seed {s}: validation AUC {v['val_auc']:.3f} | test AUC (reported only) {v['test_auc_reported_only']:.3f} | {v['minutes']:.1f} min")
    P(f"  mean validation AUC {r['mean_val_auc']:.3f} +- {r['sd_val_auc']:.3f} | ensemble {r['ensemble_val_auc']:.3f} "
      f"CI {r['ensemble_val_auc_ci95'][0]:.3f}-{r['ensemble_val_auc_ci95'][1]:.3f} -> USABLE AS JUDGE: {r['usable_as_judge']}")
r = J("A6b_view.json")
P("\n[A6b] VIEW CLASSIFIER CC vs MLO (rule: validation AUC >= 0.95)")
P("  missing (see logs/A6b.log)" if r is None else f"  validation AUC {r['val_auc']:.4f} accuracy {r['val_accuracy']:.3f} -> usable {r['usable']}")

P("\n[A7] GENERATION (seed 42; speed; guidance-0 identity test)")
for f in sorted(os.listdir(OUT)):
    if f.startswith("A7_gen_") and f.endswith(".json"):
        g = J(f)
        for k, v in g["runs"].items():
            P(f"  {f[7:-5]} {k}: {v['images']} images, {v['minutes']:.1f} min, {v['s_per_image']:.2f} s/image"
              + (f" | IDENTITY TEST: max |benign-malignant| {v['identity_test']['max_abs_diff_float']:.2e} -> label ignored "
                 f"{v['identity_test']['label_ignored']}" if "identity_test" in v else ""))

r = J("A9_metrics.json")
if r is None:
    P("\n[A9] missing (see logs/A9.log)")
else:
    c = r["conditioning"]
    P("\n[A9a] CONDITIONING GATE: AUC_gen - 0.5 >= 0.5 (AUC_real,val - 0.5), lower 95 % bound")
    if "real_val" in c:
        P(f"  real validation AUC {c['real_val']['auc']:.3f} -> threshold {c['real_val']['gate_threshold']:.3f} | A6 judge usable: "
          f"{c['real_val']['a6_usable_as_judge']}")
    for k, v in c.items():
        if k != "real_val":
            P(f"  {k}: AUC {v['auc']:.3f} CI {v['ci95'][0]:.3f}-{v['ci95'][1]:.3f} -> {'PASS' if v['gate_pass'] else 'FAIL'} | mean "
              f"P(malignant): benign class {v['p_mal_benign_class']:.3f}, malignant class {v['p_mal_malignant_class']:.3f}")
    v = r["view"]
    P("\n[A9b] PREDICTED MLO SHARE (generated vs real validation)")
    if "real_val" in v:
        P(f"  real validation: benign {v['real_val']['benign']:.2f} | malignant {v['real_val']['malignant']:.2f}")
    for k, w in v.items():
        if k != "real_val":
            P(f"  {k}: benign {w['benign']:.2f} (gap {w['gap_benign']:+.2f}) | malignant {w['malignant']:.2f} (gap {w['gap_malignant']:+.2f})")
    P("\n[A9c] INTENSITY (benign / malignant)")
    for k, w in r["intensity"].items():
        P(f"  {k}: " + " | ".join(f"{s} {w['benign'][s]:.3f}/{w['malignant'][s]:.3f}" for s in w["benign"]))
    P("\n[A9d] METRIC SANITY (rule: floor < mildest degradation, monotone in blur and in noise)")
    P(f"  DINOv2 source: {r.get('dinov2_source')}")
    for w, row in r["metric_sanity"].items():
        P(f"  {w:9s}: floor {row['floor_trainfit_vs_val'][0]:.4f} | blur .5/1/2: " + " / ".join(f"{row[f'blur{s:g}'][0]:.4f}" for s in (0.5, 1, 2))
          + " | noise .02/.05/.10: " + " / ".join(f"{row[f'noise{s:g}'][0]:.4f}" for s in (0.02, 0.05, 0.1)) + f" -> VALID {row['valid']}")
    P(f"  PRIMARY FEATURE SPACE: {r['primary_feature_space']}")
    P("\n[A9e] CALIBRATION KID vs TRAIN-FIT images (100 per class; NOT the test set)")
    for k, row in r["calibration_kid_vs_trainfit"].items():
        if k == "gs4_size_check":
            for w, z in row.items():
                P(f"  size check gs4 {w}: 100/class {z['100_per_class'][0]:.4f} vs 200/class {z['200_per_class'][0]:.4f}")
        else:
            P(f"  {k}: " + " | ".join(f"{w} {z[0]:.4f}+-{z[1]:.4f}" for w, z in row.items()))
    m = r["memorisation"]
    P(f"\n[A9f] MEMORISATION: reference (validation->train-fit) pixel corr q99 {m['reference_pixel_corr_q99']:.3f}, "
      f"DINOv2 dist q01 {m['reference_dino_dist_q01']:.4f}")
    for k, z in m.items():
        if isinstance(z, dict):
            P(f"  {k}: share above pixel q99 {z['share_pixel_above_ref_q99']:.3f} (max r {z['max_pixel_corr']:.3f}) | share below "
              f"DINOv2 q01 {z['share_dino_below_ref_q01']:.3f} (min {z['min_dino_dist']:.4f})")
    g = r["guidance_rule"]
    P("\n[A9g] GUIDANCE RULE (recommendation for the LOCK)")
    for k, z in g["criteria_per_guidance"].items():
        P(f"  {k}: {z}")
    P(f"  eligible {g['eligible']} | feature space {g['feature_space']} | RECOMMENDED {g['recommended']}" + (f" | FLAG: {g['flag']}" if g["flag"] else ""))

r = J("A8_fp_clean.json")
P("\n[A8] FOKKER-PLANCK ON THE CLEAN BASELINE (validation images; fp32 FD with the p(t) table)")
if r is None:
    P("  missing (see logs/A8.log)")
else:
    for c in r["jvp_spot_check"]:
        P(f"  JVP spot check: {c}")
    for w in r["residual_rows"]:
        P(f"  t={w['t']:4d} p={w['p']:.0e} | z {w['z']:5.2f} rho {w['rho']:.3f} | LHS RMS {w['lhs_rms']:.3e}")
    P("  S_ref(t): " + ", ".join(f"{k}: {v:.3g}" for k, v in r["S_ref_table"].items()))
    g = r["gradient_diagnostics"]; med = lambda z: sorted(z)[len(z) // 2]
    P(f"  gradient: median |g_FP| {med(g['fp_grad_norm']):.3g} | median |g_base| per optimizer step "
      f"{med(g['base_optimizer_step_grad_norm(mean of 4)']):.3g} | ratio {g['ratio_median_base_step_over_median_fp']:.3g}")
    P(f"  cos(g_FP, g_base) per sample: {[round(z, 3) for z in g['cos_fp_vs_own_base']]}")
    P(f"  mean pairwise cos: FP {g['mean_pairwise_cos_fp']:+.3f} | base {g['mean_pairwise_cos_base']:+.3f} | cos(mean FP, mean base) "
      f"{g['cos_meanFP_vs_meanBase']:+.3f}")
    P(f"  lambda candidates: { {k: float(f'{v:.3g}') for k, v in g['lambda_candidates'].items()} } | {g['clip_note']}")
P("\n" + "=" * 100)
P("Send this file (or S2_results.zip) and the png images to Claude. Decisions go into the LOCK document afterwards.")
txt = "\n".join(L)
open(os.path.join(OUT, "S2_report.txt"), "w").write(txt)
print(txt)
