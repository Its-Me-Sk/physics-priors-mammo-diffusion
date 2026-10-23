# ===================== CELL 5 — EVALUATION base2 vs pinn2 (run in the NEXT session) =====================
# Before running: Add Input -> Your Work -> (this notebook's training version) so its /ckpt folder is visible.
# Run Cells 1, 2, then this cell. Do NOT run Cells 3/4 in the evaluation session.
import os, re, glob, json, math, time
import numpy as np
A, B = "base2", "pinn2"
used = (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
def find(name):
    fs = glob.glob(f"/kaggle/working/ckpt/{name}") + glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return fs[0] if fs else None
def epochs_of(arm):
    fs = glob.glob(f"/kaggle/working/ckpt/{arm}_w_ep*.pt") + glob.glob(f"/kaggle/input/**/{arm}_w_ep*.pt", recursive=True)
    return {int(re.search(r"_w_ep(\d+)\.pt$", f).group(1)) for f in fs}
common = sorted(epochs_of(A) & epochs_of(B))
if used > 10.0:
    print(f"{used:.1f} h already used this session -> skipping evaluation; run Cells 1, 2, 5 next session.")
elif not common:
    print("No matching snapshots found. Attach the training notebook's output (Add Input -> Your Work).")
else:
    E = common[-1]                                   # same epoch for both arms = fair comparison
    print(f"Evaluating both arms at epoch {E} (common snapshots: {common})")
    rc, out = run_parallel([(0, ["eval_arm.py", "--arm", A, "--epoch", str(E)]),
                            (1, ["eval_arm.py", "--arm", B, "--epoch", str(E)])], poll_min=10)
    D = "/kaggle/working/eval"
    ra, rb = [json.load(open(f"{D}/results_{x}_ep{E:03d}.json")) for x in (A, B)]
    ca, cb = [np.load(f"{D}/samples_{x}_ep{E:03d}.npz")["correct"].astype(bool) for x in (A, B)]
    def wilson(k, n, z=1.96):
        p = k / n; den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())   # paired: same starting noise per sample
    n = b + c
    p_mc = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    REF = {"original (Gaussian, old run)": dict(oracle=0.600, ben=0.285, mal=0.915, KID=0.0698, FID=119.01, phys=1.0118)}
    print("\n" + "=" * 76)
    print(f"{'metric':34s}{A:>14s}{B:>14s}{'original ref':>14s}")
    o = REF["original (Gaussian, old run)"]
    for key, lab, ref in [("mean", "oracle acc (mean)", o["oracle"]), ("benign", "oracle acc benign", o["ben"]),
                          ("malignant", "oracle acc malignant", o["mal"])]:
        print(f"{lab:34s}{ra['oracle_acc'][key]:>14.3f}{rb['oracle_acc'][key]:>14.3f}{ref:>14.3f}")
    la, ha = wilson(int(ca.sum()), len(ca)); lb, hb = wilson(int(cb.sum()), len(cb))
    print(f"{'  95% CI (Wilson)':34s}{f'[{la:.3f},{ha:.3f}]':>14s}{f'[{lb:.3f},{hb:.3f}]':>14s}")
    print(f"{'KID (lower=better)':34s}{ra['KID_mean']:>14.4f}{rb['KID_mean']:>14.4f}{o['KID']:>14.4f}")
    print(f"{'  KID std':34s}{ra['KID_std']:>14.4f}{rb['KID_std']:>14.4f}")
    print(f"{'FID (lower=better)':34s}{ra.get('FID', float('nan')):>14.2f}{rb.get('FID', float('nan')):>14.2f}{o['FID']:>14.2f}")
    print(f"{'physics residual (closer to REAL)':34s}{ra['physics_residual_generated']:>14.4f}"
          f"{rb['physics_residual_generated']:>14.4f}{o['phys']:>14.4f}")
    print(f"{'physics residual, REAL test':34s}{ra['physics_residual_real_test']:>14.4f}")
    print(f"McNemar {A} vs {B}: {A}-only correct={b}, {B}-only correct={c}, exact p={p_mc:.4f}")
    print(f"Oracle accuracy on REAL test images: {ra.get('oracle_real_acc')}")
    json.dump({"epoch": E, A: ra, B: rb, "mcnemar": {"b": b, "c": c, "p": p_mc}},
              open(f"{D}/summary2_ep{E:03d}.json", "w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for arm in (A, B):
        f = find(f"{arm}_history.json")
        if f:
            h = json.load(open(f)); ep = [x["epoch"] for x in h]
            ax[0].plot(ep, [x["comp"] for x in h], label=arm)
            ax[1].plot(ep, [x["phys"] for x in h], label=arm)
            ax[2].plot(ep, [x["hinge"] for x in h], label=arm)
    ax[0].set_title("original composite loss"); ax[1].set_title("physics residual (t-weighted)")
    ax[2].set_title("physics hinge violation (t-weighted)")
    for x in ax: x.set_xlabel("epoch"); x.legend()
    plt.tight_layout(); plt.savefig(f"{D}/training_curves2.png", dpi=120); plt.show()
    print(f"\nSaved in {D}: summary2_ep{E:03d}.json, training_curves2.png, grid_{A}_ep{E:03d}.png, grid_{B}_ep{E:03d}.png")
    print("DOWNLOAD THESE NOW (interactive sessions delete /kaggle/working when they end).")
