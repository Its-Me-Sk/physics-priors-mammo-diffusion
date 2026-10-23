# ===================== CELL 4 — FULL TRAINING: base2 on GPU 0 + pinn2 on GPU 1, at the same time =====================
# BUDGET_H = hours of THIS session the training may use (measured from Cell 1).
# RULE: BUDGET_H must be at least 0.8 h LESS than your remaining weekly GPU quota, or Kaggle kills the run
#       and the output is lost. Full run: 10.9 (needs >= 11.7 h quota left).
BUDGET_H = 10.9
import subprocess
if not globals().get("SMOKE_OK", False):
    print("Smoke test did not pass (or Cell 3 was not run) -> NOT starting the long run.")
else:
    extra = ["--budget_h", str(BUDGET_H)]
    rc, out = run_parallel([(0, ["train_arm2.py", "--arm", "base2", "--lam", "0.0"] + extra),
                            (1, ["train_arm2.py", "--arm", "pinn2", "--lam", "0.01"] + extra)], poll_min=30)
    print("exit codes (0 = ok):", rc)
    print(subprocess.run("ls -lh /kaggle/working/ckpt", shell=True, capture_output=True, text=True).stdout)
