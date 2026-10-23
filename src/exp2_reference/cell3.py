# ===================== CELL 3 — ~10 min smoke test: train + resume + eval on BOTH GPUs =====================
# Runs in commits too. If anything fails, Cell 4 will NOT start the long run.
import os, re, glob, shutil
SM, EV = "/kaggle/working/ckpt_smoke", "/kaggle/working/eval_smoke"
shutil.rmtree(SM, ignore_errors=True); shutil.rmtree(EV, ignore_errors=True)
common = ["--smoke_batches", "30", "--ckpt_dir", SM, "--snap_every", "1", "--log_every", "10"]
def train_jobs(ep):
    return [(0, ["train_arm2.py", "--arm", "base2", "--lam", "0.0", "--epochs", str(ep)] + common),
            (1, ["train_arm2.py", "--arm", "pinn2", "--lam", "0.01", "--epochs", str(ep)] + common)]
ev_args = ["--epoch", "2", "--n_per_class", "8", "--steps", "5", "--bs", "8", "--ckpt_dir", SM, "--out", EV]
rc1, o1 = run_parallel(train_jobs(1), poll_min=5)
rc2, o2 = run_parallel(train_jobs(2), poll_min=5) if rc1 == [0, 0] else ([1, 1], ["", ""])
rc3, o3 = (run_parallel([(0, ["eval_arm.py", "--arm", "base2"] + ev_args),
                         (1, ["eval_arm.py", "--arm", "pinn2"] + ev_args)], poll_min=5)
           if rc2 == [0, 0] else ([1, 1], ["", ""]))

S = r"START epoch=(\d+) step=(\d+) opt_states=(\d+) lam=\S+ checksum=(\S+)"
def g(pat, s):
    m = re.search(pat, s); return m.groups() if m else None
def lc(s):   # last logged (loss, comp)
    m = re.findall(r"loss (\S+) comp (\S+) mse", s); return (float(m[-1][0]), float(m[-1][1])) if m else None
checks = []
for k, arm in enumerate(["base2", "pinn2"]):
    checks += [
        (f"{arm}: run 1 started from scratch (epoch 0, step 0)", lambda k=k: g(S, o1[k])[:2] == ("0", "0")),
        (f"{arm}: run 2 RESUMED at epoch 1, step 30",          lambda k=k: g(S, o2[k])[:2] == ("1", "30")),
        (f"{arm}: run 2 optimizer state restored",             lambda k=k: int(g(S, o2[k])[2]) > 0),
        (f"{arm}: rolling files written atomically",
         lambda arm=arm: all(os.path.exists(f"{SM}/{arm}_{x}") for x in
                             ["last.pt", "w_latest.pt", "w_ep001.pt", "w_ep002.pt", "history.json"])
                         and not glob.glob(f"{SM}/*.tmp")),
        (f"{arm}: evaluation pipeline produced results",       lambda arm=arm: os.path.exists(f"{EV}/results_{arm}_ep002.json")),
    ]
checks += [
    ("both arms start from IDENTICAL weights",               lambda: g(S, o1[0])[3] == g(S, o1[1])[3]),
    ("base2 loss == original composite loss (no physics)",   lambda: abs(lc(o2[0])[0] - lc(o2[0])[1]) < 1e-6),
    ("pinn2 loss > composite (physics hinge is active)",     lambda: lc(o2[1])[0] > lc(o2[1])[1]),
    ("oracle classifier found and evaluated",                lambda: "oracle on REAL test" in o3[0]),
]
def ok(f):
    try: return bool(f())
    except Exception: return False
print("\n" + "=" * 64)
for name, f in checks:
    print(("PASS  " if ok(f) else "FAIL  ") + name)
SMOKE_OK = all(ok(f) for _, f in checks)
if SMOKE_OK:
    shutil.rmtree(SM, ignore_errors=True); shutil.rmtree(EV, ignore_errors=True)
    print("ALL PASS -> smoke files deleted, Cell 4 will start the long run.")
else:
    print("SMOKE TEST FAILED -> Cell 4 will not train. Send Claude this block + the first error lines above.")
