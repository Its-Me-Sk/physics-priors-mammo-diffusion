# a6b_view.py - A6b: CC-vs-MLO view classifier on REAL images (label "MLO" = 1), same fixed recipe as A6, 1 seed,
# 8 epochs, trained on train-fit patients with a known view, checked on validation patients (never on test).
# Used by A9 to measure whether generated "malignant" images are side views (MLO) more often than real ones.
# PRE-STATED RULE: usable if validation AUC >= 0.95 (view is an easy, geometric property).
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, OUT
from a6_classifier import train_clf, predict, auc

TAG = "A6b"
t0 = time.time()
x, y, pat, _ = load_clean_meta("train")
vl = np.load(os.path.join(OUT, "A10_view_labels.npz"))["view_train"].astype(str)
vm = val_mask(pat)
known = np.isin(vl, ["CC", "MLO"])
yv = (vl == "MLO").astype(int)
fit, val = known & ~vm, known & vm
log(TAG, f"fit {fit.sum()} images (MLO share {yv[fit].mean():.3f}) | validation {val.sum()} images")
m = train_clf(x[fit], yv[fit], seed=0, epochs=8, tag=TAG)
pv = predict(m, x[val])
A = auc(pv, yv[val])
acc = float(((pv >= 0.5) == (yv[val] == 1)).mean())
torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(OUT, "a6b_view.pt"))
res = {"n_fit": int(fit.sum()), "n_val": int(val.sum()), "val_auc": A, "val_accuracy": acc, "usable": bool(A >= 0.95),
       "minutes": (time.time() - t0) / 60}
save_json(res, "A6b_view.json")
log(TAG, f"view classifier: validation AUC {A:.4f} accuracy {acc:.3f} -> usable {res['usable']} | {res['minutes']:.1f} min")
