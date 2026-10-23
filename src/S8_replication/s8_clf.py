# s8_clf.py - KID-mammo feature extractor for a NEW dataset (CMMD): the S2 classifier recipe (a6_classifier.train_clf,
# ConvNeXt-Tiny, 15 epochs, seed 0, final epoch) trained on train-fit = train minus the patient-level validation split.
# Validation AUC is reported (S2 rule: usable as feature extractor if AUC >= 0.65; reported, not used for any decision).
import os, sys, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, OUT
from a6_classifier import train_clf, predict, auc

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)                 # e.g. cmmd_clf_s0
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
TAG = "S8-clf"
xtr, ytr, pat, _ = load_clean_meta("train"); vm = val_mask(pat)
X, Y, XV, YV = xtr[~vm], ytr[~vm], xtr[vm], ytr[vm]
if a.smoke:
    X, Y, a.epochs = X[:64], Y[:64], 1
t0 = time.time()
m = train_clf(X, Y, a.seed, a.epochs, tag=TAG)
p = predict(m, XV); va = auc(p, YV)
log(TAG, f"{a.name}: trained on {len(Y)} train-fit images | VALIDATION AUC {va:.4f} on {len(YV)} images "
         f"({'usable' if va >= 0.65 else 'BELOW 0.65 - KID-mammo less informative, reported'}) | {(time.time() - t0) / 60:.1f} min")
torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(OUT, f"{a.name}.pt"))
save_json({"name": a.name, "val_auc": va, "n_train": int(len(Y)), "n_val": int(len(YV))}, f"{a.name}.json")
