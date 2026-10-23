# s6_downstream.py - S6 downstream utility (SECONDARY, descriptive; pre-stated): the fixed A6 classifier recipe
# (ConvNeXt-Tiny, 15 epochs, seeds 0/1/2) trained on the real train-fit images, with or without the 300 synthetic images
# of one test-stage set added (same labels as generated). Evaluated once on the untouched TEST images (AUC benign vs
# malignant). Summary (s6_stats.downstream_summary): mean +- SD over seeds and the seed-ensemble AUC difference vs
# real-only with a 2000x stratified bootstrap CI.
import os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, load_cache, val_mask, OUT
from a6_classifier import train_clf, predict, auc

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--synthetic", default="")          # gen_<name>.npz to add, empty = real only
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
TAG = f"S6-down-{a.arm}"
xtr, ytr, pat, _ = load_clean_meta("train"); fit = ~val_mask(pat)
X, Y = xtr[fit], ytr[fit]
if a.synthetic:
    d = np.load(os.path.join(OUT, f"gen_{a.synthetic}.npz")); X, Y = np.concatenate([X, d["x"]]), np.concatenate([Y, d["y"]])
XT, YT = load_cache("test")
if a.smoke:
    X, Y, XT, YT, a.epochs = X[:64], Y[:64], XT[:32], YT[:32], 1
log(TAG, f"train {len(Y)} images ({'+ ' + a.synthetic if a.synthetic else 'real only'}) | test {len(YT)}")
preds = {}
for s in [int(v) for v in a.seeds.split(",")]:
    t0 = time.time(); m = train_clf(X, Y, s, a.epochs, tag=TAG); p = predict(m, XT); preds[str(s)] = p
    log(TAG, f"seed {s}: TEST AUC {auc(p, YT):.4f} | {(time.time() - t0) / 60:.1f} min"); del m; torch.cuda.empty_cache()
np.savez(os.path.join(OUT, f"down_{a.arm}.npz"), y=YT, **{f"p{k}": v for k, v in preds.items()})
save_json({"arm": a.arm, "synthetic": a.synthetic, "auc": {k: auc(v, YT) for k, v in preds.items()}}, f"down_{a.arm}.json")
