# a9_metrics.py - A9: baseline audit on generated images (seed-42 clean baseline, guidance 1/2/3/4). TEST SET NOT USED.
# Reference images: the clean TRAIN set, split by patient into train-fit and VALIDATION (phaseA_common.val_mask).
#  (a) CONDITIONING GATE (A6 classifiers, averaged probability):
#        AUC_gen - 0.5 >= 0.5 * (AUC_real,val - 0.5), tested with the lower bound of a 1000x bootstrap 95 % CI.
#        The 50 % is OUR chosen criterion (not an established standard). If A6 fails its own validation gate,
#        conditioning is reported descriptively only.
#  (b) VIEW: predicted MLO share (A6b view classifier) of generated vs real VALIDATION images, per class.
#  (c) INTENSITY: breast-pixel mean, saturation (share of breast pixels > 0.95), high-frequency energy, breast area,
#        border brightness - generated vs real validation, per class.
#  (d) METRIC SANITY (DINOv2 ViT-B/14, Inception-v3, mammography features = A6 seed 0 penultimate layer):
#        valid if KID(train-fit sample, validation) < KID(validation, mildest degradation) and KID rises
#        monotonically with blur (sigma .5/1/2 px) and with noise (sd .02/.05/.10).
#  (e) CALIBRATION KID of each guidance value vs the TRAIN-FIT images (not test) with equal n (100 per class),
#        plus a sample-size check for guidance 4 (100 vs 200 per class).
#  (f) MEMORISATION: nearest TRAIN image in pixel space (64x64 correlation) and in DINOv2 space (cosine distance);
#        reference = validation images -> train-fit images (different patients); share of generated images more
#        similar than the reference 99th percentile; sheet of the 8 most similar generated/train pairs.
#  (g) PRE-WRITTEN GUIDANCE RULE (recommendation for the LOCK): eligible guidance = conditioning gate passes AND
#        saturation <= 2x real AND |view-share gap| <= 0.20 in both classes; choose the eligible value with the lowest
#        calibration KID (primary feature space = DINOv2 if valid, else mammography features). If none eligible:
#        lowest calibration KID, flagged "conditioning not usable -> H3 exploratory".
# KID = unbiased MMD^2, cubic polynomial kernel, 100 subsets of 100 (same function for every feature space).
import os, sys, glob, json, math, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, val_mask, OUT
from a6_classifier import build, prep, auc, boot_auc

TAG = "A9"
dev = torch.device("cuda")
rep = {}
t_all = time.time()
x, y, pat, _ = load_clean_meta("train")
vm = val_mask(pat)
XV, YV, XF, YF = x[vm], y[vm], x[~vm], y[~vm]
GS = {}
for f in sorted(glob.glob(os.path.join(OUT, "gen_s42_ep040_gs*.npz"))):
    k = os.path.basename(f)[len("gen_s42_ep040_"):-4]
    if "samenoise" not in k:
        GS[k] = np.load(f)
GS = {k: {"x": d["x"], "y": d["y"]} for k, d in GS.items()}
log(TAG, f"generated sets: { {k: len(v['y']) for k, v in GS.items()} } | validation {len(YV)} | train-fit {len(YF)}")
def first_n(d, n):
    keep = np.concatenate([np.where(d["y"] == c)[0][:n] for c in (0, 1)])
    return {"x": d["x"][keep], "y": d["y"][keep]}

# ---------------- classifiers ----------------
def load_clf(f):
    m = build(); m.load_state_dict({k: v.float() for k, v in torch.load(f, map_location="cpu").items()}); return m.to(dev).eval()
clfs = [load_clf(f) for f in sorted(glob.glob(os.path.join(OUT, "a6_clf_s*.pt")))]
viewm = load_clf(os.path.join(OUT, "a6b_view.pt")) if os.path.exists(os.path.join(OUT, "a6b_view.pt")) else None
@torch.no_grad()
def probs(models, X_u8, bs=64):
    out = np.zeros(len(X_u8))
    for m in models:
        ps = []
        for i in range(0, len(X_u8), bs):
            xb = torch.from_numpy(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(torch.softmax(m(prep(xb)).float(), 1)[:, 1].cpu().numpy())
        out += np.concatenate(ps) / len(models)
    return out

# ---------------- (a) conditioning gate ----------------
a6 = json.load(open(os.path.join(OUT, "A6_classifier.json"))) if os.path.exists(os.path.join(OUT, "A6_classifier.json")) else {}
cond = {}
if clfs:
    pv = probs(clfs, XV)
    auc_val = auc(pv, YV)
    thr = 0.5 + 0.5 * (auc_val - 0.5)
    cond["real_val"] = {"auc": auc_val, "ci95": boot_auc(pv, YV), "gate_threshold": thr,
                        "a6_usable_as_judge": a6.get("usable_as_judge")}
    for k, d in GS.items():
        pg = probs(clfs, d["x"]); ci = boot_auc(pg, d["y"])
        cond[k] = {"auc": auc(pg, d["y"]), "ci95": ci, "gate_pass": bool(ci[0] >= thr),
                   "p_mal_benign_class": float(pg[d["y"] == 0].mean()), "p_mal_malignant_class": float(pg[d["y"] == 1].mean())}
        log(TAG, f"conditioning {k}: AUC {cond[k]['auc']:.3f} CI {ci[0]:.3f}-{ci[1]:.3f} | gate (>= {thr:.3f}) "
                 f"{'PASS' if cond[k]['gate_pass'] else 'FAIL'} | real val AUC {auc_val:.3f}")
rep["conditioning"] = cond

# ---------------- (b) view ----------------
view = {}
if viewm is not None:
    pr = probs([viewm], XV)
    view["real_val"] = {nm: float((pr[YV == c] >= 0.5).mean()) for c, nm in ((0, "benign"), (1, "malignant"))}
    for k, d in GS.items():
        pg = probs([viewm], d["x"])
        view[k] = {nm: float((pg[d["y"] == c] >= 0.5).mean()) for c, nm in ((0, "benign"), (1, "malignant"))}
        view[k]["gap_benign"] = view[k]["benign"] - view["real_val"]["benign"]
        view[k]["gap_malignant"] = view[k]["malignant"] - view["real_val"]["malignant"]
        log(TAG, f"predicted MLO share {k}: benign {view[k]['benign']:.2f} (real {view['real_val']['benign']:.2f}) | malignant "
                 f"{view[k]['malignant']:.2f} (real {view['real_val']['malignant']:.2f})")
rep["view"] = view

# ---------------- (c) intensity statistics ----------------
def istats(X_u8):
    I = torch.from_numpy(X_u8).float().div(255).unsqueeze(1)
    mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True); I = (I - mn) / (mx - mn + 1e-8)
    mu = F.avg_pool2d(I, 7, 1, 3, count_include_pad=False); m = (mu > 0.04).float(); n = m.sum((1, 2, 3)).clamp_min(1)
    hp = I - F.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    fr = torch.ones_like(I); fr[..., 4:-4, 4:-4] = 0
    return {"mean_breast": float(((I * m).sum((1, 2, 3)) / n).mean()),
            "saturation_share": float((((I > 0.95).float() * m).sum((1, 2, 3)) / n).mean()),
            "hf_energy": float(((hp * hp * m).sum((1, 2, 3)) / n).mean()),
            "breast_area": float(m.mean((1, 2, 3)).mean()), "border_brightness": float(((I * fr).sum((1, 2, 3)) / fr.sum((1, 2, 3))).mean())}
inten = {"real_val": {nm: istats(XV[YV == c]) for c, nm in ((0, "benign"), (1, "malignant"))}}
for k, d in GS.items():
    inten[k] = {nm: istats(d["x"][d["y"] == c]) for c, nm in ((0, "benign"), (1, "malignant"))}
rep["intensity"] = inten
log(TAG, "saturation share: " + " | ".join(f"{k}: {v['benign']['saturation_share']:.3f}/{v['malignant']['saturation_share']:.3f}"
                                            for k, v in inten.items()) + "  (benign/malignant)")

# ---------------- feature extractors ----------------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def load_dino():
    try:
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True), "torch.hub dinov2_vitb14"
    except Exception as e:
        log(TAG, f"torch.hub DINOv2 failed ({repr(e)[:150]}) -> timm")
        import timm
        return timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224), "timm dinov2 vitb14"
dino, rep["dinov2_source"] = load_dino(); dino = dino.to(dev).eval()
from torchmetrics.image.kid import KernelInceptionDistance
incep = KernelInceptionDistance(feature=2048, subset_size=100).to(dev).inception
mam = clfs[0] if clfs else None
@torch.no_grad()
def feats(X01, which, bs=64):
    out = []
    for i in range(0, len(X01), bs):
        xb = X01[i:i + bs].to(dev)
        if which == "dinov2":
            z = F.interpolate(xb, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            out.append(dino((z - MEAN) / STD).float().cpu())
        elif which == "inception":
            out.append(incep((xb.clamp(0, 1) * 255).round().to(torch.uint8).repeat(1, 3, 1, 1)).float().cpu())
        else:
            with torch.autocast("cuda", dtype=torch.float16):
                h = mam.classifier[1](mam.classifier[0](mam.avgpool(mam.features(prep(xb)))))
            out.append(h.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)
def kid(fa, fb, n_sub=100, size=100, seed=0):
    r = np.random.default_rng(seed); d = fa.shape[1]; vals = []; size = min(size, len(fa), len(fb))
    for _ in range(n_sub):
        A_ = fa[r.choice(len(fa), size, replace=False)]; B_ = fb[r.choice(len(fb), size, replace=False)]
        kxx = (A_ @ A_.T / d + 1) ** 3; kyy = (B_ @ B_.T / d + 1) ** 3; kxy = (A_ @ B_.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (size * (size - 1)) + (kyy.sum() - np.trace(kyy)) / (size * (size - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
def gblur(X, s):
    k = int(2 * math.ceil(3 * s) + 1); ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    X = F.conv2d(F.pad(X, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(X, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
to01 = lambda u8: torch.from_numpy(u8).float().div(255).unsqueeze(1)
V01 = to01(XV)
F01 = to01(XF[np.random.default_rng(0).choice(len(XF), len(XV), replace=False)])
gn = torch.Generator().manual_seed(0)
deg = {f"blur{s:g}": gblur(V01, s) for s in (0.5, 1.0, 2.0)}
deg.update({f"noise{s:g}": (V01 + s * torch.randn(V01.shape, generator=gn)).clamp(0, 1) for s in (0.02, 0.05, 0.10)})
spaces = ["dinov2", "inception"] + (["mammo"] if mam is not None else [])
sanity, calib, FEAT = {}, {}, {}
for w in spaces:
    t0 = time.time()
    fv, ff = feats(V01, w), feats(F01, w)
    ffit_all = feats(to01(XF), w)
    FEAT[w] = {"val": fv, "fit_all": ffit_all}
    row = {"floor_trainfit_vs_val": kid(ff, fv)}
    for k_, X_ in deg.items():
        row[k_] = kid(fv, feats(X_, w))
    mono = lambda ks: all(row[p][0] < row[q][0] for p, q in zip(ks[:-1], ks[1:]))
    row["valid"] = bool(row["floor_trainfit_vs_val"][0] < min(row["blur0.5"][0], row["noise0.02"][0])
                        and mono(["blur0.5", "blur1", "blur2"]) and mono(["noise0.02", "noise0.05", "noise0.1"]))
    sanity[w] = row
    for k, d in GS.items():
        fg = feats(to01(first_n(d, 100)["x"]), w)
        calib.setdefault(k, {})[w] = kid(fg, ffit_all)
        if w == "dinov2":
            FEAT[w][k] = feats(to01(d["x"]), w)
    if "gs4" in GS and (GS["gs4"]["y"] == 0).sum() >= 200:
        calib.setdefault("gs4_size_check", {})[w] = {"100_per_class": calib["gs4"][w],
                                                    "200_per_class": kid(feats(to01(first_n(GS["gs4"], 200)["x"]), w), ffit_all)}
    log(TAG, f"{w}: floor {row['floor_trainfit_vs_val'][0]:.4f} | blur {[round(row[f'blur{s:g}'][0], 4) for s in (0.5, 1, 2)]} | noise "
             f"{[round(row[f'noise{s:g}'][0], 4) for s in (0.02, 0.05, 0.1)]} -> valid {row['valid']} | calibration KID "
             + " ".join(f"{k}: {calib[k][w][0]:.4f}" for k in GS) + f" | {(time.time() - t0) / 60:.1f} min")
rep["metric_sanity"] = sanity
rep["calibration_kid_vs_trainfit"] = calib
primary = "dinov2" if sanity.get("dinov2", {}).get("valid") else ("mammo" if sanity.get("mammo", {}).get("valid") else None)
rep["primary_feature_space"] = primary or "NONE valid -> discuss before LOCK"

# ---------------- (f) memorisation ----------------
def thumbs(X_u8):
    t = X_u8.reshape(len(X_u8), 64, 4, 64, 4).mean((2, 4)).reshape(len(X_u8), -1).astype(np.float64)
    t -= t.mean(1, keepdims=True); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return t
TA = thumbs(x)                                                    # ALL train images (the generator saw all of them)
TF = thumbs(XF)
ref_pix = (thumbs(XV) @ TF.T).max(1)                              # validation -> train-fit (different patients)
q99_pix = float(np.quantile(ref_pix, 0.99))
cos = lambda A_, B_: 1 - (A_ / np.linalg.norm(A_, axis=1, keepdims=True)) @ (B_ / np.linalg.norm(B_, axis=1, keepdims=True)).T
ref_dino = cos(FEAT["dinov2"]["val"], FEAT["dinov2"]["fit_all"]).min(1)
q01_dino = float(np.quantile(ref_dino, 0.01))
fdino_all = np.concatenate([FEAT["dinov2"]["fit_all"], FEAT["dinov2"]["val"]])
xall_order = np.concatenate([XF, XV])
mem = {"reference_pixel_corr_q99": q99_pix, "reference_dino_dist_q01": q01_dino}
best = []
for k, d in GS.items():
    cg = thumbs(d["x"]) @ thumbs(xall_order).T
    pc, pi = cg.max(1), cg.argmax(1)
    dd = cos(FEAT["dinov2"][k], fdino_all).min(1)
    mem[k] = {"share_pixel_above_ref_q99": float((pc > q99_pix).mean()), "max_pixel_corr": float(pc.max()),
              "share_dino_below_ref_q01": float((dd < q01_dino).mean()), "min_dino_dist": float(dd.min())}
    best += [(float(pc[i]), k, i, int(pi[i])) for i in np.argsort(-pc)[:8]]
best = sorted(best, reverse=True)[:8]
from PIL import Image
sheet = np.concatenate([np.concatenate([GS[k]["x"][i], xall_order[j]], 1) for _, k, i, j in best], 0)
Image.fromarray(sheet).save(os.path.join(OUT, "memorisation_top8_generated_vs_nearest_train.png"))
mem["top8"] = [{"set": k, "corr": c} for c, k, _, _ in best]
rep["memorisation"] = mem
log(TAG, "memorisation (pixel): " + " ".join(f"{k}: {mem[k]['share_pixel_above_ref_q99']:.3f} (max r {mem[k]['max_pixel_corr']:.3f})" for k in GS)
         + f" | reference q99 r {q99_pix:.3f}")

# ---------------- (g) guidance rule ----------------
elig, why = [], {}
for k in GS:
    ok_c = cond.get(k, {}).get("gate_pass", False)
    rs = max(inten["real_val"]["benign"]["saturation_share"], inten["real_val"]["malignant"]["saturation_share"])
    ok_s = max(inten[k]["benign"]["saturation_share"], inten[k]["malignant"]["saturation_share"]) <= 2 * rs + 1e-6
    ok_v = bool(view.get(k)) and abs(view[k]["gap_benign"]) <= 0.20 and abs(view[k]["gap_malignant"]) <= 0.20
    why[k] = {"conditioning_gate": ok_c, "saturation_ok": bool(ok_s), "view_gap_ok": bool(ok_v)}
    if ok_c and ok_s and ok_v:
        elig.append(k)
sp = primary or "inception"
pick = min(elig or list(GS), key=lambda k: calib[k][sp][0])
rep["guidance_rule"] = {"criteria_per_guidance": why, "eligible": elig, "feature_space": sp, "recommended": pick,
                        "flag": None if elig else "no guidance value eligible -> conditioning not usable -> H3 exploratory"}
log(TAG, f"guidance rule: eligible {elig} -> recommended {pick}" + ("" if elig else " (FLAG: none eligible)"))
rep["minutes"] = (time.time() - t_all) / 60
save_json(rep, "A9_metrics.json")
