# s8_eval.py - S8 evaluation = s7_eval.py (same features, same pre-stated rule from s7_stats.evaluate) plus
#   --clf    : the KID-mammo feature extractor of THIS dataset (CBIS: a6_clf_s0.pt from S2, CMMD: cmmd_clf_s0.pt)
#   Inception-v3 (torchvision ImageNet weights, pool 2048) FID and KID for every set - descriptive, not in the rule.
#   --stage val : reference = train-fit images of the dataset ($PA_DATA) | --stage test: reference = its TEST images ($PA_TMP)
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, load_cache, val_mask, OUT
from a6_classifier import build, prep
from s7_stats import evaluate, _kRR, _mmd, kern

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["val", "test"], required=True)
ap.add_argument("--sets", required=True)            # comma list, the FIRST is the reference arm
ap.add_argument("--clf", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
TAG = f"{a.tag}-{a.stage}"
dev = torch.device("cuda")
names = a.sets.split(","); base = names[0]
if a.stage == "val":
    xtr, ytr, pat, _ = load_clean_meta("train"); REF = xtr[~val_mask(pat)]
else:
    REF, _ = load_cache("test")
log(TAG, f"reference: {'train-fit' if a.stage == 'val' else 'TEST'} images {REF.shape} | sets {names} | classifier {a.clf}")

mam = build(); mam.load_state_dict({k: v.float() for k, v in torch.load(a.clf, map_location="cpu").items()}); mam = mam.to(dev).eval()
def load_dino():
    try:
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    except Exception as e:
        log(TAG, f"torch.hub DINOv2 failed ({repr(e)[:150]}) -> timm")
        import timm
        return timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
dino = load_dino().to(dev).eval()
import torchvision
inc = torchvision.models.inception_v3(weights="IMAGENET1K_V1"); inc.fc = torch.nn.Identity(); inc = inc.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(X_u8, bs=50):
    fm, fd, fi = [], [], []
    for i in range(0, len(X_u8), bs):
        xb = torch.from_numpy(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            fm.append(mam.classifier[1](mam.classifier[0](mam.avgpool(mam.features(prep(xb))))).float().cpu())
        z = F.interpolate(xb, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        fd.append(dino((z - MEAN) / STD).float().cpu())
        z = F.interpolate(xb, size=299, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        fi.append(inc((z - MEAN) / STD).float().cpu())
    c = lambda L: torch.cat(L).numpy().astype(np.float64)
    return c(fm), c(fd), c(fi)
def istats_img(X_u8):
    I = torch.from_numpy(X_u8).float().div(255).unsqueeze(1)
    mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True); I = (I - mn) / (mx - mn + 1e-8)
    mu = F.avg_pool2d(I, 7, 1, 3, count_include_pad=False); m = (mu > 0.04).float(); n = m.sum((1, 2, 3)).clamp_min(1)
    hp = I - F.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    return np.stack([((I * m).sum((1, 2, 3)) / n).numpy(), ((((I > 0.95).float() * m).sum((1, 2, 3))) / n).numpy(),
                     ((hp * hp * m).sum((1, 2, 3)) / n).numpy(), m.mean((1, 2, 3)).numpy()], 1)
def fid(A, B):
    from scipy import linalg
    m1, m2 = A.mean(0), B.mean(0); c1, c2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    s, _ = linalg.sqrtm(c1 @ c2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * np.real(s)))

Fd = {}
Fd["ref_mammo"], Fd["ref_dino"], Fd["ref_incep"] = feats(REF); Fd["ref_istats"] = istats_img(REF)
for nm in names:
    d = np.load(os.path.join(OUT, f"gen_{nm}.npz"))
    Fd[f"{nm}_mammo"], Fd[f"{nm}_dino"], Fd[f"{nm}_incep"] = feats(d["x"]); Fd[f"{nm}_istats"] = istats_img(d["x"])
np.savez(os.path.join(OUT, f"{a.tag}_{a.stage}_features.npz"), **Fd)
L, out, win = evaluate({k: v for k, v in Fd.items() if not k.endswith("_incep")}, base, a.n_boot, alpha=a.alpha)
kRR = _kRR(Fd["ref_incep"]); inc_lines = []
for nm in names:
    G = Fd[f"{nm}_incep"]
    kid = _mmd(kern(G, G), kern(G, Fd["ref_incep"]).mean(1), kRR, np.ones(len(G)))
    f_ = fid(G, Fd["ref_incep"]); out.setdefault(nm, {}).update({"kid_inception": kid, "fid_inception": f_})
    inc_lines.append(f"  {nm:16s}: Inception KID {kid:.5f} | Inception FID {f_:.2f}  (descriptive, n = {len(G)})")
ri = Fd["ref_istats"]
head = [f"[{a.tag} {a.stage.upper()}] CI level {100 * (1 - a.alpha):.2f} % | reference {'train-fit' if a.stage == 'val' else 'TEST'} "
        f"images (n = {len(REF)}): saturation {ri[:, 1].mean():.4f} HF {ri[:, 2].mean():.4f}"]
verdict = {n: bool(out[n].get("passed")) for n in names if n != base}
tail = [f"  -> {n}: {'PASS' if v else 'fail'}" for n, v in verdict.items()]
save_json({"stage": a.stage, "alpha": a.alpha, "reference_arm": base, "verdict": verdict, "results": out}, f"{a.tag}_{a.stage}.json")
txt = "\n".join(head + L + ["  Inception-v3 (torchvision, descriptive):"] + inc_lines + tail)
open(os.path.join(OUT, f"{a.tag}_{a.stage}_report.txt"), "w").write(txt)
print(txt, flush=True)
