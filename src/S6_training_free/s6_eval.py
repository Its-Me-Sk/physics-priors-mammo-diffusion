# s6_eval.py - S6 evaluation on the GPU (features) + s6_stats.evaluate (pre-stated rule, see s6_stats.py header).
#   --stage val : reference = train-fit images (validation split never used for training decisions); writes the WINNER
#   --stage test: reference = the untouched TEST images (run ONCE on the frozen winner vs baseline, fresh noise seeds)
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_clean_meta, load_cache, val_mask, find, OUT
from a6_classifier import build, prep
from s6_stats import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["val", "test"], required=True)
ap.add_argument("--sets", required=True)            # comma list, the FIRST is the baseline
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
TAG = f"S6-eval-{a.stage}"
dev = torch.device("cuda")
names = a.sets.split(","); base = names[0]
if a.stage == "val":
    xtr, ytr, pat, _ = load_clean_meta("train"); REF = xtr[~val_mask(pat)]
else:
    REF, _ = load_cache("test")
log(TAG, f"reference: {'train-fit' if a.stage == 'val' else 'TEST'} images {REF.shape} | sets {names}")

mam = build(); mam.load_state_dict({k: v.float() for k, v in torch.load(find("a6_clf_s0.pt")[0], map_location="cpu").items()})
mam = mam.to(dev).eval()
def load_dino():
    try:
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    except Exception as e:
        log(TAG, f"torch.hub DINOv2 failed ({repr(e)[:150]}) -> timm")
        import timm
        return timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
dino = load_dino().to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(X_u8, bs=64):
    fm, fd = [], []
    for i in range(0, len(X_u8), bs):
        xb = torch.from_numpy(X_u8[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            fm.append(mam.classifier[1](mam.classifier[0](mam.avgpool(mam.features(prep(xb))))).float().cpu())
        z = F.interpolate(xb, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        fd.append(dino((z - MEAN) / STD).float().cpu())
    return torch.cat(fm).numpy().astype(np.float64), torch.cat(fd).numpy().astype(np.float64)
def istats_img(X_u8):
    I = torch.from_numpy(X_u8).float().div(255).unsqueeze(1)
    mn = I.amin(dim=(1, 2, 3), keepdim=True); mx = I.amax(dim=(1, 2, 3), keepdim=True); I = (I - mn) / (mx - mn + 1e-8)
    mu = F.avg_pool2d(I, 7, 1, 3, count_include_pad=False); m = (mu > 0.04).float(); n = m.sum((1, 2, 3)).clamp_min(1)
    hp = I - F.avg_pool2d(I, 7, 1, 3, count_include_pad=False)
    return np.stack([((I * m).sum((1, 2, 3)) / n).numpy(), ((((I > 0.95).float() * m).sum((1, 2, 3))) / n).numpy(),
                     ((hp * hp * m).sum((1, 2, 3)) / n).numpy(), m.mean((1, 2, 3)).numpy()], 1)

Fd = {}
Fd["ref_mammo"], Fd["ref_dino"] = feats(REF); Fd["ref_istats"] = istats_img(REF)
for nm in names:
    d = np.load(os.path.join(OUT, f"gen_{nm}.npz"))
    Fd[f"{nm}_mammo"], Fd[f"{nm}_dino"] = feats(d["x"]); Fd[f"{nm}_istats"] = istats_img(d["x"])
np.savez(os.path.join(OUT, f"S6_{a.stage}_features.npz"), **Fd)
L, out, win = evaluate(Fd, base, a.n_boot)
ri = Fd["ref_istats"]
head = [f"[{a.stage.upper()}] reference {'train-fit' if a.stage == 'val' else 'TEST'} images: saturation {ri[:, 1].mean():.4f} HF {ri[:, 2].mean():.4f}"]
if a.stage == "val":
    tail = [f"  -> WINNER: {win}" if win else "  -> no arm passed: no winner (test stage compares the baseline only descriptively)"]
    save_json({"winner": win, "results": out}, "S6_decision.json")
else:
    w = [n for n in names if n != base]
    ok = bool(w and out[w[0]].get("passed"))
    tail = [f"  -> TEST CONFIRMATION of {w[0] if w else '-'}: {'CONFIRMED' if ok else 'NOT CONFIRMED'}"]
    save_json({"confirmed": ok, "results": out}, "S6_test.json")
txt = "\n".join(head + L + tail)
open(os.path.join(OUT, f"S6_{a.stage}_report.txt"), "w").write(txt)
print(txt, flush=True)
