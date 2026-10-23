# a5_bench.py - A5: compute benchmark for the Fokker-Planck (FP) training term (no real training).
#  1. unit test: the "manual recompute" gradient of the FP product loss  L = r(u,v1) r(u,v2)  equals autograd (fp64,
#     tiny network). Manual recompute = all 13 network evaluations without graph -> analytic dL/ds_k -> recompute each
#     evaluation with graph and backpropagate (dL/ds_k . s_k): exact gradient, memory of ONE evaluation.
#  2. baseline: one training micro-step of the original recipe (batch 2, 0.7 MSE + 0.2 (1-MS-SSIM) + 0.05 HF, fp16
#     autocast, GradScaler, Adam every 4 micro-steps, clip 1.0) - time and peak memory
#  3. FP term on ONE image (13 evaluations: t+-k, s0, +-h_u u, 2 probes x 4 mixed) - fp16 and fp32, recompute in groups
#     of 1 or 2 evaluations; full-graph autograd attempt at batch 1 (expected to run out of memory)
#  4. cost ratio = (T_base + T_fp / k) / T_base for "FP on one image every k micro-steps", k in {1, 2, 4, 8}
#     PRE-STATED SEED RULE (k = 4 = once per optimizer step): ratio <= 2 -> 5 seeds; <= 3 -> 4 seeds; > 3 -> reconsider
#  5. Phase B warm start: does base2_last.pt (with optimizer state) exist?
import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache, minmax_pm1, ctx_of, eps_pred, load_model, main_ckpt, find, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=int, default=100)
ap.add_argument("--k_dt", type=int, default=1)
ap.add_argument("--p", type=float, default=0.0)        # 0 = take the best setting from A4_fp.json (else 3e-3)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--base_steps", type=int, default=32)
a = ap.parse_args()
TAG = "A5"
dev = torch.device("cuda")
DTAU = 1.0 / 1000
rep = {}


# ======================= FP product loss: plan, residuals, analytic coefficients =======================
def plan(x, t, u, v1, v2, p, k):
    hu, hv = 256.0 * p, p
    E = [("tp", x, t + k), ("tm", x, t - k), ("s0", x, t), ("up", x + hu * u, t), ("um", x - hu * u, t)]
    for j, v in ((1, v1), (2, v2)):
        E += [(f"pp{j}", x + hu * u + hv * v, t), (f"pm{j}", x + hu * u - hv * v, t),
              (f"mp{j}", x - hu * u + hv * v, t), (f"mm{j}", x - hu * u - hv * v, t)]
    return E


def residuals(s, x, u, v1, v2, p, k, beta):
    hu, hv = 256.0 * p, p
    A = (u * (s["tp"] - s["tm"])).sum() / (2 * k * DTAU)
    Ju = (s["up"] - s["um"]) / (2 * hu)
    base = (u * s["s0"]).sum() + (Ju * (x + 2 * s["s0"])).sum()
    out = []
    for j, v in ((1, v1), (2, v2)):
        D2 = (v * (s[f"pp{j}"] - s[f"pm{j}"] - s[f"mp{j}"] + s[f"mm{j}"])).sum() / (4 * hu * hv)
        out.append(A - 0.5 * beta * (base + D2))
    return out[0], out[1]


def coeffs(s, x, u, v1, v2, p, k, beta, r1, r2, w=1.0):
    """dL/ds_k for L = w r1 r2 (shared evaluations: w (r1 + r2) dr/ds_k; probe-specific: w r_other dr_j/ds_k)"""
    hu, hv = 256.0 * p, p
    b = 0.5 * beta
    Ju = (s["up"] - s["um"]) / (2 * hu)
    shared = {"tp": u / (2 * k * DTAU), "tm": -u / (2 * k * DTAU), "s0": -b * (u + 2 * Ju),
              "up": -b * (x + 2 * s["s0"]) / (2 * hu), "um": b * (x + 2 * s["s0"]) / (2 * hu)}
    g = {key: w * (r1 + r2) * val for key, val in shared.items()}
    for j, v, other in ((1, v1, r2), (2, v2, r1)):
        c = b * v / (4 * hu * hv)
        g[f"pp{j}"], g[f"pm{j}"], g[f"mp{j}"], g[f"mm{j}"] = -w * other * c, w * other * c, w * other * c, -w * other * c
    return g


def evaluate_nograd(sfn, E):
    """all evaluations without graph; the ones that share a timestep are batched"""
    s = {}
    with torch.no_grad():
        for tt in sorted({e[2] for e in E}):
            keys = [e[0] for e in E if e[2] == tt]
            out = sfn(torch.cat([e[1] for e in E if e[2] == tt]), tt)
            for i, kk in enumerate(keys):
                s[kk] = out[i:i + 1]
    return s


def manual_grad(sfn, x, t, u, v1, v2, p, k, beta, group=1, w=1.0):
    E = plan(x, t, u, v1, v2, p, k)
    s = evaluate_nograd(sfn, E)
    with torch.no_grad():
        r1, r2 = residuals(s, x, u, v1, v2, p, k, beta)
        g = coeffs(s, x, u, v1, v2, p, k, beta, r1, r2, w)
    for tt in sorted({e[2] for e in E}):
        grp = [e for e in E if e[2] == tt]
        for i in range(0, len(grp), group):
            part = grp[i:i + group]
            out = sfn(torch.cat([e[1] for e in part]), tt)
            (torch.cat([g[e[0]] for e in part]) * out).sum().backward()
    return float(r1), float(r2)


def full_graph(sfn, x, t, u, v1, v2, p, k, beta, w=1.0):
    s = {kk: sfn(inp, tt) for kk, inp, tt in plan(x, t, u, v1, v2, p, k)}
    r1, r2 = residuals(s, x, u, v1, v2, p, k, beta)
    (w * r1 * r2).backward()
    return float(r1), float(r2)


# ======================= 1. unit test on a tiny fp64 network =======================
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.e = nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 1, 3, padding=1), nn.Linear(1, 8)

    def forward(self, x, t):
        tt = torch.full((x.shape[0], 1), t / 1000.0, dtype=x.dtype, device=x.device)
        h = torch.tanh(self.c1(x) + self.e(tt)[:, :, None, None])
        return self.c2(h * h)


torch.manual_seed(0)
tiny = Tiny().double().to(dev)
xs = torch.randn(1, 1, 16, 16, dtype=torch.float64, device=dev)
us = torch.randn_like(xs); us = us / us.norm()
v1s, v2s = (torch.randint(0, 2, xs.shape, device=dev) * 2 - 1).double(), (torch.randint(0, 2, xs.shape, device=dev) * 2 - 1).double()
sf_t = lambda xb, tt: tiny(xb, tt)
grads = []
for fn in (full_graph, manual_grad):
    tiny.zero_grad(set_to_none=True)
    fn(sf_t, xs, 100, us, v1s, v2s, 1e-2, 1, 1.3)
    grads.append(torch.cat([q.grad.flatten() for q in tiny.parameters()]))
ut = float((grads[0] - grads[1]).norm() / grads[0].norm())
rep["unit_test_rel_diff_manual_vs_autograd"] = ut
rep["unit_test_pass"] = bool(ut < 1e-8)
log(TAG, f"unit test: manual-recompute gradient vs autograd rel. diff = {ut:.2e} -> {'PASS' if ut < 1e-8 else 'FAIL'}")
del tiny

# ======================= real network =======================
path = main_ckpt()
unet, ac, meta = load_model(path, dev)
for q in unet.parameters():
    q.requires_grad_(True)
unet.train()
acf = ac.float()
LOGAB = torch.log(ac)
beta_t = float(-(LOGAB[a.t + 1] - LOGAB[a.t - 1]) / (2 * DTAU))
sig_t = lambda tt: math.sqrt(1.0 - float(ac[tt]))
nparam = sum(q.numel() for q in unet.parameters())
rep["unet_parameters_M"] = nparam / 1e6
log(TAG, f"UNet parameters: {nparam / 1e6:.1f} M | {torch.cuda.get_device_name(0)}")

# ---------------- 2. baseline micro-step (original recipe) ----------------
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim
_sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
KX, KY = (_sob[None, None] / 8).to(dev), (_sob.t()[None, None] / 8).to(dev)
def hf_loss(x, y):
    gx = F.conv2d(x, KX, padding=1) - F.conv2d(y, KX, padding=1)
    gy = F.conv2d(x, KY, padding=1) - F.conv2d(y, KY, padding=1)
    return (gx.pow(2) + gy.pow(2)).mean()

xtr, ytr = load_cache("train")
Xg, Yg = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).long().to(dev)
opt = torch.optim.Adam(unet.parameters(), lr=1e-5)
scaler = torch.amp.GradScaler("cuda")
gg = torch.Generator(device=dev).manual_seed(0)
state0 = {kk: v.detach().clone() for kk, v in unet.state_dict().items()}   # restored after the benchmark


def base_step(i):
    idx = torch.randint(0, Xg.shape[0], (2,), generator=gg, device=dev)
    x0 = minmax_pm1(Xg[idx].float().unsqueeze(1))
    y = Yg[idx]
    t = torch.randint(0, 1000, (2,), generator=gg, device=dev)
    eps = torch.randn(x0.shape, generator=gg, device=dev)
    drop = torch.rand(2, generator=gg, device=dev) < 0.15
    cond = torch.where(drop, torch.full_like(y, -1), y).float().view(2, 1, 1)
    ab = acf[t].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    with torch.autocast("cuda", dtype=torch.float16):
        eh = unet(xt, timesteps=t, context=cond)
    eh = eh.float()
    mse = F.mse_loss(eh, eps)
    x0h = ((xt - (1 - ab).sqrt() * eh) / ab.sqrt()).clamp(-1, 1)
    ms = 1.0 - msssim((x0h + 1) / 2, (x0 + 1) / 2, data_range=2.0, gaussian_kernel=False, normalize="relu")
    loss = 0.7 * mse + 0.2 * ms + 0.05 * hf_loss(x0h, x0)
    scaler.scale(loss / 4).backward()
    if (i + 1) % 4 == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)


for i in range(8):
    base_step(i)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for i in range(a.base_steps):
    base_step(i)
torch.cuda.synchronize()
T_base = (time.time() - t0) / a.base_steps
rep["baseline_micro_step_s"] = T_base
rep["baseline_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
rep["baseline_epoch_min_est_1240_steps"] = 1240 * T_base / 60
log(TAG, f"baseline micro-step (batch 2): {T_base:.3f} s | peak {rep['baseline_peak_mem_GB']:.1f} GB | "
         f"-> {rep['baseline_epoch_min_est_1240_steps']:.1f} min/epoch (Experiment 2 measured 12.4)")
opt.zero_grad(set_to_none=True)
del opt, scaler
unet.load_state_dict(state0)
del state0
torch.cuda.empty_cache()

# ---------------- 3. FP term on one image ----------------
a4 = None
try:
    a4 = json.load(open(os.path.join(OUT, "A4_fp.json")))
except Exception:
    pass
if a.p > 0:
    P_USE, PREC_USE, src = a.p, "fp16", "command line"
elif a4 is not None:
    ch = a4["precision_choice"]
    PREC_USE = "fp16" if ch["fp16"]["usable"] else "fp32"
    P_USE = float(ch[PREC_USE]["best_setting"].split("_p")[1])
    src = f"A4 ({a4['verdict_estimator']})"
else:
    P_USE, PREC_USE, src = 3e-3, "fp32", "default (A4_fp.json not found)"
rep["fp_setting_used"] = {"p": P_USE, "precision": PREC_USE, "source": src}
log(TAG, f"FP setting: p={P_USE:g}, {PREC_USE} ({src})")

xte, yte = load_cache("test")
x0 = minmax_pm1(torch.from_numpy(xte[:1]).float().unsqueeze(1).to(dev))
ab = float(ac[a.t])
gx = torch.Generator(device=dev).manual_seed(1)
x = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * torch.randn(x0.shape, generator=gx, device=dev)
u = torch.randn(x.shape, generator=gx, device=dev); u = u / u.norm()
v1 = (torch.randint(0, 2, x.shape, generator=gx, device=dev) * 2 - 1).float()
v2 = (torch.randint(0, 2, x.shape, generator=gx, device=dev) * 2 - 1).float()
lab = int(yte[0])


def make_sfn(prec):
    def sfn(xb, tt):
        c = ctx_of([lab] * xb.shape[0], dev)
        return -eps_pred(unet, xb, tt, c, amp=(prec == "fp16")) / sig_t(tt)
    return sfn


fp_rows = {}
for prec in ("fp16", "fp32"):
    sfn = make_sfn(prec)
    for group in (1, 2):
        key = f"{prec}_group{group}"
        try:
            unet.zero_grad(set_to_none=True)
            manual_grad(sfn, x, a.t, u, v1, v2, P_USE, a.k_dt, beta_t, group)          # warm-up
            unet.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(a.reps):
                manual_grad(sfn, x, a.t, u, v1, v2, P_USE, a.k_dt, beta_t, group)
                unet.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            fp_rows[key] = {"s": (time.time() - t0) / a.reps, "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        except torch.cuda.OutOfMemoryError:
            fp_rows[key] = {"s": None, "peak_mem_GB": None, "oom": True}
        unet.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        log(TAG, f"FP term, 13 evaluations, manual recompute {key}: {fp_rows[key]}")
rep["fp_manual_recompute"] = fp_rows

# full-graph autograd at batch 1 (expected: out of memory)
try:
    unet.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    full_graph(make_sfn("fp16"), x, a.t, u, v1, v2, P_USE, a.k_dt, beta_t)
    torch.cuda.synchronize()
    rep["fp_full_graph_fp16"] = {"s": time.time() - t0, "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
except torch.cuda.OutOfMemoryError:
    rep["fp_full_graph_fp16"] = {"oom": True}
except Exception as e:
    rep["fp_full_graph_fp16"] = {"error": repr(e)[:200]}
unet.zero_grad(set_to_none=True); torch.cuda.empty_cache()
log(TAG, f"FP term full-graph autograd (fp16, batch 1): {rep['fp_full_graph_fp16']}")

# ---------------- 4. cost ratio + seed rule ----------------
cands = [v["s"] for kk, v in fp_rows.items() if kk.startswith(PREC_USE) and v.get("s")]
if cands:
    T_fp = min(cands)
    rep["T_fp_s_used"] = T_fp
    rep["cost_ratio"] = {str(k): 1 + T_fp / (k * T_base) for k in (1, 2, 4, 8)}
    r4 = rep["cost_ratio"]["4"]
    rep["seed_rule_k4"] = "5 seeds" if r4 <= 2 else "4 seeds" if r4 <= 3 else "reconsider (cost > 3x)"
    rep["epoch_min_est"] = {str(k): rep["baseline_epoch_min_est_1240_steps"] * v for k, v in
                            ((int(k), v) for k, v in rep["cost_ratio"].items())}
    log(TAG, "cost ratio (FP on one image every k micro-steps): " +
        ", ".join(f"k={k}: {v:.2f}x" for k, v in rep["cost_ratio"].items()) + f" | seed rule at k=4: {rep['seed_rule_k4']}")
else:
    rep["seed_rule_k4"] = "not computed (FP benchmark failed)"

# ---------------- 5. Phase B warm-start file ----------------
lp = find("base2_last.pt")
if lp:
    try:
        ck = torch.load(lp[0], map_location="cpu", weights_only=False, mmap=True)
        rep["base2_last"] = {"path": lp[0], "keys": sorted(ck.keys()), "epoch": ck.get("epoch"), "step": ck.get("step"),
                             "optimizer_states": sum(1 for s in ck["opt"]["state"].values() if s) if "opt" in ck else 0,
                             "size_GB": os.path.getsize(lp[0]) / 1e9}
        del ck
    except Exception as e:
        rep["base2_last"] = {"path": lp[0], "error": repr(e)[:200]}
else:
    rep["base2_last"] = "NOT FOUND (Phase B would warm-start from base2_w_ep040.pt weights with a fresh optimizer)"
log(TAG, f"Phase B warm start file: {rep['base2_last']}")
save_json(rep, "A5_bench.json")
