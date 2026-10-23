# a4_fp.py - A4: validation of the Fokker-Planck (score-FPE) residual estimator on the REAL trained network.
# No training. Network: Experiment-2 full-data baseline (epoch 40). Score s(x,t) = -eps_hat(x,t) / sqrt(1 - abar_t).
# Score FPE of the VP-SDE (tau = t/1000, beta(tau) = -d log abar / d tau):
#     d_tau s = 1/2 beta grad_x( x.s + div s + |s|^2 )
# Projected on a unit direction u (no symmetry of J_s needed):
#     r(u) = u.d_tau s - 1/2 beta [ u.s + (J_s u).(x + 2 s) + d_u(div s) ],   d_u(div s) ~ v.(d_u J_s) v  (Hutchinson)
# Estimators compared (same u, same Rademacher v):
#   REF  : exact directional derivatives by forward-mode AD (torch.func.jvp, nested for d_u div s), fp32
#          (fallback if jvp is not supported by the network: central finite differences in fp64, p = 1e-4)
#   FD   : central finite differences with per-pixel step p (h_u = 256 p for unit u, h_v = p for Rademacher v),
#          in fp32 and in fp16 autocast (= training precision), p in {1e-3, 3e-3, 1e-2}
#   d_tau s always by central differences in the integer timestep (+-k steps, k in {1, 2, 5} reported)
#   product estimator r(u,v1) r(u,v2) (independent v) = unbiased estimate of (u.R)^2
#   guided score (CFG scale 4, as used for sampling): FD fp32, p = 3e-3, 2 probes
# PRE-STATED RULES:
#   estimator usable at precision P : for the best p, rel.error = RMS(r_P - r_REF)/RMS(r_REF) <= 0.10 at every grid
#                                     t in [10, 300]
#   FP residual measurable at t      : z = mean(product)/SE(product) > 3 (REF, all images x directions)
import os, sys, math, time, copy, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phaseA_common import log, save_json, load_cache, to_model_domain, ctx_of, eps_pred, load_model, main_ckpt, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--n_img", type=int, default=6)
ap.add_argument("--n_u", type=int, default=3)
ap.add_argument("--n_v", type=int, default=4)
ap.add_argument("--grid", default="10,20,50,100,200,300,500,700")
ap.add_argument("--ps", default="1e-3,3e-3,1e-2")
ap.add_argument("--dts", default="1,2,5")
ap.add_argument("--gs", type=float, default=4.0)
ap.add_argument("--p_guided", type=float, default=3e-3)
ap.add_argument("--fp64_check_t", default="20,100")
ap.add_argument("--fp64_p", type=float, default=1e-4)
a = ap.parse_args()
TAG = "A4"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
path = main_ckpt()
unet, ac, meta = load_model(path, dev)                      # fp32 weights (fp16 mode = autocast, as in training)
log(TAG, f"checkpoint {path} | epoch {meta.get('epoch')}")
T = ac.numel()
DTAU = 1.0 / T
LOGAB = torch.log(ac)
GRID = [int(g) for g in a.grid.split(",")]
PS = [float(p) for p in a.ps.split(",")]
DTS = [int(k) for k in a.dts.split(",")]
assert max(GRID) + max(DTS) < T and min(GRID) - max(DTS) >= 0
NETS = {"fp32": unet, "fp16": unet}
CHUNK = {"fp16": 8, "fp32": 6, "fp64": 2}


def beta_eff(t):
    return float(-(LOGAB[t + 1] - LOGAB[t - 1]) / (2 * DTAU))


def sig(t):
    return math.sqrt(1.0 - float(ac[t]))


def dot(p, q):
    return (p * q).flatten(1).sum(1)


def S(xb, t, lab, mode, gs=None, chunk=None):
    """score of the network at integer step t; mode fp32 | fp16 (autocast) | fp64 (double copy)"""
    net = NETS[mode]
    dt = torch.float64 if mode == "fp64" else torch.float32
    xb = xb.to(dt)
    B = xb.shape[0]
    ch = chunk or CHUNK[mode]
    e = eps_pred(net, xb, t, ctx_of([lab] * B, dev, dt), amp=(mode == "fp16"), chunk=ch).to(dt)
    if gs is not None:
        eu = eps_pred(net, xb, t, ctx_of([-1] * B, dev, dt), amp=(mode == "fp16"), chunk=ch).to(dt)
        e = eu + gs * (e - eu)
    return -e / sig(t)


@torch.no_grad()
def lhs(x, t, lab, mode, U, k=1, gs=None):
    d = (S(x, t + k, lab, mode, gs) - S(x, t - k, lab, mode, gs)) / (2 * k * DTAU)
    return dot(U.to(d.dtype), d.expand(U.shape[0], -1, -1, -1))


@torch.no_grad()
def terms_fd(x, t, lab, U, V, p, mode, gs=None, k=1):
    dt = torch.float64 if mode == "fp64" else torch.float32
    x, U, V = x.to(dt), U.to(dt), V.to(dt)
    nu, nv = U.shape[0], V.shape[0]
    hu, hv = 256.0 * p, p
    s0 = S(x, t, lab, mode, gs)
    L = lhs(x, t, lab, mode, U, k, gs)
    us = dot(U, s0.expand(nu, -1, -1, -1))
    Xb = x.expand(nu, -1, -1, -1)
    sj = S(torch.cat([Xb + hu * U, Xb - hu * U]), t, lab, mode, gs)
    Ju = (sj[:nu] - sj[nu:]) / (2 * hu)
    Jt = dot(Ju, (x + 2 * s0).expand(nu, -1, -1, -1))
    Up = U[:, None].expand(nu, nv, *U.shape[1:]).reshape(nu * nv, *U.shape[1:])
    Vp = V[None].expand(nu, nv, *V.shape[1:]).reshape(nu * nv, *V.shape[1:])
    Xm = x.expand(nu * nv, -1, -1, -1)
    n = nu * nv
    D2 = torch.zeros(n, dtype=dt, device=dev)
    for sgn_u, sgn_v, c in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):    # mixed second difference
        D2 += c * dot(Vp, S(Xm + sgn_u * hu * Up + sgn_v * hv * Vp, t, lab, mode, gs))
    D2 = (D2 / (4 * hu * hv)).view(nu, nv)
    return {"L": L, "us": us, "Jt": Jt, "D2": D2, "beta": beta_eff(t)}


def terms_jvp(x, t, lab, U, V, k=1):
    from torch.func import jvp
    nu, nv = U.shape[0], V.shape[0]
    with torch.no_grad():
        s0 = S(x, t, lab, "fp32")
        L = lhs(x, t, lab, "fp32", U, k)
    us = dot(U, s0.expand(nu, -1, -1, -1))
    Xb = x.expand(nu, -1, -1, -1).contiguous()
    _, Ju = jvp(lambda z: S(z, t, lab, "fp32", chunk=3), (Xb,), (U.contiguous(),))
    Jt = dot(Ju, (x + 2 * s0).expand(nu, -1, -1, -1))
    D2 = torch.zeros(nu, nv, device=dev)
    for j in range(nv):
        vj = V[j:j + 1].expand(nu, -1, -1, -1).contiguous()

        def g(z, vj=vj):
            _, jv = jvp(lambda w: S(w, t, lab, "fp32", chunk=1), (z,), (vj,))
            return dot(vj, jv)

        _, d = jvp(g, (Xb,), (U.contiguous(),))
        D2[:, j] = d.detach()
    return {"L": L, "us": us.detach(), "Jt": Jt.detach(), "D2": D2, "beta": beta_eff(t)}


def resid(d):
    return (d["L"][:, None] - 0.5 * d["beta"] * (d["us"][:, None] + d["Jt"][:, None] + d["D2"])).double().cpu().numpy()


def np_terms(d):
    b = 0.5 * d["beta"]
    f = lambda z: z.double().cpu().numpy()
    return {"L": f(d["L"]), "us": b * f(d["us"]), "Jt": b * f(d["Jt"]), "D2": b * f(d["D2"])}


def rel(p, q):
    return float(np.sqrt(np.mean((p - q) ** 2)) / (np.sqrt(np.mean(q ** 2)) + 1e-300))


def prod_stats(r):
    """r: (N, n_u, n_v) -> product (U-statistic over probe pairs) = unbiased (u.R)^2 per (image, u)"""
    V_ = r.shape[-1]
    s1, s2 = r.sum(-1), (r ** 2).sum(-1)
    P = ((s1 ** 2 - s2) / (V_ * (V_ - 1))).ravel()
    m, se = float(P.mean()), float(P.std(ddof=1) / np.sqrt(len(P)))
    return {"mean_sq_residual": m, "se": se, "z": m / (se + 1e-300), "rms_residual": float(np.sqrt(max(m, 0.0))),
            "single_probe_var": float(r.var(-1, ddof=1).mean())}


# ---------------- data: n_img real test images (half benign / half malignant), fixed noise ----------------
xte, yte = load_cache("test")
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(yte == c)[0], a.n_img // 2, replace=False) for c in (0, 1)])
X0 = to_model_domain(xte[idx], dev)
LAB = [int(v) for v in yte[idx]]
gen = torch.Generator(device=dev).manual_seed(7)
H, W = X0.shape[-2:]
U = torch.randn((a.n_u, 1, H, W), generator=gen, device=dev)
U = U / U.flatten(1).norm(dim=1).view(-1, 1, 1, 1)                        # unit directions (per-pixel RMS 1/256)
V = (torch.randint(0, 2, (a.n_v, 1, H, W), generator=gen, device=dev) * 2 - 1).float()   # Rademacher probes
NOISE = torch.randn(X0.shape, generator=gen, device=dev)

# ---------------- reference method: JVP (fallback FD fp64) ----------------
REF = "jvp"
try:
    _ = terms_jvp(X0[:1] * 0 + 0.1, GRID[0], LAB[0], U, V[:1])        # full direction batch -> also a memory test
    log(TAG, "forward-mode AD (torch.func.jvp, nested) works on this network -> reference = JVP fp32 | peak "
             f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
except Exception as e:
    REF = "fd64"
    log(TAG, f"jvp NOT usable ({repr(e)[:200]}) -> reference = central FD in fp64, p = {a.fp64_p}, on 2 images")
    NETS["fp64"] = copy.deepcopy(unet).double()
    keep = [0, X0.shape[0] - 1]                                           # one benign, one malignant
    X0, NOISE, LAB = X0[keep], NOISE[keep], [LAB[k] for k in keep]
torch.cuda.empty_cache()

methods = [(m, p) for m in ("fp32", "fp16") for p in PS]
rows = []
raw = {}
t_all = time.time()
for t in GRID:
    t1 = time.time()
    ab = float(ac[t])
    Xt = math.sqrt(ab) * X0 + math.sqrt(1 - ab) * NOISE
    R_ref, R_m, terms_ref, terms_m, dts_rows, R_g, R_ug = [], {mp: [] for mp in methods}, [], {mp: [] for mp in methods}, [], [], []
    for i in range(Xt.shape[0]):
        x, lab = Xt[i:i + 1], LAB[i]
        d_ref = terms_jvp(x, t, lab, U, V) if REF == "jvp" else terms_fd(x, t, lab, U, V, a.fp64_p, "fp64")
        R_ref.append(resid(d_ref)); terms_ref.append(np_terms(d_ref))
        # time-derivative step sensitivity (reference spatial terms, fp32 time differences)
        Lk = {k: lhs(x, t, lab, "fp32", U, k).double().cpu().numpy() for k in DTS}
        dts_rows.append(Lk)
        for mp in methods:
            d = terms_fd(x, t, lab, U, V, mp[1], mp[0])
            R_m[mp].append(resid(d)); terms_m[mp].append(np_terms(d))
        dg = terms_fd(x, t, lab, U, V[:2], a.p_guided, "fp32", gs=a.gs)
        du = terms_fd(x, t, lab, U, V[:2], a.p_guided, "fp32")
        R_g.append(resid(dg)); R_ug.append(resid(du))
    R_ref = np.stack(R_ref)
    st = prod_stats(R_ref)
    TR = {k: np.stack([d[k] for d in terms_ref]) for k in ("L", "us", "Jt", "D2")}
    lhs_rms = float(np.sqrt(np.mean(TR["L"] ** 2)))
    row = {"t": t, "abar": ab, "beta_eff": beta_eff(t), "reference": REF, **st,
           "lhs_rms": lhs_rms, "relative_residual_rho": st["rms_residual"] / (lhs_rms + 1e-300),
           "term_rms": {k: float(np.sqrt(np.mean(v ** 2))) for k, v in TR.items()},
           "dtau_step_rel_change_vs_k1": {}, "methods": {}}
    L1 = np.stack([d[DTS[0]] for d in dts_rows])
    for k in DTS[1:]:
        row["dtau_step_rel_change_vs_k1"][str(k)] = rel(np.stack([d[k] for d in dts_rows]), L1)
    for mp in methods:
        Rm = np.stack(R_m[mp])
        TM = {k: np.stack([d[k] for d in terms_m[mp]]) for k in ("L", "us", "Jt", "D2")}
        row["methods"][f"{mp[0]}_p{mp[1]:g}"] = {
            "rel_err_r": rel(Rm, R_ref), "rel_err_Jterm": rel(TM["Jt"], TR["Jt"]), "rel_err_D2term": rel(TM["D2"], TR["D2"]),
            "rel_err_lhs": rel(TM["L"], TR["L"]), "z_product": prod_stats(Rm)["z"]}
        raw[f"r_{mp[0]}_p{mp[1]:g}_t{t}"] = Rm
    Rg, Rug = np.stack(R_g), np.stack(R_ug)
    Pg, Pu = (Rg[..., 0] * Rg[..., 1]).ravel(), (Rug[..., 0] * Rug[..., 1]).ravel()
    row["guided"] = {"gs": a.gs, "p": a.p_guided, "mean_sq_residual_guided": float(Pg.mean()),
                     "mean_sq_residual_unguided_same_setting": float(Pu.mean()),
                     "rms_ratio_guided_over_unguided": float(np.sqrt(max(Pg.mean(), 0)) / (np.sqrt(max(Pu.mean(), 0)) + 1e-300)),
                     "z_guided": float(Pg.mean() / (Pg.std(ddof=1) / np.sqrt(len(Pg)) + 1e-300))}
    raw[f"r_ref_t{t}"] = R_ref
    rows.append(row)
    save_json({"partial": True, "reference": REF, "rows": rows}, "A4_fp_partial.json")
    best16 = min((v["rel_err_r"], k) for k, v in row["methods"].items() if k.startswith("fp16"))
    best32 = min((v["rel_err_r"], k) for k, v in row["methods"].items() if k.startswith("fp32"))
    log(TAG, f"t={t:4d} | z={st['z']:6.2f} rho={row['relative_residual_rho']:.3f} | best fp32 {best32[1]} err {best32[0]:.3f}"
             f" | best fp16 {best16[1]} err {best16[0]:.3f} | guided/unguided RMS "
             f"{row['guided']['rms_ratio_guided_over_unguided']:.2f} | {time.time() - t1:.0f} s")

# ---------------- fp64 cross-check of the JVP reference (1 image, 1 direction, 2 probes) ----------------
check = []
if REF == "jvp":
    try:
        NETS["fp64"] = copy.deepcopy(unet).double()
        for t in [int(v) for v in a.fp64_check_t.split(",") if v]:
            ab = float(ac[t])
            x = math.sqrt(ab) * X0[:1] + math.sqrt(1 - ab) * NOISE[:1]
            rj = resid(terms_jvp(x, t, LAB[0], U[:1], V[:2]))
            rf = resid(terms_fd(x, t, LAB[0], U[:1], V[:2], a.fp64_p, "fp64"))
            check.append({"t": t, "rel_diff_jvp_vs_fd64": rel(rj, rf)})
            log(TAG, f"fp64 cross-check t={t}: rel diff JVP vs FD64 = {check[-1]['rel_diff_jvp_vs_fd64']:.2e}")
        del NETS["fp64"]
    except Exception as e:
        check.append({"error": repr(e)[:300]})

# ---------------- verdicts ----------------
inwin = [r for r in rows if 10 <= r["t"] <= 300]
def worst(key):
    return max(r["methods"][key]["rel_err_r"] for r in inwin)
choice = {}
for prec in ("fp16", "fp32"):
    keys = [k for k in rows[0]["methods"] if k.startswith(prec)]
    wk = {k: worst(k) for k in keys}
    kb = min(wk, key=wk.get)
    choice[prec] = {"best_setting": kb, "worst_rel_err_t10_300": wk[kb], "usable": bool(wk[kb] <= 0.10),
                    "all_settings_worst_rel_err": wk}
meas = [r["t"] for r in rows if r["z"] > 3]
res = {"checkpoint": path, "epoch": meta.get("epoch"), "reference": REF, "n_img": int(X0.shape[0]), "n_u": a.n_u,
       "n_v": a.n_v, "grid": GRID, "rows": rows, "fp64_crosscheck": check, "precision_choice": choice,
       "t_with_z_gt_3": meas,
       "fp_residual_measurable_all_t_10_300": bool(all(r["z"] > 3 for r in inwin)),
       "verdict_estimator": ("fp16 FD usable (" + choice["fp16"]["best_setting"] + ")") if choice["fp16"]["usable"] else
                            ("fp16 NOT usable; fp32 FD usable (" + choice["fp32"]["best_setting"] + ")") if choice["fp32"]["usable"] else
                            "neither fp16 nor fp32 FD within 10 % of the reference -> FD estimator not usable as is",
       "minutes": (time.time() - t_all) / 60}
save_json(res, "A4_fp.json")
np.savez_compressed(os.path.join(OUT, "A4_raw.npz"), **raw)
log(TAG, f"estimator verdict: {res['verdict_estimator']}")
log(TAG, f"FP residual measurable (z>3) at t = {meas} | all t in [10,300]: {res['fp_residual_measurable_all_t_10_300']}")
log(TAG, f"done in {res['minutes']:.1f} min")
