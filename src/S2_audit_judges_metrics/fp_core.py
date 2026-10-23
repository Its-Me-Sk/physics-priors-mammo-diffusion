# fp_core.py - the Fokker-Planck (score-FPE) residual and its exact "manual recompute" gradient.
# Shared by the Phase A-2 diagnostics and (later) the Phase B/C training, so both use byte-identical maths.
# VP-SDE of the DDPM (tau = t/T, beta(tau) = -d log abar / d tau), score s = -eps_hat / sqrt(1 - abar_t):
#     d_tau s = 1/2 beta grad_x( x.s + div s + |s|^2 )
# projected on a unit direction u, Hutchinson probe v (Rademacher) for d_u(div s):
#     r(u,v) = u.d_tau s - 1/2 beta [ u.s + (J_s u).(x + 2 s) + v.(d_u J_s) v ]
# all derivatives by central finite differences (per-pixel step p; h_u = 256 p for unit u on 256x256, h_v = p);
# d_tau s by +-k integer steps. Product of two independent probes = unbiased estimate of (u.R)^2.
# Verified in Phase A: unit test vs autograd (1e-13), analytic Gaussian score -> residual ~0, wrong score -> O(1).
import math
import torch

T_STEPS = 1000
DTAU = 1.0 / T_STEPS


def p_of_t(t):
    """finite-difference step per t (fixed before Phase B from the A4 accuracy table: <= 1 % error vs the exact JVP)"""
    return 1e-3 if t <= 20 else (3e-3 if t <= 50 else 1e-2)


def beta_eff(logab, t):
    return float(-(logab[t + 1] - logab[t - 1]) / (2 * DTAU))


def plan(x, t, u, v1, v2, p, k=1):
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
    return out[0], out[1], A


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
    s = {}
    with torch.no_grad():
        for tt in sorted({e[2] for e in E}):
            keys = [e[0] for e in E if e[2] == tt]
            out = sfn(torch.cat([e[1] for e in E if e[2] == tt]), tt)
            for i, kk in enumerate(keys):
                s[kk] = out[i:i + 1]
    return s


def fp_loss_and_grad(sfn, x, t, u, v1, v2, p, k, beta, w=1.0, group=1, backward=True):
    """exact gradient of L = w r1 r2 accumulated into .grad (memory of ONE network evaluation). Returns (L, r1, r2, lhs)."""
    E = plan(x, t, u, v1, v2, p, k)
    s = evaluate_nograd(sfn, E)
    with torch.no_grad():
        r1, r2, A = residuals(s, x, u, v1, v2, p, k, beta)
        g = coeffs(s, x, u, v1, v2, p, k, beta, r1, r2, w)
    if backward:
        for tt in sorted({e[2] for e in E}):
            grp = [e for e in E if e[2] == tt]
            for i in range(0, len(grp), group):
                part = grp[i:i + group]
                out = sfn(torch.cat([e[1] for e in part]), tt)
                (torch.cat([g[e[0]] for e in part]) * out).sum().backward()
    return float(w * r1 * r2), float(r1), float(r2), float(A)


def plan_n(x, t, u, vs, p, k=1):
    """same as plan() but for any number of probes vs (list of tensors)"""
    hu, hv = 256.0 * p, p
    E = [("tp", x, t + k), ("tm", x, t - k), ("s0", x, t), ("up", x + hu * u, t), ("um", x - hu * u, t)]
    for j, v in enumerate(vs):
        E += [(f"pp{j}", x + hu * u + hv * v, t), (f"pm{j}", x + hu * u - hv * v, t),
              (f"mp{j}", x - hu * u + hv * v, t), (f"mm{j}", x - hu * u - hv * v, t)]
    return E


def residuals_n(s, x, u, vs, p, k, beta):
    """r(u, v_j) for every probe j (same formula as residuals()); returns (list of r_j, lhs, terms dict)"""
    hu, hv = 256.0 * p, p
    A = (u * (s["tp"] - s["tm"])).sum() / (2 * k * DTAU)
    Ju = (s["up"] - s["um"]) / (2 * hu)
    us, Jt = (u * s["s0"]).sum(), (Ju * (x + 2 * s["s0"])).sum()
    rs, d2s = [], []
    for j, v in enumerate(vs):
        D2 = (v * (s[f"pp{j}"] - s[f"pm{j}"] - s[f"mp{j}"] + s[f"mm{j}"])).sum() / (4 * hu * hv)
        rs.append(A - 0.5 * beta * (us + Jt + D2)); d2s.append(D2)
    return rs, A, {"us": 0.5 * beta * us, "Jt": 0.5 * beta * Jt, "D2": [0.5 * beta * d for d in d2s]}


def make_score_fn(unet, ac, ctx_value, amp=False, chunk=4):
    """score function s(x, t) of the network for one class label (context value); fp32 by default (A4: fp16 unusable).
    Batches larger than `chunk` are split (non-flash attention at 64x64 needs ~0.5 GB per image in fp32)."""
    def sfn(xb, tt):
        if xb.shape[0] > chunk:
            return torch.cat([sfn(xb[i:i + chunk], tt) for i in range(0, xb.shape[0], chunk)])
        c = torch.full((xb.shape[0], 1, 1), float(ctx_value), device=xb.device, dtype=xb.dtype)
        tb = torch.full((xb.shape[0],), int(tt), device=xb.device, dtype=torch.long)
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                e = unet(xb, timesteps=tb, context=c).float()
        else:
            e = unet(xb, timesteps=tb, context=c)
        return -e / math.sqrt(1.0 - float(ac[tt]))
    return sfn


def sref_interp(table, t):
    """log-linear interpolation of the reference scale table {t: RMS(u.d_tau s)} used to normalise the loss"""
    ts = sorted(int(k) for k in table)
    t = min(max(t, ts[0]), ts[-1])
    for a, b in zip(ts[:-1], ts[1:]):
        if a <= t <= b:
            la, lb = math.log(table[str(a)] if str(a) in table else table[a]), math.log(table[str(b)] if str(b) in table else table[b])
            return math.exp(la + (lb - la) * (t - a) / max(b - a, 1))
    return table[str(ts[-1])] if str(ts[-1]) in table else table[ts[-1]]
