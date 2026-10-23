# pinn_physics.py - physics-informed soft constraint (PINN-style residual) for mammogram DDPM.
# Physical model: signal-dependent noise-level function (NLF) of the X-ray film/digitiser chain
#   sigma^2(mu) = b + a*mu + c*mu^2
#   b: Gaussian read/digitiser noise, a*mu: Poisson (quantum) noise, c*mu^2: multiplicative film-grain/gain term
# The coefficients are CALIBRATED on real training images, then the residual between the NLF of a
# generated image and the physical NLF is added to the diffusion loss (lambda * residual).
import torch
import torch.nn.functional as F


class PhysicsNLF:
    def __init__(self, device, k=7, K=16, lo=0.08, hi=0.95, mask_lo=0.04):
        ax = torch.arange(7, device=device, dtype=torch.float32) - 3
        g = torch.exp(-ax ** 2 / 2.0)
        self.g = g / g.sum()
        self.k, self.K, self.lo, self.hi, self.mask_lo = k, K, lo, hi, mask_lo
        self.c = torch.linspace(lo, hi, K, device=device)
        self.bw = float(hi - lo) / (K - 1) * 0.6
        self.p = None                     # (b, a, c) of the calibrated NLF

    def blur(self, I):
        g = self.g
        I = F.conv2d(F.pad(I, (3, 3, 0, 0), mode="reflect"), g.view(1, 1, 1, 7))
        return F.conv2d(F.pad(I, (0, 0, 3, 3), mode="reflect"), g.view(1, 1, 7, 1))

    def stats(self, I):
        # I: (B,1,H,W) intensities in [0,1]. Returns per-image binned high-frequency variance vs local mean.
        h = I - self.blur(I)
        mu = F.avg_pool2d(I, self.k, 1, self.k // 2, count_include_pad=False).flatten(1)
        v = F.avg_pool2d(h * h, self.k, 1, self.k // 2, count_include_pad=False).flatten(1)
        W = torch.exp(-(mu.unsqueeze(-1) - self.c) ** 2 / (2 * self.bw ** 2))
        W = W * (mu > self.mask_lo).unsqueeze(-1).float()        # ignore black background (air)
        mass = W.sum(1)                                           # (B,K)
        vbin = torch.einsum("bpk,bp->bk", W, v) / (mass + 1e-6)   # (B,K)
        return vbin, mass

    def model_var(self):
        b, a, c = self.p
        return (b + a * self.c + c * self.c ** 2).clamp_min(1e-7)

    def residual(self, I):
        # per-image physics residual: mass-weighted squared log-ratio of measured vs physical NLF
        vbin, mass = self.stats(I)
        r = (torch.log(vbin + 1e-5) - torch.log(self.model_var() + 1e-5)) ** 2
        w = mass / (mass.sum(1, keepdim=True) + 1e-6)
        return (w * r).sum(1)

    @torch.no_grad()
    def fit(self, X_u8, bs=32):
        dev = self.c.device
        tv = torch.zeros(self.K, device=dev, dtype=torch.float64)
        tm = torch.zeros(self.K, device=dev, dtype=torch.float64)
        for i in range(0, X_u8.shape[0], bs):
            I = X_u8[i:i + bs].to(dev).float().div(255).unsqueeze(1)
            vb, m = self.stats(I)
            tv += (vb * m).sum(0).double()
            tm += m.sum(0).double()
        V = tv / tm.clamp_min(1e-6)
        wt = tm / tm.sum()
        c = self.c.double()

        def wls(cols):
            A = torch.stack(cols, 1)
            th = torch.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * V))
            pred = (A @ th).clamp_min(1e-9)
            ss_res = (wt * (V - pred) ** 2).sum()
            ss_tot = (wt * (V - (wt * V).sum()) ** 2).sum()
            lr = torch.log(V) - torch.log(pred)
            lres = (wt * lr ** 2).sum()
            ltot = (wt * (torch.log(V) - (wt * torch.log(V)).sum()) ** 2).sum()
            return th, float(1 - ss_res / ss_tot), float(1 - lres / ltot)

        th_lin, r2_lin, lr2_lin = wls([torch.ones_like(c), c])
        th_q, r2_q, lr2_q = wls([torch.ones_like(c), c, c ** 2])
        self.p = (float(th_q[0]), float(th_q[1]), float(th_q[2]))
        return {"model": "sigma^2 = b + a*mu + c*mu^2 (Poisson-Gaussian + multiplicative)",
                "b": self.p[0], "a": self.p[1], "c": self.p[2],
                "R2_var": r2_q, "R2_logvar": lr2_q,
                "linear_poisson_gaussian_only": {"b": float(th_lin[0]), "a": float(th_lin[1]),
                                                  "R2_var": r2_lin, "R2_logvar": lr2_lin},
                "bin_centers": [float(x) for x in c], "measured_var": [float(x) for x in V],
                "bin_weight": [float(x) for x in wt]}

    @torch.no_grad()
    def mean_residual(self, X_u8, bs=32):
        dev = self.c.device
        tot, n = 0.0, 0
        for i in range(0, X_u8.shape[0], bs):
            I = X_u8[i:i + bs].to(dev).float().div(255).unsqueeze(1)
            tot += float(self.residual(I).sum())
            n += I.shape[0]
        return tot / max(n, 1)

    def state(self):
        return {"p": list(self.p), "k": self.k, "K": self.K, "lo": self.lo, "hi": self.hi, "mask_lo": self.mask_lo}

    def load(self, s):
        self.p = tuple(s["p"])
