# imputers.py
# =========================================
# Unified API for VAE/MIWAE/H-VAE/HMC-VAE/HH-VAEM
# =========================================
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
class StandardScalerPerFeature:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        obs = ~np.isnan(X)
        means = np.zeros(X.shape[1], dtype=np.float32)
        stds = np.ones(X.shape[1], dtype=np.float32)
        for j in range(X.shape[1]):
            col = X[obs[:, j], j]
            means[j] = col.mean() if col.size else 0.0
            s = col.std(ddof=0) if col.size else 1.0
            stds[j] = s if s > 1e-6 else 1.0
        self.mean_, self.scale_ = means, stds
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X * self.scale_ + self.mean_


def to_torch(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def masked_mse(recon, target, mask, eps=1.0):
    # recon, target: [B, D], mask: [B, D] (1 for observed)
    diff = (recon - target) * mask
    per_row = (diff * diff).sum(-1) / (mask.sum(-1).clamp_min(eps))
    return per_row.mean()


def kl_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


# -----------------------------
# Simple MLP blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Single-layer VAE Core
# -----------------------------
class VAECore(nn.Module):
    def __init__(self, x_dim, z_dim=32, hidden=(256, 256)):
        super().__init__()
        self.enc = MLP(x_dim, 2 * z_dim, hidden)
        self.dec = MLP(z_dim, x_dim, hidden)
        self.z_dim = z_dim

    def encode(self, x):
        stats = self.enc(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * (0.5 * logvar).exp()

    def decode(self, z):
        return self.dec(z)


# -----------------------------
# Two-layer Hierarchical VAE Core (H-VAE)
# z2 -> z1 -> x ; both Gaussian
# -----------------------------
class HVAECore(nn.Module):
    def __init__(self, x_dim, z1_dim=32, z2_dim=16, hidden=(256, 256)):
        super().__init__()
        self.enc = MLP(x_dim, 2 * z1_dim, hidden)          # q(z1|x)
        self.enc_top = MLP(z1_dim, 2 * z2_dim, hidden)     # q(z2|z1)
        self.dec_top_mu = MLP(z2_dim, z1_dim, hidden)      # p(z1|z2): mean
        self.dec_top_logvar = MLP(z2_dim, z1_dim, hidden)  # p(z1|z2): logvar
        self.dec = MLP(z1_dim, x_dim, hidden)              # p(x|z1): mean
        self.z1_dim, self.z2_dim = z1_dim, z2_dim

    def encode(self, x):
        mu1, logvar1 = self.enc(x).chunk(2, dim=-1)           # q(z1|x)
        z1 = mu1 + torch.randn_like(mu1) * (0.5 * logvar1).exp()
        mu2, logvar2 = self.enc_top(z1).chunk(2, dim=-1)      # q(z2|z1)
        return (mu1, logvar1), (mu2, logvar2)

    def prior_z2(self, bsz, device):
        return torch.zeros(bsz, self.z2_dim, device=device), torch.zeros(bsz, self.z2_dim, device=device)

    def decode(self, z2=None, z1=None):
        if z1 is None:
            # sample z1 ~ p(z1|z2)
            mu1_p = self.dec_top_mu(z2)
            logvar1_p = self.dec_top_logvar(z2)
            z1 = mu1_p + torch.randn_like(mu1_p) * (0.5 * logvar1_p).exp()
        x_mu = self.dec(z1)
        return x_mu

    def kl_terms(self, x):
        # returns z1, z2 samples and KL(q(z1|x)||p(z1|z2)) + KL(q(z2|z1)||p(z2))
        (mu1, logvar1), (mu2, logvar2) = self.encode(x)
        z1 = mu1 + torch.randn_like(mu1) * (0.5 * logvar1).exp()
        z2 = mu2 + torch.randn_like(mu2) * (0.5 * logvar2).exp()
        # p(z2)=N(0,I)
        kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)
        # p(z1|z2)=N(mu1_p, Sigma1_p)
        mu1_p = self.dec_top_mu(z2)
        logvar1_p = self.dec_top_logvar(z2)
        # KL(q||p) for diagonal Gaussians
        kl1 = 0.5 * torch.sum(
            logvar1_p - logvar1
            + (logvar1.exp() + (mu1 - mu1_p).pow(2)) / logvar1_p.exp()
            - 1.0,
            dim=-1
        )
        return z1, z2, kl1.mean() + kl2.mean()


# -----------------------------
# Minimal HMC sampler for z
# target: log p(z|x) ∝ log p(x|z) + log p(z)
# We approximate p(x|z) with Gaussian (MSE), and p(z)=N(0,I)
# -----------------------------
@torch.no_grad()
def hmc_sample_z(init_z, x, mask, decoder, step_size=0.05, leapfrog_steps=5, chains=1, noise_std=1.0):
    """
    init_z: [B, Z]
    x, mask: [B, D]
    decoder: z -> x_mu
    returns: z' [B, Z]
    """
    def log_prob(z):
        x_mu = decoder(z)
        # Gaussian likelihood with fixed noise_std
        # log p(x|z) on observed dims
        diff = (x_mu - x) * mask
        ll = -0.5 / (noise_std ** 2) * (diff * diff).sum(-1)
        # log p(z)
        lpz = -0.5 * (z * z).sum(-1)
        return ll + lpz

    def grad_log_prob(z):
        # 在 no_grad 环境内重新启用梯度跟踪
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)   # 先 detach 再开启 requires_grad
            val = log_prob(z).sum()
            g, = torch.autograd.grad(val, z, create_graph=False, retain_graph=False)
        return g

    z = init_z.clone()
    B, Z = z.shape
    # simple single-chain per batch element
    for _ in range(chains):
        r = torch.randn_like(z)
        z_new, r_new = z.clone(), r.clone()
        # Leapfrog
        g = grad_log_prob(z_new)
        r_new = r_new + 0.5 * step_size * g
        for _lf in range(leapfrog_steps):
            z_new = z_new + step_size * r_new
            g = grad_log_prob(z_new)
            if _lf != leapfrog_steps - 1:
                r_new = r_new + step_size * g
        r_new = r_new + 0.5 * step_size * g
        # Metropolis-Hastings acceptance
        def H(zv, rv):
            return -log_prob(zv) + 0.5 * (rv * rv).sum(-1)
        curr_H = H(z, r)
        prop_H = H(z_new, r_new)
        accept = (torch.rand(B, device=z.device) < torch.exp(curr_H - prop_H).clamp(max=50.0)).float().unsqueeze(-1)
        z = accept * z_new + (1 - accept) * z
    return z


# -----------------------------
# Base Imputer
# -----------------------------
class _BaseImputer:
    def __init__(self, lr=1e-3, batch_size=256, epochs=200, device=None, seed=42, verbose=True):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScalerPerFeature()
        self.model = None

    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float32)
        mask = ~np.isnan(X)
        X0 = X.copy()
        col_means = np.nanmean(X0, axis=0)
        inds = np.where(~mask)
        X0[inds] = col_means[inds[1]].astype(np.float32)
        return X, X0, mask.astype(np.float32)

    def fit(self, X):
        raise NotImplementedError

    @torch.no_grad()
    def transform(self, X):
        raise NotImplementedError

    def sample_imputations(self, X, n_samples=5):
        raise NotImplementedError


# -----------------------------
# 1) Vanilla VAE  (also acts as simplified "VAEM")
# -----------------------------
class VanillaVAE(_BaseImputer):
    def __init__(self, z_dim=32, hidden=(256, 256), beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.z_dim, self.hidden, self.beta = z_dim, hidden, beta

    def fit(self, X):
        X_raw, X_init, M = self._prepare(X)
        self.scaler.fit(X_raw)
        Xn = self.scaler.transform(X_init)

        x_dim = Xn.shape[1]
        self.model = VAECore(x_dim, self.z_dim, self.hidden).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        n = Xn.shape[0]
        idx = np.arange(n)

        for ep in range(1, self.epochs + 1):
            self.rng.shuffle(idx)
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                xb, mb = Xn_t[b], Mn_t[b]
                mu, logvar = self.model.encode(xb)
                z = self.model.reparam(mu, logvar)
                xmu = self.model.decode(z)
                rec = masked_mse(xmu, xb, mb)
                kl = kl_normal(mu, logvar)
                loss = rec + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
            if self.verbose and (ep == 1 or ep % max(1, self.epochs // 20) == 0):
                print(f"[VAE] epoch {ep:4d} rec={rec.item():.4f} kl={kl.item():.4f}")
        return self

    @torch.no_grad()
    def transform(self, X, n_samples=8):
        X_raw, X_init, M = self._prepare(X)
        Xn = self.scaler.transform(X_init)
        Xn_t = to_torch(Xn, self.device)
        outs = []
        self.model.eval()
        for _ in range(n_samples):
            mu, logvar = self.model.encode(Xn_t)
            z = self.model.reparam(mu, logvar)
            xmu = self.model.decode(z).cpu().numpy()
            outs.append(xmu)
        rec_mean = np.mean(outs, axis=0)
        miss = (~np.isnan(X_raw))
        Xn_f = Xn.copy()
        Xn_f[~miss] = rec_mean[~miss]
        return self.scaler.inverse_transform(Xn_f)

    def sample_imputations(self, X, n_samples=5):
        X_raw, X_init, M = self._prepare(X)
        Xn = self.scaler.transform(X_init)
        Xn_t = to_torch(Xn, self.device)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, logvar = self.model.encode(Xn_t)
                z = self.model.reparam(mu, logvar)
                xmu = self.model.decode(z).cpu().numpy()
                Xn_s = Xn.copy()
                miss = np.isnan(X_raw)
                Xn_s[miss] = xmu[miss]
                outs.append(self.scaler.inverse_transform(Xn_s))
        return outs


# -----------------------------
# 2) MIWAE (IWAE objective; K>1)
# -----------------------------
class MIWAE(VanillaVAE):
    def __init__(self, K=5, **kwargs):
        super().__init__(**kwargs)
        self.K = K

    def fit(self, X):
        X_raw, X_init, M = self._prepare(X)
        self.scaler.fit(X_raw)
        Xn = self.scaler.transform(X_init)

        x_dim = Xn.shape[1]
        self.model = VAECore(x_dim, self.z_dim, self.hidden).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        n = Xn.shape[0]; idx = np.arange(n)

        for ep in range(1, self.epochs + 1):
            self.rng.shuffle(idx)
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                xb, mb = Xn_t[b], Mn_t[b]
                mu, logvar = self.model.encode(xb)   # [B, z]
                # importance samples
                zs = []
                log_ws = []
                for _ in range(self.K):
                    eps = torch.randn_like(mu)
                    z = mu + eps * (0.5 * logvar).exp()
                    xmu = self.model.decode(z)
                    # log p(x|z) on observed dims (Gaussian with unit variance for simplicity)
                    diff = (xmu - xb) * mb
                    log_pxz = -0.5 * (diff * diff).sum(-1)
                    # log p(z)
                    log_pz = -0.5 * (z * z).sum(-1)
                    # log q(z|x)
                    log_q = -0.5 * (((z - mu) ** 2) / logvar.exp() + logvar).sum(-1)
                    log_w = log_pxz + log_pz - log_q
                    log_ws.append(log_w.unsqueeze(-1))
                log_ws = torch.cat(log_ws, dim=-1)          # [B, K]
                # IWAE bound: -E[log(1/K sum exp(log_w))]
                m, _ = torch.max(log_ws, dim=-1, keepdim=True)
                iw = m.squeeze(-1) + torch.log(torch.mean(torch.exp(log_ws - m), dim=-1) + 1e-9)
                loss = -iw.mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if self.verbose and (ep == 1 or ep % max(1, self.epochs // 20) == 0):
                print(f"[MIWAE] epoch {ep:4d} loss={loss.item():.4f}")
        return self


# -----------------------------
# 3) H-VAE (hierarchical Gaussian; 2-layer)
# -----------------------------
class H_VAEM(_BaseImputer):
    def __init__(self, z1_dim=32, z2_dim=16, hidden=(256, 256), beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.z1_dim, self.z2_dim, self.hidden, self.beta = z1_dim, z2_dim, hidden, beta

    def fit(self, X):
        X_raw, X_init, M = self._prepare(X)
        self.scaler.fit(X_raw); Xn = self.scaler.transform(X_init)
        x_dim = Xn.shape[1]
        self.model = HVAECore(x_dim, self.z1_dim, self.z2_dim, self.hidden).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        n = Xn.shape[0]; idx = np.arange(n)

        for ep in range(1, self.epochs + 1):
            self.rng.shuffle(idx)
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                xb, mb = Xn_t[b], Mn_t[b]
                z1, z2, kl = self.model.kl_terms(xb)
                xmu = self.model.decode(z2=z2, z1=z1)
                rec = masked_mse(xmu, xb, mb)
                loss = rec + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
            if self.verbose and (ep == 1 or ep % max(1, self.epochs // 20) == 0):
                print(f"[H-VAE] epoch {ep:4d} rec={rec.item():.4f} kl={kl.item():.4f}")
        return self

    @torch.no_grad()
    def transform(self, X, n_samples=8):
        X_raw, X_init, M = self._prepare(X)
        Xn = self.scaler.transform(X_init)
        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        outs = []
        self.model.eval()
        for _ in range(n_samples):
            (mu1, logvar1), (mu2, logvar2) = self.model.encode(Xn_t)
            z1 = mu1 + torch.randn_like(mu1) * (0.5 * logvar1).exp()
            z2 = mu2 + torch.randn_like(mu2) * (0.5 * logvar2).exp()
            xmu = self.model.decode(z2=z2, z1=z1).cpu().numpy()
            outs.append(xmu)
        rec_mean = np.mean(outs, axis=0)
        miss = np.isnan(X_raw)
        Xn_f = Xn.copy()
        Xn_f[miss] = rec_mean[miss]
        return self.scaler.inverse_transform(Xn_f)


# -----------------------------
# 4) HMC-VAE (single-layer + HMC refinement)
# -----------------------------
class HMC_VAE(VanillaVAE):
    def __init__(self, hmc_step=0.05, hmc_leapfrog=5, hmc_chains=1, **kwargs):
        super().__init__(**kwargs)
        self.hmc_step = hmc_step
        self.hmc_leapfrog = hmc_leapfrog
        self.hmc_chains = hmc_chains

    def _refine_z(self, mu, logvar, xb, mb):
        z0 = mu  # good init; could also sample
        z_ref = hmc_sample_z(
            init_z=z0,
            x=xb,
            mask=mb,
            decoder=self.model.decode,
            step_size=self.hmc_step,
            leapfrog_steps=self.hmc_leapfrog,
            chains=self.hmc_chains,
            noise_std=1.0
        )
        return z_ref

    def fit(self, X):
        # first, standard training (like VAE)
        super().fit(X)
        return self

    @torch.no_grad()
    def transform(self, X, n_samples=4):
        X_raw, X_init, M = self._prepare(X)
        Xn = self.scaler.transform(X_init)
        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        outs = []
        self.model.eval()
        for _ in range(n_samples):
            mu, logvar = self.model.encode(Xn_t)
            z_ref = self._refine_z(mu, logvar, Xn_t, Mn_t)
            xmu = self.model.decode(z_ref).cpu().numpy()
            outs.append(xmu)
        rec_mean = np.mean(outs, axis=0)
        miss = np.isnan(X_raw)
        Xn_f = Xn.copy()
        Xn_f[miss] = rec_mean[miss]
        return self.scaler.inverse_transform(Xn_f)


# -----------------------------
# 5) HH-VAEM (hierarchical + HMC)
# We HMC-sample on concatenated latent [z2, z1] via a decoder that maps (z2,z1)->x
# For simplicity, build a wrapper decoder that takes concat latent into x
# -----------------------------
class HH_VAEM(_BaseImputer):
    def __init__(self, z1_dim=32, z2_dim=16, hidden=(256, 256), beta=1.0,
                 hmc_step=0.03, hmc_leapfrog=5, hmc_chains=1, **kwargs):
        super().__init__(**kwargs)
        self.z1_dim, self.z2_dim, self.hidden, self.beta = z1_dim, z2_dim, hidden, beta
        self.hmc_step, self.hmc_leapfrog, self.hmc_chains = hmc_step, hmc_leapfrog, hmc_chains

    def fit(self, X):
        X_raw, X_init, M = self._prepare(X)
        self.scaler.fit(X_raw); Xn = self.scaler.transform(X_init)
        x_dim = Xn.shape[1]
        self.core = HVAECore(x_dim, self.z1_dim, self.z2_dim, self.hidden).to(self.device)
        opt = optim.Adam(self.core.parameters(), lr=self.lr)

        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        n = Xn.shape[0]; idx = np.arange(n)

        for ep in range(1, self.epochs + 1):
            self.rng.shuffle(idx)
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                xb, mb = Xn_t[b], Mn_t[b]
                z1, z2, kl = self.core.kl_terms(xb)
                xmu = self.core.decode(z2=z2, z1=z1)
                rec = masked_mse(xmu, xb, mb)
                loss = rec + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
            if self.verbose and (ep == 1 or ep % max(1, self.epochs // 20) == 0):
                print(f"[HH-VAEM/pretrain] epoch {ep:4d} rec={rec.item():.4f} kl={kl.item():.4f}")
        # done pretraining; keep core for HMC refine at inference
        return self

    @torch.no_grad()
    def transform(self, X, n_samples=4):
        X_raw, X_init, M = self._prepare(X)
        Xn = self.scaler.transform(X_init)
        Xn_t = to_torch(Xn, self.device)
        Mn_t = to_torch(M, self.device)
        self.core.eval()

        # Build a "decoder on concatenated z" for HMC
        def dec_concat(zcat):
            # zcat: [B, z2+z1]
            z2 = zcat[:, :self.z2_dim]
            z1 = zcat[:, self.z2_dim:]
            # (optionally) one step of p(z1|z2) alignment:
            # skip for speed; directly decode
            return self.core.decode(z2=z2, z1=z1)

        outs = []
        for _ in range(n_samples):
            # init from encoder
            (mu1, logvar1), (mu2, logvar2) = self.core.encode(Xn_t)
            z1_0 = mu1
            z2_0 = mu2
            z0 = torch.cat([z2_0, z1_0], dim=-1)
            z_ref = hmc_sample_z(
                init_z=z0,
                x=Xn_t,
                mask=Mn_t,
                decoder=dec_concat,
                step_size=self.hmc_step,
                leapfrog_steps=self.hmc_leapfrog,
                chains=self.hmc_chains,
                noise_std=1.0
            )
            xmu = dec_concat(z_ref).cpu().numpy()
            outs.append(xmu)
        rec_mean = np.mean(outs, axis=0)
        miss = np.isnan(X_raw)
        Xn_f = Xn.copy()
        Xn_f[miss] = rec_mean[miss]
        return self.scaler.inverse_transform(Xn_f)


class VAEImputer:
    """
    统一封装，提供 fit_transform 接口，可选择不同 VAE 变体。
    """
    def __init__(self, variant: str = "vanilla", **kwargs):
        self.variant = (variant or "vanilla").lower()
        self.kwargs = kwargs
        self._model = None
        self._columns = None
        self._index = None

    def _build_model(self):
        mapping = {
            "vanilla": VanillaVAE,
            "vae": VanillaVAE,
            "miwae": MIWAE,
            "h_vae": H_VAEM,
            "hver": H_VAEM,
            "hmc_vae": HMC_VAE,
            "hh_vaem": HH_VAEM,
        }
        if self.variant not in mapping:
            raise ValueError(f"Unknown VAE variant: {self.variant}")
        cls = mapping[self.variant]
        return cls(**self.kwargs)

    def fit(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        self._columns = df.columns
        self._index = df.index
        self._model = self._build_model()
        self._model.fit(df.to_numpy(dtype=np.float32))
        return self

    def transform(self, df: pd.DataFrame):
        if self._model is None:
            raise RuntimeError("VAEImputer must be fitted before calling transform.")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=self._columns)
        arr = df.to_numpy(dtype=np.float32)
        output = self._model.transform(arr)
        return pd.DataFrame(output, columns=df.columns, index=df.index)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        output = self._model.transform(df.to_numpy(dtype=np.float32))
        return pd.DataFrame(output, columns=self._columns, index=self._index)


# -----------------------------
# Minimal demo
# -----------------------------
from sklearn.metrics import r2_score
if __name__ == "__main__":
    np.random.seed(0)
    n, d = 800, 12
    X_full = np.random.randn(n, d).astype(np.float32)
    # make a few nonlinear dependencies to be learnable
    X_full[:, -3] = np.sin(X_full[:, 0]) + 0.1 * np.random.randn(n)
    X_full[:, -2] = X_full[:, 1] * X_full[:, 2] + 0.1 * np.random.randn(n)
    X_full[:, -1] = np.tanh(X_full[:, 0] - X_full[:, 1]) + 0.1 * np.random.randn(n)

    miss = (np.random.rand(n, d) < 0.2)
    X_miss = X_full.copy()
    X_miss[miss] = np.nan

    print("\n== Vanilla VAE ==")
    vae = VanillaVAE(z_dim=16, epochs=80, batch_size=256, verbose=True).fit(X_miss)
    X_imp_vae = vae.transform(X_miss)
    rmse = np.sqrt(np.nanmean((X_imp_vae[miss] - X_full[miss]) ** 2))
    VAER2 = r2_score(X_full[miss],X_imp_vae[miss])
    print("VAE RMSE:", round(float(rmse), 4))

    print("\n== MIWAE ==")
    miwae = MIWAE(z_dim=16, epochs=80, batch_size=256, K=5, verbose=True).fit(X_miss)
    X_imp_miwae = miwae.transform(X_miss)
    rmse = np.sqrt(np.nanmean((X_imp_miwae[miss] - X_full[miss]) ** 2))
    MIWAER2 = r2_score(X_full[miss], X_imp_miwae[miss])
    print("MIWAE RMSE:", round(float(rmse), 4))


    print("\n== H-VAE ==")
    hver = H_VAEM(z1_dim=16, z2_dim=8, epochs=80, batch_size=256, verbose=True).fit(X_miss)
    X_imp_hv = hver.transform(X_miss)
    rmse = np.sqrt(np.nanmean((X_imp_hv[miss] - X_full[miss]) ** 2))
    HVAER2 = r2_score(X_full[miss], X_imp_hv[miss])
    print("H-VAE RMSE:", round(float(rmse), 4))


    print("\n== HMC-VAE ==")
    hmcvae = HMC_VAE(z_dim=16, epochs=80, batch_size=256, hmc_step=0.05, hmc_leapfrog=5, verbose=True).fit(X_miss)
    X_imp_hmc = hmcvae.transform(X_miss)
    rmse = np.sqrt(np.nanmean((X_imp_hmc[miss] - X_full[miss]) ** 2))
    HMCR2 = r2_score(X_full[miss], X_imp_hmc[miss])
    print("HMC-VAE RMSE:", round(float(rmse), 4))


    print("\n== HH-VAEM ==")
    hh = HH_VAEM(z1_dim=16, z2_dim=8, epochs=80, batch_size=256, hmc_step=0.03, hmc_leapfrog=5, verbose=True).fit(X_miss)
    X_imp_hh = hh.transform(X_miss)
    rmse = np.sqrt(np.nanmean((X_imp_hh[miss] - X_full[miss]) ** 2))
    HHR2 = r2_score(X_full[miss], X_imp_hh[miss])
    print("HH-VAEM RMSE:", round(float(rmse), 4))
    print('HH-VAEM R2:', HHR2)
    print('HMC-VAE R2:', HMCR2)
    print('VAE R2:', VAER2)
    print('MIWAE R2:', MIWAER2)
    print('H-VAE R2:', HVAER2)
