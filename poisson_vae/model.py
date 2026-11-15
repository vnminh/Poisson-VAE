"""β-VAE of Baptiste et al.: encoder 64-32-16 -> (mu, logvar) in R^8, decoder 16-32-64 -> one head per column type.
Count columns get a log-mean head trained with a Poisson or negative-binomial likelihood."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import poisson_nll, negbin_nll, half_deviance

LOSSES = ("mse", "poisson", "negbin")


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class BetaVAE(nn.Module):
    def __init__(self, d_in, cat_sizes, n_bool, n_mse, n_count, loss="negbin", latent=8, eta_clamp=(-10.0, 12.0)):
        super().__init__()
        assert loss in LOSSES and (n_count == 0) == (loss == "mse")
        self.loss, self.cat_sizes, self.n_bool, self.n_mse, self.n_count = loss, list(cat_sizes), n_bool, n_mse, n_count
        self.eta_clamp = eta_clamp
        if loss == "negbin":
            self.log_r = nn.Parameter(torch.full((n_count,), 2.0))      # one dispersion per count column
        self.enc = mlp([d_in, 64, 32, 16])
        self.mu = nn.Linear(16, latent)
        self.logvar = nn.Linear(16, latent)
        self.dec = mlp([latent, 16, 32, 64])
        self.out = nn.Linear(64, sum(cat_sizes) + n_bool + n_mse + n_count)

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-30.0, 30.0)

    def sample(self, x):
        mu, lv = self.encode(x)
        return mu + torch.randn_like(mu) * (0.5 * lv).exp(), mu, lv

    def reconstruction(self, z, cat, boo, mse, counts):
        """Per-sample reconstruction terms -> (nll, dev): `nll` uses the count NLL (training objective and the
        paper-style score), `dev` replaces it with the half-deviance (proposed anomaly score). Equal for loss='mse'."""
        o = self.out(self.dec(z))
        p, rest = 0, torch.zeros(len(z), device=z.device)
        for i, n in enumerate(self.cat_sizes):
            rest = rest + F.cross_entropy(o[:, p:p + n], cat[:, i], reduction="none"); p += n
        rest = rest + F.binary_cross_entropy_with_logits(o[:, p:p + self.n_bool], boo, reduction="none").sum(1)
        p += self.n_bool
        rest = rest + ((o[:, p:p + self.n_mse] - mse) ** 2).sum(1); p += self.n_mse
        if self.loss == "mse":
            return rest, rest
        eta = o[:, p:].clamp(*self.eta_clamp)
        log_r = self.log_r.clamp(-5.0, 12.0) if self.loss == "negbin" else None
        nll = negbin_nll(counts, eta, log_r) if self.loss == "negbin" else poisson_nll(counts, eta)
        return rest + nll.sum(1), rest + half_deviance(counts, eta, log_r).sum(1)

    @staticmethod
    def kl(mu, lv):
        return 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1)
