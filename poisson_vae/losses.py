"""Per-element negative log-likelihoods and half-deviances of the count distributions (mean mu = exp(eta))."""
import torch
import torch.nn.functional as F


def poisson_nll(x, eta):
    return eta.exp() - x * eta + torch.lgamma(x + 1)


def negbin_nll(x, eta, log_r):
    """Negative binomial with mean exp(eta) and size r = exp(log_r): Var = mu + mu^2 / r.
    Computed in float64 because lgamma(x + r) - lgamma(r) cancels badly for large r."""
    dt = x.dtype
    x, eta, log_r = x.double(), eta.double(), log_r.double()
    r = log_r.exp()
    ll = (torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
          - r * F.softplus(eta - log_r) - x * F.softplus(log_r - eta))
    return (-ll).to(dt)


def half_deviance(x, eta, log_r=None):
    """NLL(x | mu) - NLL(x | mu = x): non-negative, zero for a perfect fit. Poisson if log_r is None, else NegBin."""
    x, eta = x.double(), eta.double()
    mu = eta.exp()
    xlog = torch.xlogy(x, x) - x * eta                          # x log(x / mu), 0 at x = 0
    if log_r is None:
        return (xlog - (x - mu)).float()
    r = log_r.double().exp()
    return (xlog - (x + r) * (torch.log(x + r) - torch.log(mu + r))).float()
