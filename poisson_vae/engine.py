"""Training loop, anomaly scoring and AUROC evaluation."""
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .data import targets
from .model import BetaVAE

K_LIST = [1, 100, 500, 1000, 5000]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(D, loss, latent=8):
    mse, counts = targets(D, loss)
    return BetaVAE(D["x"].shape[1], D["cat_sizes"], D["boo"].shape[1], mse.shape[1], counts.shape[1], loss, latent)


def _batch(arrays, idx, dev):
    return [torch.as_tensor(a[idx], device=dev) for a in arrays]


def train(D, loss, beta, seed, epochs=200, batch_size=2048, lr=1e-3, log=print):
    """Train on the normal training rows only. The count terms are optimised through their NLL: its gradient with
    respect to the network equals that of the deviance, and it also fits the NegBin dispersion r."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device()
    m = build_model(D, loss).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    mse, counts = targets(D, loss)
    tr = np.where(D["part"] == "train")[0]
    X = _batch([D["x"], D["cat"], D["boo"], mse, counts], tr, dev)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train(); t0, tot, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(tr), generator=g).to(dev)
        for b in range(0, len(tr), batch_size):
            x, cat, boo, ms, cn = [a[perm[b:b + batch_size]] for a in X]
            z, mu, lv = m.sample(x)
            nll, _ = m.reconstruction(z, cat, boo, ms, cn)
            l = (nll + beta * m.kl(mu, lv)).mean()
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item(); nb += 1
        if not np.isfinite(tot):
            raise RuntimeError("non-finite loss")
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"epoch {ep + 1}/{epochs}  loss {tot / nb:.4f}  ({time.time() - t0:.1f}s)")
    return m


@torch.no_grad()
def score(m, D, seed=0, batch_size=8192, knn=False, n_ref=50_000):
    """Anomaly scores for the evaluation rows (test ∪ attack), one z sample per row:
    Lrec_dev (proposed), Lrec_nll (paper-style) and, if knn, Z_k = mean distance to the k nearest training latents."""
    dev = next(m.parameters()).device
    m.eval(); torch.manual_seed(10_000 + seed)
    mse, counts = targets(D, m.loss)
    arrays = [D["x"], D["cat"], D["boo"], mse, counts]
    ev = np.where(D["part"] != "train")[0]
    nll, dv, zs = [], [], []
    for b in range(0, len(ev), batch_size):
        x, cat, boo, ms, cn = _batch(arrays, ev[b:b + batch_size], dev)
        z, _, _ = m.sample(x)
        a, d = m.reconstruction(z, cat, boo, ms, cn)
        nll.append(a.cpu().numpy()); dv.append(d.cpu().numpy()); zs.append(z)
    out = {"Lrec_dev": np.concatenate(dv), "Lrec_nll": np.concatenate(nll)}
    if knn:
        tr = np.where(D["part"] == "train")[0]
        ref = np.sort(np.random.RandomState(seed).choice(tr, min(n_ref, len(tr)), replace=False))
        zr = torch.cat([m.sample(torch.as_tensor(D["x"][ref[b:b + batch_size]], device=dev))[0]
                        for b in range(0, len(ref), batch_size)])
        zq, ks = torch.cat(zs), [k for k in K_LIST if k <= len(zr)]
        kidx = torch.as_tensor([k - 1 for k in ks], device=dev)
        Z = np.concatenate([(torch.cdist(zq[b:b + 1024], zr).topk(max(ks), dim=1, largest=False).values
                             .cumsum(1)[:, kidx] / (kidx + 1)).cpu().numpy() for b in range(0, len(zq), 1024)])
        out.update({f"Z{k}": Z[:, j] for j, k in enumerate(ks)})
    return ev, out


def evaluate(D, ev, scores):
    """AUROC of every score on test ∪ attack, and per attack category (held-out normal rows vs that category)."""
    part, cat = D["part"][ev], D["category"][ev]
    y, normal = part == "attack", part == "test"
    res = {"overall": {k: float(roc_auc_score(y, v)) for k, v in scores.items()}, "per_category": {}}
    for g in sorted(set(cat[y])):
        k = normal | (cat == g)
        res["per_category"][g] = {s: float(roc_auc_score(y[k], v[k])) for s, v in scores.items()} | {"n": int((cat == g).sum())}
    return res
