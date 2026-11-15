"""Train one β-VAE.   python scripts/train.py --dataset nslkdd --loss negbin --beta 1e-5 --seed 0"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from poisson_vae.data import DATASETS, load_dataset
from poisson_vae.engine import train
from poisson_vae.model import LOSSES

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dataset", required=True, choices=DATASETS)
p.add_argument("--loss", default="negbin", choices=LOSSES, help="likelihood of the count columns (mse = baseline)")
p.add_argument("--beta", type=float, default=1e-5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--batch-size", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--data-dir", default="data")
p.add_argument("--out", default=None, help="run directory (default runs/<dataset>/<loss>/b<beta>/s<seed>)")
a = p.parse_args()

out = a.out or os.path.join("runs", a.dataset, a.loss, f"b{a.beta:g}", f"s{a.seed}")
os.makedirs(out, exist_ok=True)
D = load_dataset(a.dataset, a.data_dir)
t0 = time.time()
m = train(D, a.loss, a.beta, a.seed, a.epochs, a.batch_size, a.lr)
torch.save(m.state_dict(), os.path.join(out, "model.pt"))
cfg = dict(vars(a), out=out, train_seconds=round(time.time() - t0, 1))
if a.loss == "negbin":
    cfg["negbin_r"] = dict(zip(D["count_cols"], m.log_r.detach().clamp(-5, 12).exp().cpu().tolist()))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
print(f"saved {out}/model.pt  ({cfg['train_seconds']}s)")
