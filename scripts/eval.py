"""Evaluate trained runs (AUROC).
  python scripts/eval.py --dataset nslkdd --run runs/nslkdd/negbin/b1e-05/s0
  python scripts/eval.py --dataset cicids2017            # every run under runs/cicids2017, with mean ± std
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from poisson_vae.data import DATASETS, load_dataset
from poisson_vae.engine import build_model, device, evaluate, score

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=DATASETS)
p.add_argument("--run", default=None, help="one run directory; default: all runs under runs/<dataset>")
p.add_argument("--runs-dir", default="runs")
p.add_argument("--data-dir", default="data")
p.add_argument("--knn", action="store_true", help="also compute the latent-distance scores Z_k")
a = p.parse_args()

runs = [a.run] if a.run else sorted(os.path.dirname(f) for f in
                                    glob.glob(os.path.join(a.runs_dir, a.dataset, "**", "model.pt"), recursive=True))
assert runs, "no trained runs found; run scripts/train.py first"
D = load_dataset(a.dataset, a.data_dir)
rows = []
for r in runs:
    cfg = json.load(open(os.path.join(r, "config.json")))
    assert cfg["dataset"] == a.dataset, f"{r} was trained on {cfg['dataset']}"
    m = build_model(D, cfg["loss"]).to(device())
    m.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location=device(), weights_only=True))
    ev, sc = score(m, D, cfg["seed"], knn=a.knn)
    res = evaluate(D, ev, sc)
    json.dump(res, open(os.path.join(r, "metrics.json"), "w"), indent=1)
    rows.append((cfg["loss"], cfg["beta"], cfg["seed"], res))
    print(f"\n{r}\n  overall   " + "  ".join(f"{k} {v:.4f}" for k, v in res["overall"].items()))
    for g, d in res["per_category"].items():
        print(f"  {g:<12} n={d['n']:<7} Lrec_dev {d['Lrec_dev']:.4f}  Lrec_nll {d['Lrec_nll']:.4f}")

if len(rows) > 1:
    print(f"\nSummary on {a.dataset}: AUROC (%) mean ± std over seeds")
    print(f"{'loss':<8} {'beta':<8} {'seeds':<5} {'Lrec_dev':<14} Lrec_nll")
    for key in sorted({(l, b) for l, b, _, _ in rows}):
        v = np.array([[r["overall"]["Lrec_dev"], r["overall"]["Lrec_nll"]] for l, b, _, r in rows if (l, b) == key]) * 100
        sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(2)
        print(f"{key[0]:<8} {key[1]:<8g} {len(v):<5} {v[:, 0].mean():.2f} ± {sd[0]:.2f}   {v[:, 1].mean():.2f} ± {sd[1]:.2f}")
