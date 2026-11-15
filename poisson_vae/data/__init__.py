"""Dataset loaders. Each returns a dict with the same keys:

  x        float32 [n, d_in]   encoder input (one-hot | booleans | z-scored numeric columns)
  cat      int64   [n, n_cat]  categorical targets (class index per column)
  boo      float32 [n, n_bool] boolean targets
  num      float32 [n, n_num]  all numeric columns, transformed and z-scored (MSE targets)
  counts   float32 [n, n_cnt]  raw values of the count columns (Poisson / NegBin targets)
  part     str     [n]         "train" (normal, for training) | "test" (normal, held out) | "attack"
  label, category  str [n]     attack name and attack group ("normal" for benign rows)
  cat_sizes, num_cols, count_cols   metadata
"""
import os
import zipfile
import numpy as np

DATASETS = ("nslkdd", "cicids2017")


def ensure_extracted(data_dir, name):
    """Return data_dir/<name>, extracting data_dir/<name>.zip on first use."""
    out, z = os.path.join(data_dir, name), os.path.join(data_dir, name + ".zip")
    if not os.path.isdir(out):
        if not os.path.exists(z):
            raise FileNotFoundError(f"{z} not found; run `bash scripts/download_data.sh` first")
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    return out


def load_dataset(name, data_dir="data"):
    if name == "nslkdd":
        from .nslkdd import load
    elif name == "cicids2017":
        from .cicids2017 import load
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    return load(data_dir)


def targets(D, loss):
    """MSE and count targets for a loss: 'mse' puts every numeric column under MSE; 'poisson' / 'negbin' move the
    count columns to the count likelihood."""
    if loss == "mse":
        return D["num"], np.zeros((len(D["num"]), 0), np.float32)
    keep = [i for i, c in enumerate(D["num_cols"]) if c not in D["count_cols"]]
    return np.ascontiguousarray(D["num"][:, keep]), D["counts"]
