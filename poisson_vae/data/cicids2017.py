"""CIC-IDS2017 (original GeneratedLabelledFlows CSVs).
train = all Monday flows (benign); test = benign flows of Tuesday-Friday; attack = attack flows of Tuesday-Friday.
Preprocessing uses Monday statistics only. The processed arrays are cached in <data_dir>/cic-ids2017/cache.npz."""
import glob
import json
import os
import numpy as np
import pandas as pd
from . import ensure_extracted

DROP = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Label"]
CAT_COLS = ["Protocol", "Destination Port"]
BOOL_COLS = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "FIN Flag Count", "SYN Flag Count",
             "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count"]
COUNT_CANDIDATES = ["Total Fwd Packets", "Total Backward Packets", "Subflow Fwd Packets", "Subflow Bwd Packets",
                    "act_data_pkt_fwd"]
PORT_MIN = 100                      # destination ports with fewer Monday flows are grouped as "other"
CATEGORY = {"DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
            "DDoS": "DDoS", "PortScan": "PortScan", "Bot": "Bot", "Infiltration": "Infiltration",
            "Web Attack - Brute Force": "WebAttack", "Web Attack - XSS": "WebAttack",
            "Web Attack - Sql Injection": "WebAttack", "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
            "Heartbleed": "Heartbleed"}
EXPECTED = dict(train=529918, test=1743179, attack=557646)
ARRAYS = ["x", "cat", "boo", "num", "counts", "part", "label", "category"]


def _read(root):
    feats, labels, monday = [], [], []
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    assert len(files) == 8, f"expected 8 CSV files under {root}, found {len(files)}"
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Label"].notna()]
        labels.append(df["Label"].astype(str).str.replace("\x96", "-", regex=False).str.replace("�", "-", regex=False)
                      .str.strip().to_numpy())
        monday.append(np.full(len(df), "Monday" in os.path.basename(f)))
        df = df.drop(columns=DROP)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        feats.append(df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float64))
    return pd.concat(feats, ignore_index=True), np.concatenate(labels), np.concatenate(monday)


def _build(root):
    df, lab, mon = _read(root)
    part = np.where(mon, "train", np.where(lab == "BENIGN", "test", "attack"))
    assert {k: int((part == k).sum()) for k in EXPECTED} == EXPECTED
    category = np.array([CATEGORY.get(l, "normal") for l in lab])
    assert not ((part == "attack") & (category == "normal")).any(), "attack label missing from CATEGORY"

    num_cols = []                                                # drop columns identical to an earlier one on Monday
    for c in df.columns:
        if c in CAT_COLS or c in BOOL_COLS:
            continue
        v = df[c].to_numpy()[mon]
        if not any(np.array_equal(v, df[k].to_numpy()[mon]) for k in num_cols):
            num_cols.append(c)
    count_cols = [c for c in COUNT_CANDIDATES if c in num_cols]

    cats, cat_sizes = [], []
    port_n = df.loc[mon, "Destination Port"].astype(np.int64).value_counts()
    vocabs = {"Protocol": sorted(df.loc[mon, "Protocol"].astype(np.int64).unique().tolist()),
              "Destination Port": sorted(port_n[port_n >= PORT_MIN].index.tolist())}
    for c in CAT_COLS:
        m = {v: i for i, v in enumerate(vocabs[c])}
        cats.append(df[c].astype(np.int64).map(m).fillna(len(m)).to_numpy(np.int64))     # last index = "other"
        cat_sizes.append(len(m) + 1)
    cat = np.stack(cats, 1)
    onehot = np.concatenate([np.eye(s, dtype=np.float32)[cat[:, i]] for i, s in enumerate(cat_sizes)], 1)
    boo = (df[BOOL_COLS].to_numpy() > 0).astype(np.float32)
    raw = df[num_cols].to_numpy(np.float64)
    t = np.sign(raw) * np.log1p(np.abs(raw))                     # heavy tails: signed log1p before z-scoring
    mu, sd = t[mon].mean(0), t[mon].std(0)
    num = ((t - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    return dict(x=np.concatenate([onehot, boo, num], 1), cat=cat, boo=boo, num=num,
                counts=df[count_cols].to_numpy(np.float32), part=part, label=lab, category=category,
                cat_sizes=cat_sizes, num_cols=num_cols, count_cols=count_cols)


def load(data_dir="data"):
    root = ensure_extracted(data_dir, "cic-ids2017")
    cache = os.path.join(root, "cache.npz")
    if not os.path.exists(cache):
        D = _build(root)
        np.savez(cache, **{k: D[k] for k in ARRAYS})
        json.dump({k: D[k] for k in ["cat_sizes", "num_cols", "count_cols"]}, open(cache + ".json", "w"))
        return D
    Z = np.load(cache, allow_pickle=True)
    return {**{k: Z[k] for k in ARRAYS}, **json.load(open(cache + ".json"))}
