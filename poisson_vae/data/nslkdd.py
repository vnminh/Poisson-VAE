"""NSL-KDD, protocol of Baptiste et al. (arXiv:2602.19785):
train = normal rows of KDDTrain+; test = normal rows of KDDTest+; attack = attack rows of both files."""
import glob
import os
import numpy as np
import pandas as pd
from . import ensure_extracted

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"]
CAT_COLS = ["protocol_type", "service", "flag"]
BOOL_COLS = ["land", "logged_in", "is_guest_login", "is_host_login"]
NUM_COLS = [c for c in COLS[:41] if c not in CAT_COLS + BOOL_COLS]                       # 34
COUNT_COLS = ["count", "srv_count", "dst_host_count", "dst_host_srv_count", "hot", "num_failed_logins",
              "num_compromised", "num_root", "num_file_creations", "num_shells", "num_access_files",
              "wrong_fragment", "urgent", "su_attempted", "root_shell", "num_outbound_cmds"]   # 16
CATEGORY = {}
for _cat, _names in {
    "DoS": "neptune smurf back teardrop pod land apache2 mailbomb processtable udpstorm worm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "U2R": "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster snmpgetattack snmpguess xlock "
           "xsnoop named sendmail",
}.items():
    for _n in _names.split():
        CATEGORY[_n] = _cat
EXPECTED = dict(train=67343, test=9711, attack=71463)


def _find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    assert hits, f"{name} not found under {root}"
    return hits[0]


def load(data_dir="data"):
    root = ensure_extracted(data_dir, "nsl-kdd")
    tr = pd.read_csv(_find(root, "KDDTrain+.txt"), header=None, names=COLS)
    te = pd.read_csv(_find(root, "KDDTest+.txt"), header=None, names=COLS)
    tr["src"], te["src"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    attack = (df.label != "normal").to_numpy()
    part = np.where(~attack & (df.src == "train"), "train", np.where(~attack, "test", "attack"))
    assert {k: int((part == k).sum()) for k in EXPECTED} == EXPECTED
    category = np.where(attack, df.label.map(CATEGORY), "normal")
    assert not pd.isna(category).any(), "attack label missing from CATEGORY"

    vocab = {c: sorted(df[c].unique()) for c in CAT_COLS}                    # union of both files, label-free
    cat = np.stack([df[c].map({v: i for i, v in enumerate(vocab[c])}).to_numpy() for c in CAT_COLS], 1)
    onehot = np.concatenate([np.eye(len(vocab[c]), dtype=np.float32)[cat[:, i]] for i, c in enumerate(CAT_COLS)], 1)
    boo = df[BOOL_COLS].to_numpy(np.float32)
    raw = df[NUM_COLS].to_numpy(np.float64)
    mu, sd = raw[part == "train"].mean(0), raw[part == "train"].std(0)
    num = ((raw - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)        # constant columns: divide by 1
    return dict(x=np.concatenate([onehot, boo, num], 1), cat=cat.astype(np.int64), boo=boo, num=num,
                counts=df[COUNT_COLS].to_numpy(np.float32), part=part, label=df.label.to_numpy(), category=category,
                cat_sizes=[len(vocab[c]) for c in CAT_COLS], num_cols=NUM_COLS, count_cols=COUNT_COLS)
