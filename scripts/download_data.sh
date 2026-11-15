#!/usr/bin/env bash
# Download both datasets into data/ as one zip each (data/ is not tracked by git).
set -e
mkdir -p data && cd data
if [ ! -f nsl-kdd.zip ]; then
  for f in KDDTrain+.txt KDDTest+.txt; do
    curl -fL -o "$f" "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/$f"
  done
  zip -q nsl-kdd.zip KDDTrain+.txt KDDTest+.txt && rm KDDTrain+.txt KDDTest+.txt
fi
if [ ! -f cic-ids2017.zip ]; then
  curl -fL -o cic-ids2017.zip \
    "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/GeneratedLabelledFlows.zip"
fi
ls -lh nsl-kdd.zip cic-ids2017.zip
