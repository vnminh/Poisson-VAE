# Poisson-VAE

β-VAE phát hiện xâm nhập mạng không giám sát, trong đó các đặc trưng đếm được mô hình hoá bằng phân phối Poisson
hoặc nhị thức âm (Negative Binomial) và điểm bất thường được tính bằng độ lệch deviance.
Mô hình cơ sở: Baptiste et al., arXiv:2602.19785.

## Cài đặt

```bash
pip install -r requirements.txt
bash scripts/download_data.sh        # tạo data/nsl-kdd.zip và data/cic-ids2017.zip
```

Dữ liệu được giải nén tự động ở lần chạy đầu. Với CIC-IDS2017, lần đầu mất khoảng 1–2 phút và cần khoảng 9 GB RAM
để xử lý; kết quả được lưu lại ở `data/cic-ids2017/cache.npz`.

## Huấn luyện

```bash
python scripts/train.py --dataset nslkdd --loss negbin --beta 1e-5 --seed 0
python scripts/train.py --dataset cicids2017 --loss mse --beta 1e-5 --seed 0
```

- `--dataset`: `nslkdd` hoặc `cicids2017`
- `--loss`: `mse` (đối chứng), `poisson`, `negbin`
- Mặc định 200 epoch; đổi bằng `--epochs`. Mô hình lưu ở `runs/<dataset>/<loss>/b<beta>/s<seed>/`.

## Đánh giá

```bash
python scripts/eval.py --dataset nslkdd --run runs/nslkdd/negbin/b1e-05/s0   # một mô hình
python scripts/eval.py --dataset nslkdd                                       # mọi mô hình của dataset, kèm bảng tổng hợp
```

Kết quả là AUROC tổng thể và theo từng nhóm tấn công, với hai cách chấm điểm: `Lrec_dev` (deviance, phương pháp đề
xuất) và `Lrec_nll` (NLL). Thêm `--knn` để tính cả điểm khoảng cách trong không gian ẩn `Z_k`.

## Chạy lại toàn bộ thực nghiệm

```bash
bash scripts/run_grid.sh nslkdd        # 3 loss × 7 β × 4 seed, rồi đánh giá
bash scripts/run_grid.sh cicids2017
```

## Cấu trúc

```
poisson_vae/
  data/        nslkdd.py, cicids2017.py   đọc và tiền xử lý dữ liệu
  losses.py    NLL và deviance của Poisson / NegBin
  model.py     β-VAE
  engine.py    huấn luyện, chấm điểm, tính AUROC
scripts/       train.py, eval.py, run_grid.sh, download_data.sh
```
