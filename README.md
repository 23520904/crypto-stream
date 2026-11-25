# Hệ thống truy xuất dữ liệu Coinbase & dự đoán giá BTC/ETH thời gian thực (kiến trúc Lambda)

Coinbase WebSocket → Python Producer → **Kafka** → **Spark Structured Streaming** (làm sạch, khử trùng, nến 1 phút, ghép BTC–ETH–vàng)
→ **BD-LSTM / Conv-LSTM** dự báo 5 bước → **Parquet** → **Streamlit + Plotly**.
Batch layer: dữ liệu lịch sử → huấn luyện/đánh giá (tái hiện bài báo [1]) → xuất mô hình + tham số min-max cho speed layer.

## Cấu trúc & ánh xạ mục tiêu

| Mục tiêu | Mã nguồn |
|---|---|
| M1 Thu thập Kafka ≥ 45 ngày, tự kết nối lại, phát hiện mất gói | `ingestion/coinbase_producer.py`, `ingestion/create_topics.py`, `docker-compose.yml` |
| M2 Tái hiện bài báo [1] (batch) | `batch/train_batch.py`, `batch/compare_paper.py`, `common/{data,features,models,training}.py` |
| M3 Speed layer Kafka→Spark→dự báo 5 bước | `streaming/spark_job.py`, `streaming/runtime.py`, `streaming/predictor.py` |
| M4 Đánh giá online vs persistence/ARIMA/MLP | `evaluation/online_eval.py`, `common/evalutils.py` |
| M5 Độ trễ p50/p95, thông lượng, 1/2/4 core | `benchmark/replay.py`, `benchmark/run_benchmark.py` |
| M6 Dashboard | `dashboard/app.py` |

## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate      # Linux / macOS / WSL2 (khuyến nghị; tránh Windows thuần cho Spark)
pip install -r requirements.txt
# Cần: Docker (Kafka), Java 17 (Spark). Lần chạy Spark đầu tiên cần Internet để tải gói spark-sql-kafka.
```

## Quy trình chạy

### 1. Batch layer (M2) — làm trước để có mô hình cho speed layer
```bash
python -m batch.download_data                                  # BTC, ETH (Yahoo), vàng GC=F; cache data/raw/
# (tuỳ chọn) đặt CSV Kaggle vào data/raw/kaggle/coin_Bitcoin.csv, coin_Ethereum.csv để lấy lịch sử từ 2013

python -m batch.train_batch --exp 1 --coin BTC --strategy univariate --model BD-LSTM --search   # chọn hidden theo val-loss
python -m batch.train_batch --exp 1 --suite m2                 # 30 lần chạy/cấu hình, RMSE ± CI95 từng bước 1–5
python -m batch.compare_paper                                  # bảng so với bài báo: BD-LSTM ≈0,0327 ; Conv-LSTM* ≈0,0200
python -m batch.train_batch --exp 1 --suite grid               # (tuỳ chọn) đủ 3 chiến lược × 2 mô hình × BTC/ETH
python -m batch.train_batch --exp 2 --suite grid               # (tuỳ chọn) Experiment 2, train có COVID-19

python -m batch.train_batch --exp deploy --suite deploy        # huấn luyện + XUẤT artifacts/ cho speed layer
python -m pytest tests -q
```
Kết quả: `results/batch/exp{1,2}/*.json`, `results/m2_reproduction.csv`; mô hình + scaler: `artifacts/`.

### 2. Thu thập tick (M1) — chạy liền ≥ 45 ngày
```bash
docker compose up -d            # Kafka (KRaft) + tạo topic + producer (restart: always)
docker compose logs -f producer
cat data/metrics/producer_status.json        # trạng thái, số ngày chạy, gap, reconnect
cat data/metrics/gap_events.jsonl            # từng sự kiện mất gói
```
Cơ chế: reconnect backoff 1→60 s có jitter; watchdog khi 10 s không có message; mất gói xác định bằng **`trade_id` liên tiếp**,
đối chiếu **`last_trade_id` của heartbeat** (lệch ≥ 3 heartbeat), `sequence` để bắt trùng/đảo thứ tự; tải bù qua REST `/trades`
(tick bù có `source="backfill"`). Thống kê tích luỹ qua các lần khởi động nằm ở `producer_cumulative.json`
(dùng làm bằng chứng chạy 45 ngày + số tick gốc để điền báo cáo).

> Lưu ý kỹ thuật: kênh `ticker` chỉ phát khi có khớp lệnh, còn `sequence` đánh số mọi message của product nên khoảng trống `sequence`
> là bình thường — đó là lý do dùng `trade_id`/heartbeat làm tiêu chí chính. Tick tải bù luôn vào Kafka (lưu trữ/replay), nhưng chỉ
> vào nến live nếu còn trong watermark (15 s).

### 3. Speed layer (M3)
```bash
python -m streaming.spark_job --master "local[2]"
```
Spark đọc topic `ticks`, lọc/khử trùng (`dropDuplicatesWithinWatermark`), gom nến 1 phút (open/high/low/close/volume/n_ticks)
rồi `foreachBatch` ghép BTC–ETH–vàng, dự báo 5 bước cho từng coin bằng BD-LSTM [univariate], Conv-LSTM [multivariate*], MLP
và 2 baseline (persistence, ARIMA(1,0,1)); ghi `data/stream/{candles,predictions,metrics}` (Parquet, ghi nguyên tử).
Khởi động lại thì khôi phục lịch sử nến và phút dự báo cuối từ đĩa.

### 4. Đánh giá online (M4) và Dashboard (M6)
```bash
python -m evaluation.online_eval --hours 24
streamlit run dashboard/app.py
```

### 5. Hiệu năng (M5)
```bash
python -m benchmark.replay --from-topic --max-messages 2000000     # phát lại tick thật đã thu  (hoặc --synthetic để thử nhanh)
python -m benchmark.run_benchmark --cores 1 2 4                    # results/benchmark/summary.csv
```

## Các điểm cần biết khi viết báo cáo
1. **Chuẩn hoá & tái hiện M2.** Theo mô tả đề tài, min-max fit trên tập train. Giá BTC tập test (2019–2021) vượt xa khoảng giá train nên giá trị scaled > 1;
   RMSE có thể lệch lớn so với 0,0200 / 0,0327 của bài báo. Thử `batch.scaler_fit: all` trong `config.yaml` để so sánh và nêu rõ trong báo cáo.
   Dữ liệu cũng khác (Yahoo từ 2014-09 thay vì 2013-04; vàng GC=F thay LBMA).
2. **Mô hình ngày áp vào dữ liệu phút.** Mô hình huấn luyện trên nến ngày, speed layer áp lên nến 1 phút (đúng mô tả đề tài). Đây là chuyển miền dữ liệu,
   nên RMSE online có thể kém persistence — kết quả M4 phải được báo cáo trung thực. Nếu muốn, huấn luyện lại trên chính nến phút đã thu.
3. **Đặc trưng vàng ở streaming** dùng token vàng `PAXG-USD` trên Coinbase (khác kênh ticker BTC/ETH trong mô tả); tắt bằng `coinbase.gold_proxy: null`
   (khi đó dùng giá vàng cuối của tập train).
4. **Conv-LSTM** cài theo ConvLSTM1D: cửa sổ 6 điểm chia 2 đoạn × 3 bước, kênh = số đặc trưng, không có peephole như công thức (4) của bài báo.
5. **Chọn mô hình xuất** theo val-loss (10% cuối train), không dùng test. Tham số hidden: BD-LSTM 100, Conv-LSTM 20, MLP 20 (theo Table 8 bài báo), có `--search` để chọn lại.
6. **Độ trễ:** `latency_ms` = từ lúc nến vào `foreachBatch` đến lúc có dự báo; `e2e_ms` = từ hết phút của nến (chỉ có ý nghĩa ở chế độ live; gồm watermark + trigger).
   Torch chạy ở driver nên số core ảnh hưởng chính tới phần Spark (đọc/khử trùng/gom nến); `--threads` đặt bằng số core để phép đo công bằng.
