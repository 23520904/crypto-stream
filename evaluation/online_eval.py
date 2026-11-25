"""M4 - Đánh giá online: so RMSE (thang min-max) của mô hình với persistence / ARIMA(1,0,1) / MLP khi giá thực tế đến.
  python -m evaluation.online_eval [--hours 24]"""
import argparse
import pandas as pd
from common.config import load_config, path_of, ensure
from common.evalutils import read_dir, load_scalers, join_predictions_actuals, rmse_by_step, rmse_mean, rmse_over_time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", help="thư mục stream (mặc định data/stream)")
    ap.add_argument("--hours", type=float, help="chỉ lấy N giờ gần nhất")
    a = ap.parse_args()
    cfg = load_config()
    d = a.stream or path_of(cfg, "stream")
    pred = read_dir(f"{d}/predictions")
    candles = read_dir(f"{d}/candles")
    if pred.empty:
        raise SystemExit("Chưa có dự báo nào. Hãy chạy speed layer trước.")
    if a.hours:
        pred = pred[pred["made_at"] >= pred["made_at"].max() - pd.Timedelta(hours=a.hours)]
    coins = sorted(pred["coin"].unique())
    j = join_predictions_actuals(pred, candles, load_scalers(cfg, coins))
    if j.empty:
        raise SystemExit("Chưa có giá thực tế khớp với dự báo.")
    out = ensure(path_of(cfg, "results", "online"))
    steps, mean, t = rmse_by_step(j), rmse_mean(j), rmse_over_time(j)
    steps.to_csv(out / "rmse_by_step.csv", index=False)
    mean.to_csv(out / "rmse_mean.csv", index=False)
    t.to_csv(out / "rmse_over_time.csv", index=False)
    print(f"Số dự báo đã có giá thực tế: {len(j):,}\n")
    print("RMSE trung bình 5 bước (thang min-max; vs_persistence_% > 0 = tốt hơn persistence):")
    print(mean.round(5).to_string(index=False))
    print("\nRMSE theo từng bước:")
    print(steps.pivot_table(index=["coin", "label"], columns="step", values="rmse").round(5).to_string())
    print(f"\nĐã lưu vào {out}")
