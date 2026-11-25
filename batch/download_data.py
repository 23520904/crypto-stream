"""Tải dữ liệu lịch sử BTC, ETH, vàng (Yahoo Finance) và cache vào data/raw/."""
import argparse
from common.config import load_config
from common.data import load_asset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    for name in ("BTC", "ETH", "GOLD"):
        df = load_asset(name, cfg, refresh=a.refresh)
        print(f"{name}: {len(df)} ngày  {df.index.min().date()} -> {df.index.max().date()}  close TB={df['close'].mean():.2f}")
