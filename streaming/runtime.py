"""Speed layer (phần chạy trên driver trong foreachBatch): giữ lịch sử nến, ghép BTC-ETH-vàng,
dự báo 5 bước mỗi phút, ghi Parquet + số đo độ trễ."""
from __future__ import annotations
import json, os, time, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common.config import ensure
from common.features import COLUMNS
from .predictor import Predictor

CANDLE_COLS = ["open", "high", "low", "close", "volume", "n_ticks"]
MIN = pd.Timedelta(minutes=1)


class SpeedLayer:
    def __init__(self, cfg: dict, out_dir: Path, threads: int = 1):
        torch.set_num_threads(threads)
        self.cfg, self.out = cfg, Path(out_dir)
        for sub in ("candles", "predictions", "metrics"):
            ensure(self.out / sub)
        self.W, self.H = cfg["batch"]["window"], cfg["batch"]["horizon"]
        self.keep = cfg["streaming"]["keep_minutes"]
        self.n_hist = cfg["streaming"]["arima_history"]
        self.pids = list(cfg["coinbase"]["products"])
        self.coins = [p.split("-")[0] for p in self.pids]
        self.gold_pid = cfg["coinbase"].get("gold_proxy")
        if self.gold_pid:
            self.pids.append(self.gold_pid)
        self.predictor = Predictor(cfg, self.coins)
        self.hist: dict[str, pd.DataFrame] = {}
        self.last_pred = None
        self._recover()

    # ----- lưu / khôi phục -----
    def _write(self, sub: str, df: pd.DataFrame, batch_id: int):
        d = self.out / sub
        name = f"part-{batch_id:08d}-{uuid.uuid4().hex[:6]}.parquet"
        tmp = d / f"_tmp-{name}"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, d / name)          # ghi nguyên tử: dashboard không đọc phải file dở

    def _recover(self):
        cd = self.out / "candles"
        if any(cd.glob("*.parquet")):
            df = pd.read_parquet(cd)
            for pid, g in df.groupby("product_id"):
                self.hist[pid] = g.drop_duplicates("minute", keep="last").set_index("minute").sort_index()[CANDLE_COLS].tail(self.keep)
        st = self.out / "metrics" / "state.json"
        if st.exists():
            self.last_pred = pd.Timestamp(json.loads(st.read_text())["last_pred"])

    # ----- ghép dữ liệu -----
    def _series(self, pid, m, idx):
        h = self.hist.get(pid)
        if h is None:
            return None
        d = h.loc[:m, ["open", "high", "low", "close"]].tail(self.n_hist)
        if d.empty:
            return None
        full = d.reindex(pd.date_range(d.index.min(), m, freq="min"))
        c = full["close"].ffill()
        for col in ("open", "high", "low"):
            full[col] = full[col].fillna(c)       # phút không có giao dịch: nến phẳng ở giá trước đó
        full["close"] = c
        return full.reindex(idx)

    def _frame(self, coin, m):
        partner = self.coins[1] if coin == self.coins[0] else self.coins[0]
        idx = pd.date_range(end=m, periods=self.n_hist, freq="min")
        x, p = self._series(f"{coin}-USD", m, idx), self._series(f"{partner}-USD", m, idx)
        if x is None or p is None:
            return None
        df = x.copy()
        df["partner_close"] = p["close"]
        g = self._series(self.gold_pid, m, idx) if self.gold_pid else None
        last_gold = self.predictor.scalers[coin].last["gold"]
        df["gold"] = (g["close"].ffill().bfill() if g is not None else pd.Series(np.nan, index=idx)).fillna(last_gold)
        df = df[COLUMNS].dropna()
        return df if len(df) >= self.W else None

    # ----- xử lý 1 micro-batch -----
    def process_batch(self, pdf: pd.DataFrame, batch_id: int, t_recv: float):
        t_begin = time.time()
        if pdf is None or pdf.empty:
            return
        pdf = pdf.copy()
        pdf["minute"] = pd.to_datetime(pdf["minute"]).dt.tz_localize(None)
        self._write("candles", pdf[["minute", "product_id"] + CANDLE_COLS], batch_id)
        for pid, g in pdf.groupby("product_id"):
            g = g.set_index("minute")[CANDLE_COLS].sort_index()
            h = g if pid not in self.hist else pd.concat([self.hist[pid], g])
            self.hist[pid] = h[~h.index.duplicated(keep="last")].sort_index()
        rows, n_min = self._predict_ready(batch_id, t_recv)
        for pid in self.hist:
            self.hist[pid] = self.hist[pid].tail(self.keep)
        if rows:
            self._write("predictions", pd.DataFrame(rows), batch_id)
        if self.last_pred is not None:
            (self.out / "metrics" / "state.json").write_text(json.dumps({"last_pred": str(self.last_pred)}))
        with open(self.out / "metrics" / "batches.jsonl", "a") as f:
            f.write(json.dumps(dict(batch_id=batch_id, ts=time.time(), n_candles=len(pdf), n_pred_minutes=n_min,
                                    process_ms=round((time.time() - t_begin) * 1000, 1))) + "\n")

    def _predict_ready(self, batch_id, t_recv):
        need = [f"{c}-USD" for c in self.coins]
        if any(p not in self.hist for p in need):
            return [], 0
        ready = min(self.hist[p].index.max() for p in need)
        first = max(self.hist[p].index.min() for p in need) + (self.W - 1) * MIN
        start = first if self.last_pred is None else max(first, self.last_pred + MIN)
        if start > ready:
            return [], 0
        rows, n_min = [], 0
        for m in pd.date_range(start, ready, freq="min"):
            made = False
            for coin in self.coins:
                frame = self._frame(coin, m)
                if frame is None:
                    continue
                res = self.predictor.predict(coin, frame)
                t_ready = time.time()
                infer = sum(r["infer_ms"] for r in res)
                last_close = float(frame["close"].iloc[-1])
                lat = (t_ready - t_recv) * 1000                       # nến tới -> có dự báo
                e2e = t_ready * 1000 - (m + MIN).timestamp() * 1000   # hết phút của nến -> có dự báo (ý nghĩa ở chế độ live)
                for r in res:
                    for k in range(self.H):
                        rows.append(dict(made_at=m, coin=coin, model=r["model"], strategy=r["strategy"], label=r["label"],
                                         step=k + 1, target_minute=m + (k + 1) * MIN, y_pred_scaled=float(r["scaled"][k]),
                                         y_pred_usd=float(r["usd"][k]), last_close_usd=last_close, infer_ms=infer,
                                         latency_ms=lat, e2e_ms=e2e, batch_id=batch_id))
                made = True
            if made:
                self.last_pred, n_min = m, n_min + 1
        return rows, n_min
