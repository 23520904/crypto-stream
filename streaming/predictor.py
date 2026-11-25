"""Suy luận 5 bước cho từng coin: BD-LSTM, Conv-LSTM, MLP + baseline persistence, ARIMA(1,0,1)."""
from __future__ import annotations
import time
import numpy as np
import torch
from common.config import path_of
from common.features import MinMax, COLUMNS, STRATEGIES, CLOSE_I, strat_idx
from common.models import build_model, artifact_name
from common.baselines import persistence, arima_forecast

SPECS = [("BD-LSTM", "univariate"), ("Conv-LSTM", "multivariate_star"), ("MLP", "univariate")]


def label_of(model, strategy):
    return f"{model} [{strategy}]"


class Predictor:
    def __init__(self, cfg, coins):
        self.cfg, self.coins = cfg, coins
        self.W, self.H = cfg["batch"]["window"], cfg["batch"]["horizon"]
        sc = cfg["streaming"]
        self.order, self.hist_n, self.hist_min = tuple(sc["arima_order"]), sc["arima_history"], sc["arima_min_history"]
        art = path_of(cfg, "artifacts")
        self.scalers, self.models = {}, {}
        for coin in coins:
            sp = art / f"{coin}_scaler.json"
            if not sp.exists():
                raise FileNotFoundError(f"Thiếu {sp}. Chạy trước: python -m batch.train_batch --exp deploy --suite deploy")
            self.scalers[coin] = MinMax.load(sp)
            for name, strat in SPECS:
                fp = art / artifact_name(coin, strat, name)
                if not fp.exists():
                    raise FileNotFoundError(f"Thiếu mô hình {fp}")
                blob = torch.load(fp, map_location="cpu")
                m = blob["meta"]
                net = build_model(name, m["n_features"], m["window"], m["horizon"], m["hidden"])
                net.load_state_dict(blob["state_dict"])
                net.eval()
                self.models[(coin, name, strat)] = net

    def predict(self, coin: str, frame) -> list[dict]:
        """frame: DataFrame cột COLUMNS (đơn vị USD), mới nhất ở cuối, >= W dòng."""
        sc = self.scalers[coin]
        A = sc.transform(frame)
        out = []
        with torch.no_grad():
            for (c, name, strat), net in self.models.items():
                if c != coin:
                    continue
                x = torch.from_numpy(A[-self.W:, strat_idx(strat)].astype(np.float32)).unsqueeze(0)
                t0 = time.perf_counter()
                y = net(x)[0].numpy()
                out.append(dict(model=name, strategy=strat, label=label_of(name, strat), scaled=y,
                                usd=sc.inv("close", y), infer_ms=(time.perf_counter() - t0) * 1000))
        t0 = time.perf_counter()
        y = persistence(A[-1, CLOSE_I], self.H)
        out.append(dict(model="persistence", strategy="-", label="persistence", scaled=y, usd=sc.inv("close", y),
                        infer_ms=(time.perf_counter() - t0) * 1000))
        if len(A) >= self.hist_min:
            t0 = time.perf_counter()
            y = arima_forecast(A[-self.hist_n:, CLOSE_I], self.H, self.order)
            out.append(dict(model="ARIMA(1,0,1)", strategy="-", label="ARIMA(1,0,1)", scaled=y, usd=sc.inv("close", y),
                            infer_ms=(time.perf_counter() - t0) * 1000))
        return out
