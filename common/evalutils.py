"""Ghép dự báo với giá thực tế khi giá đến (online evaluation) — dùng chung cho CLI và dashboard."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import path_of
from .features import MinMax


def read_dir(d, filters=None, columns=None) -> pd.DataFrame:
    d = Path(d)
    if not d.exists() or not any(d.glob("*.parquet")):
        return pd.DataFrame()
    return pd.read_parquet(d, filters=filters, columns=columns)


def load_scalers(cfg, coins) -> dict:
    return {c: MinMax.load(path_of(cfg, "artifacts", f"{c}_scaler.json")) for c in coins}


def actual_series(candles: pd.DataFrame, pid: str) -> pd.Series:
    c = candles[candles["product_id"] == pid].drop_duplicates("minute", keep="last").set_index("minute").sort_index()["close"]
    if c.empty:
        return c
    return c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="min")).ffill()


def join_predictions_actuals(pred: pd.DataFrame, candles: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    parts = []
    if pred.empty or candles.empty:
        return pd.DataFrame()
    for coin, g in pred.groupby("coin"):
        act = actual_series(candles, f"{coin}-USD")
        if act.empty or coin not in scalers:
            continue
        g = g.copy()
        g["actual_usd"] = g["target_minute"].map(act)
        g = g.dropna(subset=["actual_usd"])
        g["actual_scaled"] = scalers[coin].scale("close", g["actual_usd"].to_numpy())
        g["err"] = g["y_pred_scaled"] - g["actual_scaled"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def rmse_by_step(j: pd.DataFrame) -> pd.DataFrame:
    if j.empty:
        return pd.DataFrame()
    g = j.assign(sq=j["err"] ** 2).groupby(["coin", "label", "step"]).agg(mse=("sq", "mean"), n=("sq", "size")).reset_index()
    g["rmse"] = np.sqrt(g["mse"])
    return g.drop(columns="mse")


def rmse_mean(j: pd.DataFrame) -> pd.DataFrame:
    s = rmse_by_step(j)
    if s.empty:
        return s
    m = s.groupby(["coin", "label"]).agg(rmse_mean=("rmse", "mean"), n=("n", "min")).reset_index()
    pers = m[m["label"] == "persistence"].set_index("coin")["rmse_mean"]
    m["vs_persistence_%"] = m.apply(lambda r: 100 * (pers.get(r["coin"], np.nan) - r["rmse_mean"]) / pers.get(r["coin"], np.nan)
                                    if r["coin"] in pers.index else np.nan, axis=1)
    return m.sort_values(["coin", "rmse_mean"])


def rmse_over_time(j: pd.DataFrame, freq="1h") -> pd.DataFrame:
    if j.empty:
        return pd.DataFrame()
    t = j.assign(sq=j["err"] ** 2, t=j["made_at"].dt.floor(freq))
    g = t.groupby(["coin", "label", "t"])["sq"].mean().reset_index()
    g["rmse"] = np.sqrt(g["sq"])
    return g.drop(columns="sq")
