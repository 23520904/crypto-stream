"""Nạp dữ liệu lịch sử (daily) cho batch layer: BTC, ETH, vàng."""
from __future__ import annotations
import pandas as pd
from .config import ROOT, path_of, ensure

PARTNER = {"BTC": "ETH", "ETH": "BTC"}   # coin tương quan nhất (theo [1])


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df["date"] = df["date"].dt.normalize()
    if "volume" not in df:
        df["volume"] = 0.0
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    return df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)


def download_yf(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"Không tải được {ticker} từ Yahoo Finance")
    return _normalize(df.reset_index())


def load_asset(name: str, cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """name ∈ {BTC, ETH, GOLD}. Cache CSV tại data/raw/{name}.csv."""
    raw = ensure(path_of(cfg, "raw"))
    fp = raw / f"{name}.csv"
    if fp.exists() and not refresh:
        return _normalize(pd.read_csv(fp))
    df = download_yf(cfg["data"]["tickers"][name], cfg["data"]["start"])
    k = (cfg["data"].get("kaggle") or {}).get(name)
    if k and (ROOT / k).exists():   # nối thêm lịch sử Kaggle trước ngày đầu của Yahoo
        kd = _normalize(pd.read_csv(ROOT / k))
        df = pd.concat([kd[kd.index < df.index.min()], df]).sort_index()
    df.reset_index().rename(columns={"index": "date"}).to_csv(fp, index=False)
    return df


def build_frame(target: str, cfg: dict) -> pd.DataFrame:
    """DataFrame ngày: open, high, low, close (coin), gold (nội suy tuyến tính ra ngày lịch), partner_close."""
    t = load_asset(target, cfg)
    p = load_asset(PARTNER[target], cfg)
    g = load_asset("GOLD", cfg)["close"]
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D")).interpolate("linear")
    df = pd.DataFrame({
        "open": t["open"], "high": t["high"], "low": t["low"], "close": t["close"],
        "gold": g.reindex(t.index).ffill(),
        "partner_close": p["close"].reindex(t.index),
    }).dropna()
    return df
