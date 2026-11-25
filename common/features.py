"""Chuẩn hoá min-max, cửa sổ trượt, 3 chiến lược đặc trưng, chia train/test theo thời gian."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

COLUMNS = ["open", "high", "low", "close", "gold", "partner_close"]
STRATEGIES = {
    "univariate": ["close"],
    "multivariate": ["open", "high", "low", "close", "gold"],
    "multivariate_star": ["open", "high", "low", "close", "gold", "partner_close"],
}
CLOSE_I = COLUMNS.index("close")


def strat_idx(strategy: str) -> list[int]:
    return [COLUMNS.index(c) for c in STRATEGIES[strategy]]


@dataclass
class MinMax:
    columns: list
    lo: np.ndarray
    hi: np.ndarray
    last: dict

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "MinMax":
        a = df[COLUMNS].to_numpy(float)
        return cls(list(COLUMNS), a.min(0), a.max(0), {c: float(df[c].iloc[-1]) for c in COLUMNS})

    def _rng(self) -> np.ndarray:
        r = (self.hi - self.lo).astype(float)
        r[r == 0] = 1.0
        return r

    def transform(self, df) -> np.ndarray:
        a = df[COLUMNS].to_numpy(float) if hasattr(df, "columns") else np.asarray(df, float)
        return (a - self.lo) / self._rng()

    def scale(self, col: str, x):
        i = self.columns.index(col)
        return (np.asarray(x, float) - self.lo[i]) / self._rng()[i]

    def inv(self, col: str, x):
        i = self.columns.index(col)
        return np.asarray(x, float) * self._rng()[i] + self.lo[i]

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"columns": self.columns, "min": self.lo.tolist(),
                                          "max": self.hi.tolist(), "last": self.last}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "MinMax":
        j = json.loads(Path(path).read_text())
        return cls(j["columns"], np.array(j["min"]), np.array(j["max"]), j.get("last", {}))


def make_windows(arr: np.ndarray, idx: list[int], W: int, H: int):
    """X[i] = arr[i:i+W, idx]; y[i] = close[i+W : i+W+H]  (chiến lược direct, đa đầu ra)."""
    n = arr.shape[0]
    N = n - W - H + 1
    if N <= 0:
        raise ValueError("Không đủ dữ liệu để tạo cửa sổ")
    Xw = sliding_window_view(arr[:, idx], W, axis=0)            # (n-W+1, nf, W)
    X = np.transpose(Xw[:N], (0, 2, 1)).copy()
    y = sliding_window_view(arr[:, CLOSE_I], H)[W:W + N].copy()
    return X.astype(np.float32), y.astype(np.float32)


def prepare_experiment(coin: str, strategy: str, exp: str, cfg: dict, df: pd.DataFrame | None = None) -> dict:
    from .data import build_frame
    b = cfg["batch"]
    W, H = b["window"], b["horizon"]
    ec = b["experiments"][str(exp)]
    df = build_frame(coin, cfg) if df is None else df
    if ec["mode"] == "ratio":
        d = df if not ec.get("end") else df.loc[:ec["end"]]
        n_train = int(len(d) * ec["train_ratio"])
    else:
        d = df.loc[ec["train"][0]:ec["test"][1]]
        n_train = int((d.index <= pd.Timestamp(ec["train"][1])).sum())
    fit_mode = ec.get("scaler_fit", b["scaler_fit"])
    scaler = MinMax.fit(d.iloc[:n_train] if fit_mode == "train" else d)
    A = scaler.transform(d)
    idx = strat_idx(strategy)
    Xtr, ytr = make_windows(A[:n_train], idx, W, H)
    Xte, yte = make_windows(A[n_train - W:], idx, W, H)   # ngữ cảnh W điểm cuối train, mục tiêu hoàn toàn nằm trong test
    return dict(coin=coin, strategy=strategy, exp=str(exp), scaler=scaler, A=A, n_train=n_train,
                Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_features=len(idx), W=W, H=H,
                dates=dict(start=str(d.index[0].date()), train_end=str(d.index[n_train - 1].date()),
                           test_start=str(d.index[n_train].date()), end=str(d.index[-1].date())))
