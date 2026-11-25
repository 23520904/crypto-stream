"""Baseline: persistence và ARIMA(1,0,1) — dùng chung cho batch và speed layer."""
from __future__ import annotations
import warnings
import numpy as np


def persistence(last_value: float, H: int) -> np.ndarray:
    return np.repeat(float(last_value), H)


def arima_forecast(series, H: int, order=(1, 0, 1)) -> np.ndarray:
    """Fit lại trên `series` rồi dự báo H bước; lỗi/không hội tụ => rơi về persistence."""
    from statsmodels.tsa.arima.model import ARIMA
    s = np.asarray(series, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f = np.asarray(ARIMA(s, order=order).fit().forecast(H))
            if f.shape != (H,) or not np.all(np.isfinite(f)):
                raise ValueError
            return f
        except Exception:
            return persistence(s[-1], H)


def fit_arima_params(series, order=(1, 0, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(np.asarray(series, float), order=order).fit().params


def arima_forecast_fixed(history, params, H: int, order=(1, 0, 1)) -> np.ndarray:
    """Dùng tham số đã fit trên train, chỉ cập nhật trạng thái bằng `history` (nhanh, không rò rỉ test)."""
    from statsmodels.tsa.arima.model import ARIMA
    h = np.asarray(history, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f = np.asarray(ARIMA(h, order=order).smooth(params).forecast(H))
            if not np.all(np.isfinite(f)):
                raise ValueError
            return f
        except Exception:
            return persistence(h[-1], H)
