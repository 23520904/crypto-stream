import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from common.config import load_config
from common.features import COLUMNS, STRATEGIES, MinMax, make_windows, strat_idx, CLOSE_I
from common.models import MLP, BDLSTM, ConvLSTM, build_model
from common.baselines import persistence, fit_arima_params, arima_forecast, arima_forecast_fixed
from streaming.predictor import Predictor, label_of


def test_baselines_persistence():
    last_price = 65432.1
    H = 5
    pred = persistence(last_price, H)
    assert len(pred) == H
    assert np.all(pred == last_price)


def test_baselines_arima():
    rng = np.random.default_rng(42)
    # Tạo chuỗi thời gian có xu hướng
    series = np.cumsum(rng.normal(0, 1, 100)) + 50.0
    order = (1, 0, 1)
    H = 5
    
    # Test fitting
    params = fit_arima_params(series, order)
    assert isinstance(params, np.ndarray) and len(params) > 0
    assert np.all(np.isfinite(params))
    
    # Test fixed forecasting
    forecast_fixed = arima_forecast_fixed(series, params, H, order)
    assert len(forecast_fixed) == H
    assert not np.any(np.isnan(forecast_fixed))
    
    # Test dynamic forecasting
    forecast_dyn = arima_forecast(series, H, order)
    assert len(forecast_dyn) == H
    assert not np.any(np.isnan(forecast_dyn))


def test_minmax_save_and_load(tmp_path):
    df = pd.DataFrame({
        "open": [10.0, 20.0, 30.0],
        "high": [15.0, 25.0, 35.0],
        "low": [8.0, 18.0, 28.0],
        "close": [12.0, 22.0, 32.0],
        "gold": [1800.0, 1850.0, 1900.0],
        "partner_close": [100.0, 200.0, 300.0],
    })
    scaler = MinMax.fit(df)
    save_file = tmp_path / "test_scaler.json"
    scaler.save(save_file)
    assert save_file.exists()
    
    loaded = MinMax.load(save_file)
    assert loaded.columns == scaler.columns
    assert np.allclose(loaded.lo, scaler.lo)
    assert np.allclose(loaded.hi, scaler.hi)
    assert loaded.last["close"] == 32.0


def test_conv_lstm_edge_shapes():
    B, W, F, H = 2, 6, 6, 5
    model = ConvLSTM(n_features=F, window=W, horizon=H, hidden=20, sub_len=3, kernel=2)
    x = torch.randn(B, W, F)
    out = model(x)
    assert out.shape == (B, H)
    assert not torch.isnan(out).any()


def test_strat_idx_completeness():
    for name, cols in STRATEGIES.items():
        indices = strat_idx(name)
        assert len(indices) == len(cols)
        for col_name, idx in zip(cols, indices):
            assert COLUMNS[idx] == col_name


def test_predictor_with_artifacts():
    cfg = load_config()
    art_dir = Path(cfg["paths"]["artifacts"])
    if not (art_dir / "BTC_scaler.json").exists():
        pytest.skip("Chưa có artifacts deploy")
        
    predictor = Predictor(cfg, ["BTC"])
    # Tạo frame giả lập đủ W điểm (6 dòng)
    W = cfg["batch"]["window"]
    mock_frame = pd.DataFrame({
        "open": np.linspace(60000, 61000, 10),
        "high": np.linspace(60500, 61500, 10),
        "low": np.linspace(59500, 60500, 10),
        "close": np.linspace(60200, 61200, 10),
        "gold": np.linspace(2400, 2450, 10),
        "partner_close": np.linspace(3000, 3100, 10),
    })
    
    preds = predictor.predict("BTC", mock_frame)
    assert len(preds) >= 4   # BD-LSTM, Conv-LSTM, MLP, persistence
    for p in preds:
        assert "model" in p
        assert "y_pred" in p or "scaled" in p
        assert len(p["scaled"]) == cfg["batch"]["horizon"]
