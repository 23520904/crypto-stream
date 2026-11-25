import pytest
import pandas as pd
import numpy as np
from common.config import load_config
from common.evalutils import read_dir, load_scalers
import dashboard.app as app

def test_compute_model_profiles():
    cfg = load_config()
    cand = read_dir('data/stream/candles')
    pred = read_dir('data/stream/predictions')
    scalers = load_scalers(cfg, ['BTC', 'ETH'])
    
    profiles = app.compute_model_profiles('BTC', cand, pred, scalers)
    assert len(profiles) == 5
    expected_labels = {
        'BD-LSTM [univariate]',
        'Conv-LSTM [multivariate_star]',
        'MLP [univariate]',
        'ARIMA(1,0,1)',
        'persistence'
    }
    actual_labels = {p['label'] for p in profiles}
    assert actual_labels == expected_labels
    
    for p in profiles:
        assert 'params' in p
        assert 'infer_p50' in p
        assert 'rmse_minmax' in p
        assert 'hit_rate' in p
        assert 'family' in p
        assert 'input_shape' in p

def test_train_studio_model_simulation():
    class MockProgress:
        def progress(self, val): pass
        def empty(self): pass

    class MockStatus:
        def markdown(self, val): pass
        def empty(self): pass

    res = app.train_studio_model(
        sel_coin='BTC',
        sel_strategy='univariate',
        sel_model='MLP',
        sel_hidden=15,
        sel_lr=0.005,
        sel_epochs=2,
        sel_batch_size=32,
        export_deploy=False,
        progress_bar=MockProgress(),
        status_text=MockStatus(),
    )
    assert res is not None
    assert 'new_test_rmse' in res
    assert 'history' in res
    assert len(res['history']['epoch']) == 2
    assert res['new_params'] > 0
    assert res['exported_file'] is None
