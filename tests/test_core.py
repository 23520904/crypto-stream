import numpy as np
import pandas as pd
import torch
from common.features import COLUMNS, STRATEGIES, MinMax, make_windows, strat_idx, CLOSE_I
from common.models import build_model


def _df(n=60):
    r = np.random.default_rng(0)
    return pd.DataFrame({c: np.cumsum(r.normal(0, 1, n)) + 100 for c in COLUMNS})


def test_windows_direct_targets():
    A = np.arange(40 * 6, dtype=float).reshape(40, 6)
    X, y = make_windows(A, strat_idx("univariate"), 6, 5)
    assert X.shape == (30, 6, 1) and y.shape == (30, 5)
    assert np.allclose(y[0], A[6:11, CLOSE_I]) and np.allclose(X[0, :, 0], A[0:6, CLOSE_I])


def test_scaler_roundtrip():
    d = _df()
    s = MinMax.fit(d)
    A = s.transform(d)
    assert A.min() >= -1e-9 and A.max() <= 1 + 1e-9
    assert np.allclose(s.inv("close", A[:, CLOSE_I]), d["close"].to_numpy())


def test_model_shapes():
    for name, strat in (("MLP", "univariate"), ("BD-LSTM", "univariate"), ("Conv-LSTM", "multivariate_star"), ("Conv-LSTM", "univariate")):
        nf = len(STRATEGIES[strat])
        m = build_model(name, nf, 6, 5, 20)
        assert m(torch.randn(4, 6, nf)).shape == (4, 5)
