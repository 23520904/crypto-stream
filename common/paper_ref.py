"""Kết quả Experiment 1 (pre-COVID-19), RMSE test mean (Table 10 bài báo [1])."""
PAPER_REF_EXP1 = {
    ("BTC", "univariate", "BD-LSTM"): 0.0327,
    ("BTC", "univariate", "Conv-LSTM"): 0.0366,
    ("BTC", "multivariate", "BD-LSTM"): 0.0394,
    ("BTC", "multivariate", "Conv-LSTM"): 0.0771,
    ("BTC", "multivariate_star", "BD-LSTM"): 0.0236,
    ("BTC", "multivariate_star", "Conv-LSTM"): 0.0200,
    ("ETH", "univariate", "BD-LSTM"): 0.0388,
    ("ETH", "univariate", "Conv-LSTM"): 0.0519,
    ("ETH", "multivariate", "BD-LSTM"): 0.1664,
    ("ETH", "multivariate", "Conv-LSTM"): 0.1346,
    ("ETH", "multivariate_star", "BD-LSTM"): 0.0241,
    ("ETH", "multivariate_star", "Conv-LSTM"): 0.0210,
}
