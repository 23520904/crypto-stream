"""Batch layer: huấn luyện/đánh giá BD-LSTM, Conv-LSTM, MLP + baseline (persistence, ARIMA) trên dữ liệu lịch sử.

Ví dụ:
  python -m batch.train_batch --exp 1 --suite m2                 # M2: tái hiện bài báo [1] (BTC)
  python -m batch.train_batch --exp 1 --suite grid               # đủ 3 chiến lược x 2 mô hình x BTC/ETH
  python -m batch.train_batch --exp 2 --suite grid               # Experiment 2 (train có COVID-19)
  python -m batch.train_batch --exp deploy --suite deploy        # huấn luyện + XUẤT mô hình cho speed layer
  python -m batch.train_batch --exp 1 --coin BTC --strategy univariate --model BD-LSTM --search   # chọn hidden units
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime
import numpy as np
import torch
from scipy import stats

from common.config import load_config, path_of, ensure
from common.features import prepare_experiment, CLOSE_I
from common.training import train_run, predict, rmse_steps
from common.baselines import persistence, fit_arima_params, arima_forecast_fixed
from common.models import artifact_name


def agg(mat: np.ndarray):
    """mat: (runs, k) -> mean, nửa-độ-rộng khoảng tin cậy 95% (t-Student)."""
    mat = np.asarray(mat, float)
    r = mat.shape[0]
    mean = mat.mean(0)
    ci = stats.t.ppf(0.975, r - 1) * mat.std(0, ddof=1) / np.sqrt(r) if r > 1 else np.zeros_like(mean)
    return mean, ci


def out_path(cfg, exp, coin, strategy, model):
    d = ensure(path_of(cfg, "results", "batch", f"exp{exp}"))
    return d / f"{coin}__{strategy}__{model.replace('(', '').replace(')', '').replace(',', '')}.json"


def run_config(coin, strategy, model, exp, cfg, runs, epochs=None, hidden=None, export=False, device="cpu"):
    b = cfg["batch"]
    hidden = hidden or b["hidden"][model]
    prep = prepare_experiment(coin, strategy, exp, cfg)
    tr_rm, te_rm, vals, eps, states = [], [], [], [], []
    t0 = time.time()
    for r in range(runs):
        m, val, ep = train_run(model, prep["Xtr"], prep["ytr"], hidden, b, seed=1000 + r, device=device, epochs=epochs)
        tr_rm.append(rmse_steps(prep["ytr"], predict(m, prep["Xtr"])))
        te_rm.append(rmse_steps(prep["yte"], predict(m, prep["Xte"])))
        vals.append(val); eps.append(ep); states.append(m)
        print(f"  [{coin}|{strategy}|{model}] run {r + 1}/{runs}  epochs={ep}  val={val:.5f}  test_mean={te_rm[-1].mean():.4f}")
    te_rm, tr_rm = np.array(te_rm), np.array(tr_rm)
    mean, ci = agg(te_rm)
    tm, tci = agg(te_rm.mean(1, keepdims=True))
    trm, _ = agg(tr_rm.mean(1, keepdims=True))
    res = dict(coin=coin, strategy=strategy, model=model, experiment=str(exp), hidden=hidden, runs=runs,
               dates=prep["dates"], per_step=dict(mean=mean.tolist(), ci=ci.tolist()),
               test_mean=dict(mean=float(tm[0]), ci=float(tci[0])), train_mean=float(trm[0]),
               epochs_mean=float(np.mean(eps)), seconds=round(time.time() - t0, 1),
               scaler_fit=cfg["batch"]["experiments"][str(exp)].get("scaler_fit", b["scaler_fit"]),
               created=datetime.now().isoformat(timespec="seconds"))
    out_path(cfg, exp, coin, strategy, model).write_text(json.dumps(res, indent=2))
    if export:   # xuất lần chạy có val-loss thấp nhất (KHÔNG chọn theo test để tránh rò rỉ)
        best = int(np.argmin(vals))
        art = ensure(path_of(cfg, "artifacts"))
        meta = dict(coin=coin, strategy=strategy, model=model, hidden=hidden, n_features=prep["n_features"],
                    window=prep["W"], horizon=prep["H"], experiment=str(exp), dates=prep["dates"])
        torch.save({"state_dict": states[best].state_dict(), "meta": meta}, art / artifact_name(coin, strategy, model))
        prep["scaler"].save(art / f"{coin}_scaler.json")
        print(f"  -> đã xuất {artifact_name(coin, strategy, model)} (run {best + 1})")
    print(f"==> {coin}|{strategy}|{model}: test RMSE mean = {tm[0]:.4f} ± {tci[0]:.4f}")
    return res


def run_baselines(coin, exp, cfg):
    """persistence và ARIMA(1,0,1) trên đúng cửa sổ test của chiến lược univariate."""
    prep = prepare_experiment(coin, "univariate", exp, cfg)
    A, n_train, H = prep["A"], prep["n_train"], prep["H"]
    close, N = A[:, CLOSE_I], len(prep["yte"])
    order = tuple(cfg["streaming"]["arima_order"])
    hist_n = cfg["streaming"]["arima_min_history"]
    p_pers = np.array([persistence(close[n_train + i - 1], H) for i in range(N)])
    params = fit_arima_params(close[:n_train], order)
    p_arima = np.array([arima_forecast_fixed(close[max(0, n_train + i - hist_n):n_train + i], params, H, order) for i in range(N)])
    for name, p in (("persistence", p_pers), ("ARIMA(1,0,1)", p_arima)):
        rm = rmse_steps(prep["yte"], p)
        res = dict(coin=coin, strategy="univariate", model=name, experiment=str(exp), runs=1, dates=prep["dates"],
                   per_step=dict(mean=rm.tolist(), ci=[0.0] * H), test_mean=dict(mean=float(rm.mean()), ci=0.0),
                   created=datetime.now().isoformat(timespec="seconds"))
        out_path(cfg, exp, coin, "univariate", name).write_text(json.dumps(res, indent=2))
        print(f"==> {coin}|{name}: test RMSE mean = {rm.mean():.4f}")


def hparam_search(coin, strategy, model, exp, cfg, runs=5, device="cpu"):
    """Chọn số hidden unit theo VAL loss (không dùng test)."""
    prep = prepare_experiment(coin, strategy, exp, cfg)
    rows = []
    for h in (20, 50, 100):
        vals = [train_run(model, prep["Xtr"], prep["ytr"], h, cfg["batch"], 2000 + r, device)[1] for r in range(runs)]
        rows.append(dict(hidden=h, val_mse=float(np.mean(vals))))
        print(f"  hidden={h}: val_mse={np.mean(vals):.6f}")
    best = min(rows, key=lambda r: r["val_mse"])
    d = ensure(path_of(cfg, "results", "batch", "hparam"))
    (d / f"{coin}__{strategy}__{model}.json").write_text(json.dumps(dict(rows=rows, best=best), indent=2))
    print("hidden tốt nhất:", best)


SUITES = {
    "m2": [("BTC", "univariate", "BD-LSTM"), ("BTC", "multivariate_star", "Conv-LSTM"), ("BTC", "univariate", "MLP")],
    "grid": [(c, s, m) for c in ("BTC", "ETH") for s in ("univariate", "multivariate", "multivariate_star")
             for m in ("BD-LSTM", "Conv-LSTM")] + [(c, "univariate", "MLP") for c in ("BTC", "ETH")],
    "deploy": [(c, s, m) for c in ("BTC", "ETH")
               for s, m in (("univariate", "BD-LSTM"), ("multivariate_star", "Conv-LSTM"), ("univariate", "MLP"))],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="1", help="1 | 2 | deploy")
    ap.add_argument("--suite", choices=list(SUITES))
    ap.add_argument("--coin", choices=["BTC", "ETH"])
    ap.add_argument("--strategy", choices=["univariate", "multivariate", "multivariate_star"])
    ap.add_argument("--model", choices=["BD-LSTM", "Conv-LSTM", "MLP"])
    ap.add_argument("--runs", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config()
    ec = cfg["batch"]["experiments"][a.exp]
    runs = a.runs or ec.get("runs") or cfg["batch"]["runs"]
    export = a.exp == "deploy"
    if a.search:
        hparam_search(a.coin, a.strategy, a.model, a.exp, cfg, device=a.device)
        return
    jobs = SUITES[a.suite] if a.suite else [(a.coin, a.strategy, a.model)]
    if not a.suite and not (a.coin and a.strategy and a.model):
        ap.error("Cần --suite hoặc đủ --coin --strategy --model")
    for coin, strat, model in jobs:
        run_config(coin, strat, model, a.exp, cfg, runs, a.epochs, a.hidden, export, a.device)
    if not a.no_baselines and a.exp != "deploy":
        for coin in sorted({j[0] for j in jobs}):
            run_baselines(coin, a.exp, cfg)


if __name__ == "__main__":
    main()
