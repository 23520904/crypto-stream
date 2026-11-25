"""M6 - Dashboard Streamlit + Plotly:   streamlit run dashboard/app.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.config import load_config, path_of
from common.evalutils import (read_dir, load_scalers, join_predictions_actuals, rmse_by_step, rmse_mean,
                              rmse_over_time, actual_series)
from common.paper_ref import PAPER_REF_EXP1

cfg = load_config()
st.set_page_config(page_title="Crypto streaming forecast", layout="wide")
st.title("Dự báo giá BTC / ETH thời gian thực — Coinbase → Kafka → Spark → BD-LSTM / Conv-LSTM")

sd = Path(st.sidebar.text_input("Thư mục stream", str(path_of(cfg, "stream"))))
hours = st.sidebar.slider("Cửa sổ xem (giờ)", 1, 72, 6)
refresh = st.sidebar.slider("Tự làm mới (giây)", 5, 60, 10)
coins = [p.split("-")[0] for p in cfg["coinbase"]["products"]]
coin = st.sidebar.selectbox("Coin", coins)


@st.cache_data(ttl=5)
def load(sd_str, hrs):
    cand = read_dir(f"{sd_str}/candles")
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    tmax = cand["minute"].max()
    since = tmax - pd.Timedelta(hours=hrs)
    cand = cand[cand["minute"] >= since - pd.Timedelta(minutes=10)]
    pred = read_dir(f"{sd_str}/predictions", filters=[("made_at", ">=", since)])
    return cand, pred


@st.cache_data(ttl=60)
def scalers():
    try:
        return load_scalers(cfg, coins)
    except Exception:
        return {}


def jsonl(fp):
    fp = Path(fp)
    return pd.read_json(fp, lines=True) if fp.exists() and fp.stat().st_size else pd.DataFrame()


tabs = st.tabs(["Dự báo vs thực tế", "RMSE online", "Hiệu năng", "Tái hiện bài báo [1]"])


@st.fragment(run_every=refresh)
def tab_live():
    cand, pred = load(str(sd), hours)
    if cand.empty or pred.empty:
        st.info("Chưa có dữ liệu. Hãy chạy producer + speed layer.")
        return
    labels = sorted(pred["label"].unique())
    c1, c2 = st.columns(2)
    models = c1.multiselect("Mô hình", labels, default=[l for l in labels if "[" in l][:2])
    step = c2.select_slider("Bước dự báo (phút)", [1, 2, 3, 4, 5], value=1)
    act = actual_series(cand, f"{coin}-USD")
    act = act[act.index >= act.index.max() - pd.Timedelta(hours=hours)]
    p = pred[(pred["coin"] == coin) & (pred["step"] == step)]
    fig = go.Figure(go.Scatter(x=act.index, y=act.values, name="Thực tế (đóng cửa 1 phút)", line=dict(color="black")))
    for l in models:
        q = p[p["label"] == l].sort_values("target_minute")
        fig.add_trace(go.Scatter(x=q["target_minute"], y=q["y_pred_usd"], name=f"{l} bước {step}", mode="lines"))
    fig.update_layout(height=420, title=f"{coin}-USD: dự báo bước {step} so với thực tế", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

    last = pred[(pred["coin"] == coin)]
    if not last.empty:
        m0 = last["made_at"].max()
        fan = go.Figure(go.Scatter(x=act.index[-60:], y=act.values[-60:], name="Thực tế", line=dict(color="black")))
        for l in models:
            q = last[(last["made_at"] == m0) & (last["label"] == l)].sort_values("step")
            fan.add_trace(go.Scatter(x=q["target_minute"], y=q["y_pred_usd"], name=l, mode="lines+markers"))
        fan.update_layout(height=320, title=f"Dự báo 5 bước gần nhất (phát lúc {m0:%H:%M} UTC)", yaxis_title="USD")
        st.plotly_chart(fan, use_container_width=True)


@st.fragment(run_every=refresh)
def tab_rmse():
    cand, pred = load(str(sd), hours)
    sc = scalers()
    if cand.empty or pred.empty or not sc:
        st.info("Chưa đủ dữ liệu / chưa có artifacts.")
        return
    j = join_predictions_actuals(pred, cand, sc)
    if j.empty:
        st.info("Chưa có giá thực tế khớp dự báo.")
        return
    m = rmse_mean(j)
    st.subheader("RMSE trung bình 5 bước (thang min-max)")
    st.dataframe(m[m["coin"] == coin].round(5), use_container_width=True)
    s = rmse_by_step(j)
    s = s[s["coin"] == coin]
    st.plotly_chart(px.line(s, x="step", y="rmse", color="label", markers=True, title="RMSE theo bước 1–5"), use_container_width=True)
    t = rmse_over_time(j, "1h" if hours > 6 else "10min")
    t = t[t["coin"] == coin]
    st.plotly_chart(px.line(t, x="t", y="rmse", color="label", title="RMSE theo thời gian"), use_container_width=True)


@st.fragment(run_every=refresh)
def tab_perf():
    cand, pred = load(str(sd), hours)
    if not pred.empty:
        u = pred.drop_duplicates(["made_at", "coin"])
        a, b, c, d = st.columns(4)
        a.metric("Latency p50 (ms)", f"{u['latency_ms'].quantile(.5):,.0f}")
        b.metric("Latency p95 (ms)", f"{u['latency_ms'].quantile(.95):,.0f}")
        c.metric("Suy luận p50 (ms)", f"{u['infer_ms'].quantile(.5):,.1f}")
        d.metric("Suy luận p95 (ms)", f"{u['infer_ms'].quantile(.95):,.1f}")
        st.plotly_chart(px.line(u.sort_values("made_at"), x="made_at", y=["latency_ms", "infer_ms"], title="Độ trễ theo thời gian"),
                        use_container_width=True)
    pg = jsonl(sd / "metrics" / "progress.jsonl")
    if not pg.empty:
        pg["timestamp"] = pd.to_datetime(pg["timestamp"])
        st.plotly_chart(px.line(pg.tail(2000), x="timestamp", y=["inputRowsPerSecond", "processedRowsPerSecond"],
                                title="Thông lượng Spark (tick/giây)"), use_container_width=True)
    ps = path_of(cfg, "metrics", "producer_status.json")
    if ps.exists():
        s = json.loads(ps.read_text())
        cu = s.get("cumulative", {})
        st.subheader("Producer (M1)")
        a, b, c, d = st.columns(4)
        a.metric("Kết nối", "OK" if s.get("connected") else "MẤT")
        b.metric("Số ngày chạy", f"{cu.get('running_days', 0):.2f} / 45")
        c.metric("Gap / trade mất", f"{cu.get('gaps', 0)} / {cu.get('missed_trades', 0)}")
        d.metric("Reconnect", cu.get("reconnects", 0))
        st.json(s, expanded=False)
    bf = path_of(cfg, "results", "benchmark", "summary.csv")
    if bf.exists():
        st.subheader("Benchmark 1 / 2 / 4 core (M5)")
        bd = pd.read_csv(bf)
        st.dataframe(bd, use_container_width=True)
        st.plotly_chart(px.bar(bd, x="cores", y="throughput_ticks_s", title="Thông lượng theo số core"), use_container_width=True)


def tab_repro():
    rows = []
    for fp in sorted(path_of(cfg, "results", "batch", "exp1").glob("*.json")):
        r = json.loads(fp.read_text())
        ref = PAPER_REF_EXP1.get((r["coin"], r["strategy"], r["model"]))
        rows.append(dict(coin=r["coin"], strategy=r["strategy"], model=r["model"], ours=round(r["test_mean"]["mean"], 4),
                         ci95=round(r["test_mean"]["ci"], 4), paper=ref, runs=r["runs"]))
    if not rows:
        st.info("Chưa có kết quả batch. Chạy: python -m batch.train_batch --exp 1 --suite m2")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


with tabs[0]:
    tab_live()
with tabs[1]:
    tab_rmse()
with tabs[2]:
    tab_perf()
with tabs[3]:
    tab_repro()
