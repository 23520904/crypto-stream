"""Đổ dữ liệu vào topic benchmark (burst) để đo hiệu năng lặp lại được.
  python -m benchmark.replay --from-topic --max-messages 2000000      # phát lại tick thật đã thu trong Kafka
  python -m benchmark.replay --synthetic --minutes 1500 --ticks-per-min 120   # tick giả lập (để thử nhanh)"""
import argparse, json, os, time
import numpy as np
from confluent_kafka import Consumer, Producer
from common.config import load_config


def synthetic(prod, topic, minutes, tpm, pids):
    rng = np.random.default_rng(0)
    base = {"BTC-USD": 60000.0, "ETH-USD": 3000.0, "PAXG-USD": 2400.0}
    px = {p: base.get(p, 100.0) for p in pids}
    t0 = int(time.time() * 1000) - minutes * 60_000
    tid = {p: 1 for p in pids}
    n = 0
    for m in range(minutes):
        for i in range(tpm):
            ts = t0 + m * 60_000 + int(i * 60_000 / tpm)
            for p in pids:
                if p == "PAXG-USD" and i % 20:     # PAXG ít giao dịch
                    continue
                px[p] *= 1 + rng.normal(0, 4e-5)
                prod.produce(topic, key=p, value=json.dumps(dict(
                    product_id=p, trade_id=tid[p], sequence=tid[p], event_ts_ms=ts, price=round(px[p], 2),
                    last_size=float(rng.exponential(0.05)), best_bid=None, best_ask=None, volume_24h=None,
                    ingest_ts_ms=ts, source="synthetic")))
                tid[p] += 1
                n += 1
        prod.poll(0)
    return n


def from_topic(prod, bootstrap, src, dst, max_messages):
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"replay-{int(time.time())}",
                  "auto.offset.reset": "earliest", "enable.auto.commit": False})
    c.subscribe([src])
    n, idle = 0, 0
    while n < max_messages and idle < 10:
        m = c.poll(1.0)
        if m is None or m.error():
            idle += 1
            continue
        idle = 0
        prod.produce(dst, key=m.key(), value=m.value())
        n += 1
        if n % 10000 == 0:
            prod.poll(0)
    c.close()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--from-topic", action="store_true")
    ap.add_argument("--minutes", type=int, default=1500)
    ap.add_argument("--ticks-per-min", type=int, default=120)
    ap.add_argument("--max-messages", type=int, default=2_000_000)
    a = ap.parse_args()
    cfg = load_config()
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap"])
    prod = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20, "compression.type": "lz4",
                     "queue.buffering.max.messages": 1_000_000})
    dst = cfg["kafka"]["topics"]["bench"]
    pids = cfg["coinbase"]["products"] + ([cfg["coinbase"]["gold_proxy"]] if cfg["coinbase"].get("gold_proxy") else [])
    if a.synthetic:
        n = synthetic(prod, dst, a.minutes, a.ticks_per_min, pids)
    elif a.from_topic:
        n = from_topic(prod, bootstrap, cfg["kafka"]["topics"]["ticks"], dst, a.max_messages)
    else:
        ap.error("chọn --synthetic hoặc --from-topic")
    prod.flush(60)
    print(f"Đã nạp {n:,} tick vào topic '{dst}'")
