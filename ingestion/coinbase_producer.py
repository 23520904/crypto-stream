"""M1 - Thu thập tick Coinbase (kênh ticker + heartbeat) -> Kafka, chạy liên tục, tự kết nối lại,
phát hiện mất gói bằng trade_id / sequence / heartbeat và tải bù qua REST.

Lưu ý: kênh ticker chỉ phát khi có khớp lệnh nên `sequence` (đánh số cho MỌI message của product) có
khoảng trống bình thường; vì vậy mất gói được xác định chính bằng `trade_id` (liên tiếp +1) đối chiếu với
`last_trade_id` trong heartbeat; `sequence` dùng để phát hiện trùng/đảo thứ tự.
"""
from __future__ import annotations
import asyncio, json, logging, os, random, signal, time
from datetime import datetime, timezone
from pathlib import Path

import requests
import websockets
from confluent_kafka import Producer

from common.config import load_config, path_of, ensure

log = logging.getLogger("producer")


def iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class FeedState:
    def __init__(self):
        self.last_trade_id = None
        self.last_seq = None
        self.after_reconnect = False
        self.hb_mismatch = 0
        self.last_hb = None


class Collector:
    def __init__(self, cfg):
        self.cfg = cfg
        cb = cfg["coinbase"]
        self.products = list(cb["products"]) + ([cb["gold_proxy"]] if cb.get("gold_proxy") else [])
        self.topic = cfg["kafka"]["topics"]["ticks"]
        self.topic_ev = cfg["kafka"]["topics"]["events"]
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap"])
        self.producer = Producer({"bootstrap.servers": bootstrap, "acks": "all", "enable.idempotence": True,
                                  "linger.ms": 5, "compression.type": "lz4", "message.timeout.ms": 300000})
        self.state = {p: FeedState() for p in self.products}
        self.stats = dict(ticks=0, backfilled=0, gaps=0, missed_trades=0, duplicates=0, seq_regressions=0,
                          bad_msgs=0, reconnects=0, heartbeat_missing=0, kafka_errors=0)
        self.t0 = time.time()
        self.connected = False
        self.disc_since = self.t0
        self.downtime_s = 0.0
        self.last_msg = None
        self.stop = False
        mdir = ensure(path_of(cfg, "metrics"))
        self.status_fp, self.cum_fp, self.gap_fp = mdir / "producer_status.json", mdir / "producer_cumulative.json", mdir / "gap_events.jsonl"
        self.cum_base = json.loads(self.cum_fp.read_text()) if self.cum_fp.exists() else {}
        self.first_start = self.cum_base.get("first_start", self.t0)

    # ---------- Kafka ----------
    def _dr(self, err, msg):
        if err is not None:
            self.stats["kafka_errors"] += 1
            log.error("Kafka delivery failed: %s", err)

    def send(self, topic, key, obj):
        data = json.dumps(obj, separators=(",", ":"))
        try:
            self.producer.produce(topic, key=key, value=data, on_delivery=self._dr)
        except BufferError:
            self.producer.poll(0.5)
            self.producer.produce(topic, key=key, value=data, on_delivery=self._dr)
        self.producer.poll(0)

    # ---------- phát hiện mất gói ----------
    def gap(self, pid, first, last, reason):
        n = last - first + 1
        self.stats["gaps"] += 1
        self.stats["missed_trades"] += n
        ev = dict(product_id=pid, reason=reason, first_missing=first, last_missing=last, n_missed=n, detected_at=iso(time.time()))
        log.warning("GAP %s", ev)
        self.send(self.topic_ev, pid, ev)
        with open(self.gap_fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        bf = self.cfg["coinbase"]
        if bf.get("backfill") and n <= bf.get("backfill_max_trades", 20000):
            asyncio.create_task(self.backfill(pid, first, last))

    def fetch_trades(self, pid, first, last):
        url = f"{self.cfg['coinbase']['rest_url']}/products/{pid}/trades"
        out, after = [], last + 1
        for _ in range(40):
            r = requests.get(url, params={"limit": 1000, "after": after}, timeout=10)
            r.raise_for_status()
            trades = r.json()
            if not trades:
                break
            out += [t for t in trades if first <= t["trade_id"] <= last]
            oldest = min(t["trade_id"] for t in trades)
            if oldest <= first:
                break
            after = oldest
            time.sleep(0.25)          # tôn trọng rate-limit public
        return out

    async def backfill(self, pid, first, last):
        try:
            trades = await asyncio.to_thread(self.fetch_trades, pid, first, last)
        except Exception as e:
            log.error("backfill %s [%s-%s] lỗi: %s", pid, first, last, e)
            return
        for t in trades:
            self.send(self.topic, pid, dict(product_id=pid, trade_id=t["trade_id"], sequence=None,
                                            event_ts_ms=iso_to_ms(t["time"]), price=float(t["price"]),
                                            last_size=float(t["size"]), best_bid=None, best_ask=None,
                                            volume_24h=None, ingest_ts_ms=int(time.time() * 1000), source="backfill"))
        self.stats["backfilled"] += len(trades)
        log.info("backfill %s: %d/%d trade", pid, len(trades), last - first + 1)

    # ---------- xử lý message ----------
    def on_ticker(self, m):
        pid = m.get("product_id")
        st = self.state.get(pid)
        if st is None:
            return
        try:
            price, ts = float(m["price"]), iso_to_ms(m["time"])
        except (KeyError, ValueError, TypeError):
            self.stats["bad_msgs"] += 1
            return
        tid, seq = m.get("trade_id"), m.get("sequence")
        if tid is not None:
            if st.last_trade_id is not None:
                if tid <= st.last_trade_id:
                    self.stats["duplicates"] += 1
                    return
                if tid > st.last_trade_id + 1:
                    self.gap(pid, st.last_trade_id + 1, tid - 1, "reconnect_gap" if st.after_reconnect else "trade_id_gap")
            st.last_trade_id = tid
        if seq is not None:
            if st.last_seq is not None and seq <= st.last_seq:
                self.stats["seq_regressions"] += 1
            st.last_seq = seq
        st.after_reconnect = False
        f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
        self.send(self.topic, pid, dict(product_id=pid, trade_id=tid, sequence=seq, event_ts_ms=ts, price=price,
                                        last_size=f("last_size"), best_bid=f("best_bid"), best_ask=f("best_ask"),
                                        volume_24h=f("volume_24h"), ingest_ts_ms=int(time.time() * 1000), source="ws"))
        self.stats["ticks"] += 1

    def on_heartbeat(self, m):
        st = self.state.get(m.get("product_id"))
        if st is None:
            return
        st.last_hb = time.time()
        lt = m.get("last_trade_id")
        if lt is None or st.last_trade_id is None:
            return
        if lt > st.last_trade_id:
            st.hb_mismatch += 1
            if st.hb_mismatch >= 3:       # lệch kéo dài >= 3 heartbeat (~3s) => chắc chắn mất tick
                self.gap(m["product_id"], st.last_trade_id + 1, lt, "heartbeat_mismatch")
                st.last_trade_id = lt
                st.hb_mismatch = 0
        else:
            st.hb_mismatch = 0

    # ---------- kết nối ----------
    async def session(self):
        cb = self.cfg["coinbase"]
        async with websockets.connect(cb["ws_url"], ping_interval=20, ping_timeout=20, max_size=2 ** 22) as ws:
            await ws.send(json.dumps({"type": "subscribe", "product_ids": self.products, "channels": ["ticker", "heartbeat"]}))
            self.connected = True
            self.downtime_s += time.time() - self.disc_since
            for st in self.state.values():
                st.after_reconnect = True
                st.hb_mismatch = 0
            log.info("Đã kết nối %s: %s", cb["ws_url"], self.products)
            while not self.stop:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=cb["heartbeat_timeout_s"])
                except asyncio.TimeoutError:
                    raise ConnectionError(f"không có message trong {cb['heartbeat_timeout_s']}s")
                self.last_msg = time.time()
                m = json.loads(raw)
                t = m.get("type")
                if t == "ticker":
                    self.on_ticker(m)
                elif t == "heartbeat":
                    self.on_heartbeat(m)
                elif t == "error":
                    raise ConnectionError(f"server error: {m}")

    async def run_forever(self):
        backoff = 1.0
        while not self.stop:
            t_start = time.time()
            try:
                await self.session()
            except Exception as e:
                log.warning("Mất kết nối: %s", e)
            if self.connected:
                self.connected, self.disc_since = False, time.time()
            if self.stop:
                break
            backoff = 1.0 if time.time() - t_start > 60 else min(backoff * 2, 60)
            self.stats["reconnects"] += 1
            await asyncio.sleep(backoff + random.random())

    async def watchdog(self):
        warned = set()
        while not self.stop:
            await asyncio.sleep(5)
            now = time.time()
            if self.connected:
                for pid, st in self.state.items():
                    late = st.last_hb is not None and now - st.last_hb > 2 * self.cfg["coinbase"]["heartbeat_timeout_s"]
                    if late and pid not in warned:
                        self.stats["heartbeat_missing"] += 1
                        warned.add(pid)
                        log.warning("heartbeat %s bị mất > %ss", pid, 2 * self.cfg["coinbase"]["heartbeat_timeout_s"])
                    elif not late:
                        warned.discard(pid)
            self.write_status()

    def write_status(self):
        now = time.time()
        down = self.downtime_s + (0 if self.connected else now - self.disc_since)
        cum = {k: self.cum_base.get(k, 0) + self.stats[k] for k in ("ticks", "gaps", "missed_trades", "backfilled", "reconnects")}
        cum["downtime_s"] = self.cum_base.get("downtime_s", 0) + down
        cum["first_start"] = self.first_start
        cum["running_days"] = (now - self.first_start) / 86400
        status = dict(updated=iso(now), session_start=iso(self.t0), session_uptime_s=round(now - self.t0), connected=self.connected,
                      last_msg_age_s=None if self.last_msg is None else round(now - self.last_msg, 1), **self.stats,
                      products={p: dict(last_trade_id=s.last_trade_id, last_seq=s.last_seq) for p, s in self.state.items()},
                      cumulative=cum)
        self.status_fp.write_text(json.dumps(status, indent=2))
        self.cum_fp.write_text(json.dumps(cum))

    async def main(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: setattr(self, "stop", True))
            except NotImplementedError:   # Windows
                pass
        wd = asyncio.create_task(self.watchdog())
        try:
            await self.run_forever()
        finally:
            self.stop = True
            wd.cancel()
            self.producer.flush(15)
            self.write_status()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(Collector(load_config()).main())
