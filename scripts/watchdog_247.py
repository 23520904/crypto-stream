"""Kịch bản SRE Watchdog 24/7: Giám sát liên tục trạng thái cụm Kafka, Producer và Spark Streaming.
Tự động phát hiện lỗi và khởi động lại nếu tiến trình bị dừng.

Sử dụng:
  python scripts/watchdog_247.py --loop --interval 30
  hoặc chạy kiểm tra 1 lần: python scripts/watchdog_247.py
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config, path_of, ensure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [SRE-WATCHDOG] %(message)s")
log = logging.getLogger("sre_watchdog")


def check_kafka_port(bootstrap="localhost:9092") -> bool:
    import socket
    host, port = bootstrap.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=3):
            return True
    except Exception:
        return False


def check_spark_process() -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", "streaming.spark_job"], text=True)
        pids = [int(p) for p in out.strip().split() if p]
        return pids[0] if pids else None
    except Exception:
        return None


def restart_spark_job(master="local[2]"):
    log.warning("Khởi động lại Spark Streaming job...")
    py = sys.executable
    cmd = [py, "-m", "streaming.spark_job", "--master", master]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    log.info("Spark job đã được khởi động lại với PID: %s", proc.pid)
    return proc.pid


def check_stream_freshness(stream_dir: Path, max_age_seconds: float = 300) -> dict:
    cand_dir = stream_dir / "candles"
    pred_dir = stream_dir / "predictions"
    now = time.time()
    
    cand_files = list(cand_dir.glob("*.parquet"))
    pred_files = list(pred_dir.glob("*.parquet"))
    
    latest_cand_age = min((now - f.stat().st_mtime for f in cand_files), default=999999)
    latest_pred_age = min((now - f.stat().st_mtime for f in pred_files), default=999999)
    
    return {
        "candles_count": len(cand_files),
        "latest_candle_age_s": round(latest_cand_age, 1),
        "predictions_count": len(pred_files),
        "latest_prediction_age_s": round(latest_pred_age, 1),
        "is_fresh": latest_cand_age < max_age_seconds and latest_pred_age < max_age_seconds,
    }


def inspect_all(cfg: dict, auto_restart: bool = True) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kafka_ok = check_kafka_port(cfg["kafka"]["bootstrap"])
    spark_pid = check_spark_process()
    freshness = check_stream_freshness(path_of(cfg, "stream"))
    
    report = {
        "timestamp": now_iso,
        "kafka_healthy": kafka_ok,
        "spark_running": spark_pid is not None,
        "spark_pid": spark_pid,
        "stream_health": freshness,
        "status": "HEALTHY" if (kafka_ok and spark_pid and freshness["is_fresh"]) else "DEGRADED"
    }
    
    if not kafka_ok:
        log.error("CẢNH BÁO: Không thể kết nối Kafka tại %s!", cfg["kafka"]["bootstrap"])
    if not spark_pid:
        log.error("CẢNH BÁO: Spark Streaming job không hoạt động!")
        if auto_restart and kafka_ok:
            new_pid = restart_spark_job()
            report["spark_pid"] = new_pid
            report["spark_running"] = True
            report["status"] = "RECOVERED"
    elif not freshness["is_fresh"]:
        log.warning("CẢNH BÁO: Dữ liệu stream nến quá cũ (>300s). Nến mới nhất cách đây: %ss", freshness["latest_candle_age_s"])
        
    log.info("Trạng thái hệ thống: %s | Kafka: %s | Spark PID: %s | Nến: %d | Dự báo: %d",
             report["status"], "OK" if kafka_ok else "ERR", spark_pid, freshness["candles_count"], freshness["predictions_count"])
    
    # Ghi log trạng thái vào data/metrics/watchdog_status.json
    status_file = ensure(path_of(cfg, "metrics")) / "watchdog_status.json"
    status_file.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="Chạy vòng lặp giám sát liên tục")
    ap.add_argument("--interval", type=int, default=30, help="Chu kỳ kiểm tra (giây)")
    ap.add_argument("--no-restart", action="store_true", help="Không tự động khởi động lại tiến trình bị lỗi")
    a = ap.parse_args()
    
    cfg = load_config()
    log.info("Khởi động SRE Watchdog 24/7 (vòng lặp: %s, chu kỳ: %ss)...", a.loop, a.interval)
    
    if a.loop:
        while True:
            try:
                inspect_all(cfg, auto_restart=not a.no_restart)
            except Exception as e:
                log.error("Lỗi trong vòng lặp watchdog: %s", e)
            time.sleep(a.interval)
    else:
        inspect_all(cfg, auto_restart=not a.no_restart)


if __name__ == "__main__":
    main()
