"""M5 - Đo hiệu năng với 1, 2, 4 core trên CÙNG một tập tick (đã nạp vào topic ticks_bench).
  python -m benchmark.replay --synthetic        # hoặc --from-topic
  python -m benchmark.run_benchmark --cores 1 2 4
Kết quả: results/benchmark/summary.csv"""
import argparse, json, subprocess, sys
import numpy as np
import pandas as pd
from common.config import load_config, path_of, ensure, ROOT
from common.evalutils import read_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--idle-timeout", type=int, default=45)
    ap.add_argument("--max-offsets", type=int, default=100000)
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for k in a.cores:
        out = path_of(cfg, "bench", f"k{k}")
        ck = path_of(cfg, "checkpoints", f"bench_k{k}")
        cmd = [sys.executable, "-m", "streaming.spark_job", "--master", f"local[{k}]", "--topic", cfg["kafka"]["topics"]["bench"],
               "--starting", "earliest", "--out", str(out), "--checkpoint", str(ck), "--reset", "--threads", str(k),
               "--idle-timeout", str(a.idle_timeout), "--max-offsets", str(a.max_offsets), "--trigger-seconds", "1"]
        print(">>", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        s = json.loads((out / "metrics" / "summary.json").read_text())
        pred = read_dir(out / "predictions")
        u = pred.drop_duplicates(["made_at", "coin"])
        rows.append(dict(cores=k, ticks=s["ticks_total"], wall_s=round(s["processing_wall_s"], 2),
                         throughput_ticks_s=round(s["throughput_ticks_per_s"], 1), pred_minutes=u["made_at"].nunique(),
                         infer_ms_p50=round(u["infer_ms"].quantile(.5), 2), infer_ms_p95=round(u["infer_ms"].quantile(.95), 2),
                         latency_ms_p50=round(u["latency_ms"].quantile(.5), 1), latency_ms_p95=round(u["latency_ms"].quantile(.95), 1)))
    df = pd.DataFrame(rows)
    f = ensure(path_of(cfg, "results", "benchmark")) / "summary.csv"
    df.to_csv(f, index=False)
    print(df.to_string(index=False))
    print("Đã lưu", f)
    print("Ghi chú: latency_ms = từ lúc nến vào foreachBatch đến lúc có dự báo (gồm thời gian chờ trong backlog khi replay burst); "
          "infer_ms = thời gian suy luận thuần cho 1 phút x 1 coin.")
