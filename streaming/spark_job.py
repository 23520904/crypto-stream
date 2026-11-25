"""M3 - Speed layer: Kafka -> Spark Structured Streaming (làm sạch, khử trùng, gom nến 1 phút)
-> dự báo 5 bước -> Parquet.

Chạy live:      python -m streaming.spark_job --master "local[2]"
Chạy benchmark: python -m streaming.spark_job --master "local[4]" --topic ticks_bench --starting earliest \
                    --out data/bench/k4 --checkpoint data/checkpoints/bench_k4 --reset --idle-timeout 45 --threads 4
Yêu cầu: Java 17, pyspark 3.5.x (lần đầu cần Internet để tải gói spark-sql-kafka).
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
import pandas as pd

from common.config import load_config, path_of, ensure
from .runtime import SpeedLayer

KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="local[2]")
    ap.add_argument("--topic")
    ap.add_argument("--starting", default="latest", choices=["latest", "earliest"])
    ap.add_argument("--out")
    ap.add_argument("--checkpoint")
    ap.add_argument("--reset", action="store_true", help="xoá checkpoint + output trước khi chạy")
    ap.add_argument("--max-offsets", type=int, default=0, help="maxOffsetsPerTrigger (0 = không giới hạn)")
    ap.add_argument("--trigger-seconds", type=float)
    ap.add_argument("--idle-timeout", type=float, default=0, help="tự dừng sau N giây không có dữ liệu mới (0 = chạy mãi)")
    ap.add_argument("--threads", type=int, default=0, help="số thread torch (mặc định = số core của master)")
    a = ap.parse_args()

    cfg = load_config()
    sc = cfg["streaming"]
    topic = a.topic or cfg["kafka"]["topics"]["ticks"]
    out = Path(a.out) if a.out else path_of(cfg, "stream")
    ckpt = Path(a.checkpoint) if a.checkpoint else path_of(cfg, "checkpoints", "speed")
    if a.reset:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(ckpt, ignore_errors=True)
    ensure(out)
    cores = int(a.master.split("[")[1].split("]")[0]) if "[" in a.master and a.master.split("[")[1][0].isdigit() else os.cpu_count()

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = (SparkSession.builder.master(a.master).appName("crypto-speed-layer")
             .config("spark.jars.packages", KAFKA_PKG)
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.driver.memory", "2g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    schema = T.StructType([
        T.StructField("product_id", T.StringType()), T.StructField("trade_id", T.LongType()),
        T.StructField("sequence", T.LongType()), T.StructField("event_ts_ms", T.LongType()),
        T.StructField("price", T.DoubleType()), T.StructField("last_size", T.DoubleType()),
        T.StructField("best_bid", T.DoubleType()), T.StructField("best_ask", T.DoubleType()),
        T.StructField("volume_24h", T.DoubleType()), T.StructField("ingest_ts_ms", T.LongType()),
        T.StructField("source", T.StringType())])

    rd = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap"]))
          .option("subscribe", topic).option("startingOffsets", a.starting).option("failOnDataLoss", "false"))
    if a.max_offsets:
        rd = rd.option("maxOffsetsPerTrigger", a.max_offsets)

    pids = cfg["coinbase"]["products"] + ([cfg["coinbase"]["gold_proxy"]] if cfg["coinbase"].get("gold_proxy") else [])
    ticks = (rd.load()
             .select(F.from_json(F.col("value").cast("string"), schema).alias("j")).select("j.*")
             # --- làm sạch ---
             .where(F.col("product_id").isin(pids) & F.col("price").isNotNull() & (F.col("price") > 0)
                    & F.col("event_ts_ms").isNotNull() & F.col("trade_id").isNotNull())
             .withColumn("last_size", F.coalesce(F.col("last_size"), F.lit(0.0)))
             .withColumn("event_time", (F.col("event_ts_ms") / 1000).cast("timestamp")))

    order = F.struct("event_time", "trade_id")
    candles = (ticks.withWatermark("event_time", sc["watermark"])
               .dropDuplicatesWithinWatermark(["product_id", "trade_id"])      # ws + backfill có thể trùng
               .groupBy(F.window("event_time", sc["candle"]).alias("w"), "product_id")
               .agg(F.min_by("price", order).alias("open"), F.max("price").alias("high"),
                    F.min("price").alias("low"), F.max_by("price", order).alias("close"),
                    F.sum("last_size").alias("volume"), F.count("*").alias("n_ticks"))
               .select(F.col("w.start").alias("minute"), "product_id", "open", "high", "low", "close", "volume", "n_ticks"))

    layer = SpeedLayer(cfg, out, threads=a.threads or cores)

    def handle(df, batch_id):
        t_recv = time.time()
        layer.process_batch(df.toPandas(), batch_id, t_recv)

    trig = a.trigger_seconds or sc["trigger_seconds"]
    q = (candles.writeStream.outputMode("append").foreachBatch(handle)
         .option("checkpointLocation", str(ckpt)).trigger(processingTime=f"{trig} seconds").start())
    print(f"[speed layer] topic={topic} master={a.master} out={out}")

    # ----- theo dõi tiến độ: thông lượng + thời gian xử lý -----
    prog_fp = ensure(out / "metrics") / "progress.jsonl"
    seen, ticks_total, first_ts, last_end, last_data = set(), 0, None, None, time.time()
    t_wall0 = time.time()
    try:
        while q.isActive:
            time.sleep(1)
            for p in list(q.recentProgress):
                if p["batchId"] in seen:
                    continue
                seen.add(p["batchId"])
                n = p.get("numInputRows", 0)
                with open(prog_fp, "a") as f:
                    f.write(json.dumps(dict(batch_id=p["batchId"], timestamp=p["timestamp"], numInputRows=n,
                                            inputRowsPerSecond=p.get("inputRowsPerSecond"),
                                            processedRowsPerSecond=p.get("processedRowsPerSecond"),
                                            durationMs=p.get("durationMs"))) + "\n")
                if n > 0:
                    ticks_total += n
                    start = pd.Timestamp(p["timestamp"])
                    end = start + pd.Timedelta(milliseconds=p["durationMs"].get("triggerExecution", 0))
                    first_ts = first_ts or start
                    last_end = end
                    last_data = time.time()
            if a.idle_timeout and ticks_total > 0 and time.time() - last_data > a.idle_timeout:
                print("[speed layer] hết dữ liệu, dừng.")
                q.stop()
                break
    except KeyboardInterrupt:
        q.stop()
    finally:
        wall = (last_end - first_ts).total_seconds() if first_ts is not None else None
        (out / "metrics" / "summary.json").write_text(json.dumps(dict(
            master=a.master, cores=cores, topic=topic, ticks_total=ticks_total, processing_wall_s=wall,
            throughput_ticks_per_s=(ticks_total / wall) if wall else None, total_runtime_s=time.time() - t_wall0), indent=2))
        spark.stop()


if __name__ == "__main__":
    main()
