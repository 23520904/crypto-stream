"""Tạo topic Kafka (retention 60 ngày cho 'ticks')."""
import os, time
from confluent_kafka.admin import AdminClient, NewTopic
from common.config import load_config


def main():
    cfg = load_config()
    k = cfg["kafka"]
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", k["bootstrap"])
    admin = AdminClient({"bootstrap.servers": bootstrap})
    md = None
    for _ in range(60):
        try:
            md = admin.list_topics(timeout=5)
            break
        except Exception as e:
            print("chờ Kafka...", e)
            time.sleep(5)
    if md is None:
        raise SystemExit("Không kết nối được Kafka")
    day = 86400 * 1000
    spec = {
        k["topics"]["ticks"]: (k["partitions"], k["retention_days"] * day),
        k["topics"]["events"]: (1, k["retention_days"] * day),
        k["topics"]["bench"]: (4, 7 * day),
    }
    new = [NewTopic(t, num_partitions=p, replication_factor=1, config={"retention.ms": str(r)})
           for t, (p, r) in spec.items() if t not in md.topics]
    for t, f in admin.create_topics(new).items():
        f.result()
        print("đã tạo topic", t)
    print("topics:", sorted(admin.list_topics(timeout=5).topics))


if __name__ == "__main__":
    main()
