"""So sánh kết quả tái hiện (results/batch/exp1) với bài báo [1] -> results/m2_reproduction.csv"""
import json, argparse
import pandas as pd
from common.config import load_config, path_of, ensure
from common.paper_ref import PAPER_REF_EXP1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.25, help="sai lệch tương đối tối đa để coi là 'khớp'")
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for fp in sorted(path_of(cfg, "results", "batch", "exp1").glob("*.json")):
        r = json.loads(fp.read_text())
        ref = PAPER_REF_EXP1.get((r["coin"], r["strategy"], r["model"]))
        mine = r["test_mean"]["mean"]
        rows.append(dict(coin=r["coin"], strategy=r["strategy"], model=r["model"], runs=r["runs"],
                         ours=round(mine, 4), ci95=round(r["test_mean"]["ci"], 4), paper=ref,
                         rel_diff=None if ref is None else round((mine - ref) / ref, 3),
                         match=None if ref is None else ("khớp" if abs(mine - ref) / ref <= a.tol else "lệch"),
                         scaler_fit=r.get("scaler_fit")))
    df = pd.DataFrame(rows)
    out = ensure(path_of(cfg, "results")) / "m2_reproduction.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nĐã lưu {out}")
