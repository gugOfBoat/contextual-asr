"""Từ kết quả A1 (a1_pool.py) tạo đầu vào cho E1 với vị trí DỰ ĐOÁN thay cho MFA.

  python analysis/a1_spans.py --split test-other --v diag

Ra (bench/protocol/<split>/):
  ctx_a1<v>_N2000.json, ctx_a1<v>10_N2000.json   top-50 / top-10 theo cách gộp <v>, xếp TĂNG dần như rank.py
  pred_a1<v>.tsv                                 uid, -, từ, bắt đầu, kết thúc (s) cho mọi từ có vị trí
                                                 (top của diag/box + từ đúng) → gate_boost.py --spans
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, WORK  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--v", default="diag", choices=["mean", "diag", "box", "s1", "s1raw"])
ap.add_argument("--list", default="N2000")
ap.add_argument("--pos", help="lấy vị trí từ cách này (mặc định = --v), ví dụ --v mean --pos s1")
a = ap.parse_args()

proto = BENCH / "protocol" / a.split
out = WORK / "out" / "a1" / a.split
rank = json.load(open(out / f"rank_{a.v}_{a.list}.json", encoding="utf-8"))
pv = a.pos or a.v
lf = "loc_s1" if pv.startswith("s1") else "loc"
loc = json.load(open(out / f"{lf}_{a.list}.json", encoding="utf-8"))
j = 0 if pv in ("diag", "s1") else 1
for k, tag in [(50, ""), (10, "10")]:
    json.dump({u: [w for w, _ in ws[:k]][::-1] for u, ws in rank.items()},
              open(proto / f"ctx_a1{a.v}{tag}_{a.list}.json", "w", encoding="utf-8"), ensure_ascii=False)
with open(proto / f"pred_a1{pv}.tsv", "w", encoding="utf-8") as f:
    for uid, d in loc.items():
        for w, x in d.items():
            f.write(f"{uid}\t-\t{w}\t{x[j] * 0.02:.2f}\t{(x[j] + x[2]) * 0.02:.2f}\n")
print("xong", a.split, a.v)
