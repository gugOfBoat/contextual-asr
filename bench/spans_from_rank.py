"""Vị trí do scorer đoán (rank.py --span → span_<L>.json) → tsv cho gate_boost.py --spans.

  python spans_from_rank.py <span_L.json> <ra.tsv>
Ra: uid, -, từ, bắt đầu, kết thúc (giây), cùng định dạng align.tsv.
"""
import json, sys

sp = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for u, ws in sp.items():
        for w, (b, e) in ws.items():
            f.write(f"{u}\t-\t{w}\t{b:.2f}\t{e:.2f}\n")
print(f"{sum(map(len, sp.values()))} vị trí -> {sys.argv[2]}")
