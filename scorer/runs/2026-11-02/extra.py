"""Số bổ sung từ out/: (a) độ trải đỉnh âm vị trong từ — tách "bậc thang" khỏi "dải dọc"; (b) định vị theo xác suất của scorer.

  python extra.py out > extra.md
"""
import json, sys
from collections import defaultdict
import numpy as np

D = sys.argv[1]
for s in ["test-clean", "test-other"]:
    P = [json.loads(l) for l in open(f"{D}/{s}_phone.jsonl")]
    W = defaultdict(list)
    for r in P:
        W[(r["model"], r["map"], r["uid"], r["w"])].append(r)
    print(f"\n## {s}: độ trải đỉnh âm vị trong từ (từ ≥ 3 âm vị)\n")
    print("| bản đồ | số đỉnh khác nhau / số âm vị | (đỉnh cuối − đỉnh đầu) / độ dài từ | từ mà ≥ 80% âm vị chung 1 đỉnh (±1 frame) (%) | tăng ngặt (%) |")
    print("| --- | --- | --- | --- | --- |")
    for mm in [("pmaxsim", "cos"), ("pxattn", "att1"), ("pxattn", "att2"), ("pxattn", "cos_raw")]:
        a, b, c, d = [], [], [], []
        for k, rs in W.items():
            if k[:2] != mm or len(rs) < 3:
                continue
            rs = sorted(rs, key=lambda r: r["j"])
            pk = np.array([r["peak_f"] for r in rs])
            wl = sum(r["seg_dur"] for r in rs) * 50
            a.append(len(set(pk)) / len(pk))
            b.append((pk[-1] - pk[0]) / wl)
            mode = np.bincount(pk - pk.min()).argmax() + pk.min()
            c.append(np.mean(np.abs(pk - mode) <= 1) >= 0.8)
            d.append(bool(np.all(np.diff(pk) > 0)))
        print(f"| {mm[0]}/{mm[1]} | {np.mean(a):.2f} | {np.mean(b):.2f} | {100 * np.mean(c):.1f} | {100 * np.mean(d):.1f} |")
    R = [json.loads(l) for l in open(f"{D}/{s}_rows.jsonl")]
    print(f"\n## {s}: định vị (streak) theo xác suất của scorer (từ đúng)\n")
    print(r"| bản đồ | p ≥ 0.5: n / median \|err\| / trúng span % | p < 0.5: n / median / trúng % |")
    print("| --- | --- | --- |")
    for mm in [("hou25", "S1"), ("pmaxsim", "cos"), ("pxattn", "att1")]:
        G = [r for r in R if r["true"] and (r["model"], r["map"]) == mm]
        cells = []
        for cond in (lambda r: r["p"] >= 0.5, lambda r: r["p"] < 0.5):
            g = [r for r in G if cond(r)]
            cells.append(f"{len(g)} / {np.median([r['err_streak'] for r in g]):.2f} / {100 * np.mean([r['hit_streak'] for r in g]):.1f}" if g else "–")
        print(f"| {mm[0]}/{mm[1]} | {cells[0]} | {cells[1]} |")
