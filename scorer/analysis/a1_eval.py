"""Chấm kết quả A1 (a1_pool.py): R#10 / R#50 theo cách gộp, theo độ dài câu, và sai số định vị của diag / box.

  python analysis/a1_eval.py --split test-other [--hyp ../bench/out/test-other/whisper_oracle_plain.tsv]

R#K: từ đúng có hạng < K trong list của câu (như outrank.py; câu > K từ hiếm không thể đạt 100%).
Định vị: tâm dự đoán = (frame đầu + số frame TTS / 2) × 0.02 s, so với tâm MFA (protocol/<split>/align.tsv).
--hyp: bản chép Whisper không context → tách nhóm từ đúng mà ASR chép sai (như Table 9 của Hoàng).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, WORK  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--list", default="N2000")
ap.add_argument("--hyp")
ap.add_argument("--v", nargs="+", default=["mean", "max", "diag", "box"])
ap.add_argument("--loc", default="loc", help="loc (diag, box) hoặc loc_s1 (s1, s1raw)")
a = ap.parse_args()

proto = BENCH / "protocol" / a.split
out = WORK / "out" / "a1" / a.split
orc = json.load(open(proto / "ctx_oracle.json", encoding="utf-8"))
nref = {l.split("\t")[0]: len(l.split("\t")[1].split()) for l in open(proto / "ref.tsv", encoding="utf-8")}
bins = [(0, 14, "<15"), (15, 29, "15-29"), (30, 49, "30-49"), (50, 999, ">=50")]

print(f"== {a.split} {a.list}")
print(f"{'cách gộp':8s} {'R#10':>6s} {'R#50':>6s} " + " ".join(f"{b[2]:>6s}" for b in bins))
for v in a.v:
    rank = json.load(open(out / f"rank_{v}_{a.list}.json", encoding="utf-8"))
    n = h10 = h50 = 0
    bn, bh = defaultdict(int), defaultdict(int)
    for uid, lst in rank.items():
        pos = {w: i for i, (w, _) in enumerate(lst)}
        b = next(x[2] for x in bins if x[0] <= nref[uid] <= x[1])
        for t in orc.get(uid, []):
            r = pos.get(t, 10 ** 9)
            n += 1
            h10 += r < 10
            h50 += r < 50
            bn[b] += 1
            bh[b] += r < 10
    print(f"{v:8s} {100 * h10 / n:6.2f} {100 * h50 / n:6.2f} " + " ".join(f"{100 * bh[x[2]] / max(bn[x[2]], 1):6.1f}" for x in bins)
          + f"   (n={n})")

spans = defaultdict(list)
for l in open(proto / "align.tsv", encoding="utf-8"):
    uid, _, w, b, e = l.rstrip("\n").split("\t")[:5]
    spans[uid, w].append((float(b), float(e)))
hyp = {}
if a.hyp:
    hyp = {l.split("\t")[0]: set(l.rstrip("\n").split("\t")[1].split()) for l in open(a.hyp, encoding="utf-8")}
loc = json.load(open(out / f"{a.loc}_{a.list}.json", encoding="utf-8"))
print("định vị từ đúng (tâm dự đoán so với MFA):")
for j, v in enumerate(["diag", "box"] if a.loc == "loc" else ["s1", "s1raw"]):
    for grp in (["tất cả", "ASR sai"] if hyp else ["tất cả"]):
        err, inside, near = [], 0, 0
        for uid, d in loc.items():
            for t in orc.get(uid, []):
                if t not in d or not spans[uid, t]:
                    continue
                if grp == "ASR sai" and t in hyp.get(uid, set()):
                    continue
                c = (d[t][j] + d[t][2] / 2) * 0.02
                e, (b0, e0) = min((abs(c - (b + e) / 2), (b, e)) for b, e in spans[uid, t])
                err.append(e)
                inside += b0 <= c <= e0
                near += e <= 0.2
        if err:
            print(f"  {v:5s} {grp:7s} n={len(err):5d}  sai số trung vị {np.median(err):.3f}s  "
                  f"tâm nằm trong span {100 * inside / len(err):5.1f}%  trong ±0.2s {100 * near / len(err):5.1f}%")
