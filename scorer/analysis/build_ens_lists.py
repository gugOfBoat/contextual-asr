"""Tạo list + vị trí cho gate từ điểm cộng của hai scorer (A: khớp tự do, B: khớp đúng thứ tự).
  python build_ens_lists.py <tên ra> <A> <B>
Ra (bench/protocol/<split>/): ctx_<tên>_<L>.json (top-50, điểm cao đứng cuối), ctx_<tên>_<L>_nms<τ>.json, span_<tên>_<L>.tsv (biên của B, thiếu thì A)."""
import json, math, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]; O = R / "scorer/runs/2026-11-14/out"; P = R / "bench/protocol"
name, an, bn = sys.argv[1:4]


def nms(order, sp):
    keep = []
    for w in order:
        if w in sp:
            b, e = sp[w]
            if any(x in sp and min(e, sp[x][1]) - max(b, sp[x][0]) > 0
                   and (min(e, sp[x][1]) - max(b, sp[x][0])) / (max(e, sp[x][1]) - min(b, sp[x][0])) > 0.3 for x in keep):
                continue
        keep.append(w)
    return keep


for split in ("dev-other", "test-other", "test-clean"):
    for L in ("N2000", "missing_N2000"):
        try:
            A = json.load(open(O / an / split / f"rank_{L}.json", encoding="utf-8")); B = json.load(open(O / bn / split / f"rank_{L}.json", encoding="utf-8"))
            SA = json.load(open(O / an / split / f"span_{L}.json", encoding="utf-8")); SB = json.load(open(O / bn / split / f"span_{L}.json", encoding="utf-8"))
        except FileNotFoundError:
            continue
        top, lists, nsp = {}, {0.3: {}, 0.5: {}}, 0
        with open(P / split / f"span_{name}_{L}.tsv", "w", encoding="utf-8") as f:
            for u in A:
                b = dict(B[u]); sc = {w: (s + b.get(w, -20)) / 2 for w, s in A[u]}
                order = sorted(sc, key=sc.get, reverse=True)[:50]
                sp = dict(SA.get(u, {})); sp.update(SB.get(u, {}))
                top[u] = order[::-1]
                kept = nms(order, sp)
                for t in lists:
                    lists[t][u] = [w for w in kept if 1 / (1 + math.exp(-sc[w])) >= t]
                for w in order:
                    if w in sp:
                        f.write(f"{u}\t-\t{w}\t{sp[w][0]:.2f}\t{sp[w][1]:.2f}\n"); nsp += 1
        json.dump(top, open(P / split / f"ctx_{name}_{L}.json", "w", encoding="utf-8"), ensure_ascii=False)
        for t, d in lists.items():
            json.dump(d, open(P / split / f"ctx_{name}_{L}_nms{t}.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(split, L, " ".join(f"τ={t}: {sum(map(len, d.values())) / len(d):.2f} từ/câu" for t, d in lists.items()), f"| có vị trí {nsp / (50 * len(A)):.0%} top-50")
