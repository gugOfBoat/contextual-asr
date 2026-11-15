"""Tranh nhau tại chỗ (không train): trong top-50, hai từ có span chồng nhau thì chỉ giữ từ điểm cao hơn.
Đo ở ngưỡng p ≥ τ: recall từ đúng, số từ nhiễu / câu, precision — trước và sau khi cho tranh nhau.
  python nms_at_place.py <split> <list> tên=thư_mục ...
"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
split, L = sys.argv[1], sys.argv[2]
ref = {l.split("\t")[0]: set(json.loads(l.rstrip("\n").split("\t")[2])) for l in open(R / "bench/protocol" / split / "ref.tsv", encoding="utf-8")}


def nms(lst, sp, iou=0.3):
    keep = []
    for w, p in lst:                                   # đã xếp giảm dần
        if w not in sp:
            keep.append((w, p)); continue
        b, e = sp[w]
        ok = True
        for x, _ in keep:
            if x in sp:
                b2, e2 = sp[x]
                inter = min(e, e2) - max(b, b2)
                if inter > 0 and inter / (max(e, e2) - min(b, b2)) > iou:
                    ok = False; break
        if ok:
            keep.append((w, p))
    return keep


print(f"## {split} {L}: ngưỡng p ≥ τ, trước → sau khi tranh nhau tại chỗ\n")
print("| hệ | τ | recall từ đúng % | nhiễu / câu | precision % |")
print("|---|---|---|---|---|")
for spec in sys.argv[3:]:
    n, _, d = spec.partition("=")
    prob = json.load(open(Path(d) / f"prob_{L}.json", encoding="utf-8"))
    span = json.load(open(Path(d) / f"span_{L}.json", encoding="utf-8"))
    tot = sum(len(ref[u]) for u in prob)
    for tau in (0.3, 0.5, 0.7, 0.9):
        cells = []
        for use in (False, True):
            tp = fp = 0
            for u, lst in prob.items():
                ls = nms(lst, span[u]) if use else lst
                for w, p in ls:
                    if p >= tau:
                        tp += w in ref[u]; fp += w not in ref[u]
            cells.append((100 * tp / max(tot, 1), fp / len(prob), 100 * tp / max(tp + fp, 1)))
        a, b = cells
        print(f"| {n} | {tau} | {a[0]:.1f} → {b[0]:.1f} | {a[1]:.2f} → {b[1]:.2f} | {a[2]:.1f} → {b[2]:.1f} |")
