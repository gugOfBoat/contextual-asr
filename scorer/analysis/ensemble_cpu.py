"""CPU: pmaxsim và pmono có bù nhau không? Cộng logit hai model trên rank_<L>.json (đủ 2000 từ) rồi đo Sạch@K.
Dùng thêm vị trí của pmono để cho tranh nhau tại chỗ và đo precision / recall ở ngưỡng.
  python ensemble_cpu.py <split> <list>
"""
import json, math, random, sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
O = R / "scorer/runs/2026-11-14/out"
split, L = sys.argv[1], sys.argv[2]
A_NAME, B_NAME = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("pmaxsim_c4", "pmono_c4")
ref = {l.split("\t")[0]: set(json.loads(l.rstrip("\n").split("\t")[2])) for l in open(R / "bench/protocol" / split / "ref.tsv", encoding="utf-8")}
A = json.load(open(O / A_NAME / split / f"rank_{L}.json", encoding="utf-8"))
B = json.load(open(O / B_NAME / split / f"rank_{L}.json", encoding="utf-8"))
SA = json.load(open(O / A_NAME / split / f"span_{L}.json", encoding="utf-8"))
SB = json.load(open(O / B_NAME / split / f"span_{L}.json", encoding="utf-8"))
uids = sorted(set(A) & set(B))


def above(order, gold):
    """→ [(từ đúng, số nhiễu đứng trên)]"""
    n, out = 0, {}
    for w in order:
        if w in gold:
            out[w] = n
        else:
            n += 1
    return [out.get(w, 9999) for w in gold]


sys_ = {}
for name, wa, wb in [("pmaxsim_c4", 1, 0), ("pmono_c4", 0, 1), ("cộng 1:1", 1, 1), ("cộng 2:1", 2, 1), ("cộng 1:2", 1, 2)]:
    sc = {}
    for u in uids:
        a, b = dict(A[u]), dict(B[u])
        sc[u] = {w: wa * a[w] + wb * b.get(w, -20) for w in a}
    sys_[name] = sc
rnd = random.Random(0)
boots = [[rnd.choice(uids) for _ in uids] for _ in range(500)]
print(f"## {split} {L}: cộng điểm pmaxsim + pmono ({len(uids)} câu)\n")
print("| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với pmaxsim_c4 [CI 95%] |\n|---|---|---|---|---|")
per = {}
for name, sc in sys_.items():
    per[name] = {u: above(sorted(sc[u], key=sc[u].get, reverse=True), ref[u]) for u in uids}
    f = lambda us, K, n=name: 100 * sum(d < K for u in us for d in per[n][u]) / max(sum(len(per[n][u]) for u in us), 1)
    ci = ""
    if name != "pmaxsim_c4":
        g = lambda us, K: 100 * sum(d < K for u in us for d in per["pmaxsim_c4"][u]) / max(sum(len(per["pmaxsim_c4"][u]) for u in us), 1)
        ds = sorted(f(b, 1) - g(b, 1) for b in boots)
        ci = f"{f(uids, 1) - g(uids, 1):+.1f} [{ds[12]:+.1f}; {ds[487]:+.1f}]"
    print(f"| {name} | {f(uids, 1):.1f} | {f(uids, 5):.1f} | {f(uids, 10):.1f} | {ci} |")


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


print("\n### Ngưỡng trên điểm cộng 1:1 (p = sigmoid(điểm / 2)), top-50, tranh nhau tại chỗ bằng vị trí của pmono (thiếu thì pmaxsim)\n")
print("| hệ | τ | giữ từ đúng % | nhiễu / câu | precision % |\n|---|---|---|---|---|")
tot = sum(len(ref[u]) for u in uids)
for name, div in [("pmaxsim_c4", 1), ("pmono_c4", 1), ("cộng 1:1", 2)]:
    for tau in (0.3, 0.5):
        tp = fp = 0
        for u in uids:
            sc = sys_[name][u]
            order = sorted(sc, key=sc.get, reverse=True)[:50]
            sp = dict(SA.get(u, {})); sp.update(SB.get(u, {})) if name != "pmaxsim_c4" else None
            for w in nms(order, sp):
                if 1 / (1 + math.exp(-sc[w] / div)) >= tau:
                    tp += w in ref[u]; fp += w not in ref[u]
        print(f"| {name} | {tau} | {100 * tp / tot:.1f} | {fp / len(uids):.2f} | {100 * tp / max(tp + fp, 1):.1f} |")
