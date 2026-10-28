"""Vì sao Recall_B@99 cao: xem 1% từ đúng khó nhất (điểm thấp nhất theo ngưỡng chung) là từ gì.

  python diag_at99.py --rank rank_N2000.json --ref ref.tsv --ctx ctx_N2000.json --phones phones.tsv --hyp qwen3asr_none.tsv
"""
import argparse, json, math
from collections import Counter

ap = argparse.ArgumentParser()
for k in ["rank", "ref", "ctx", "phones", "hyp"]:
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()

ref, text = {}, {}
for line in open(a.ref, encoding="utf-8"):
    uid, t, rare = line.rstrip("\n").split("\t")
    ref[uid], text[uid] = json.loads(rare), t
ctx = json.load(open(a.ctx, encoding="utf-8"))
rank = json.load(open(a.rank, encoding="utf-8"))
ph = dict(line.rstrip("\n").split("\t") for line in open(a.phones, encoding="utf-8"))
hyp = {l.split("\t")[0]: set(l.rstrip("\n").partition("\t")[2].lower().split()) for l in open(a.hyp, encoding="utf-8")}
uids = [u for u in ref if u in rank]
gold = {u: [w for w in ref[u] if w in set(ctx[u])] for u in uids}

allv = sorted((s for u in uids for _, s in rank[u]), reverse=True)
gs = sorted(((s, u, w) for u in uids for w, s in rank[u] if w in set(gold[u])), key=lambda t: t[0])
n = len(gs)
print(f"{len(uids)} câu, {n} từ đúng, {len(allv) / len(uids):.0f} ứng viên/câu")


def kept_at(thr):  # số từ trung bình mỗi câu có điểm ≥ thr
    import bisect
    return (len(allv) - bisect.bisect_left([-x for x in allv], -thr, key=None)) / len(uids)


for r in [90, 95, 98, 99]:
    thr = gs[math.floor((1 - r / 100) * n)][0]
    print(f"recall {r}%: ngưỡng {thr:.2f} → giữ {sum(x >= thr for x in allv) / len(uids):.1f} từ/câu")

k = math.ceil(0.01 * n) + 1
tail = gs[:k]
print(f"\n{k} từ đúng điểm thấp nhất (quyết định @99):")
nhyp = nshort = 0
for s, u, w in tail:
    r = next(i for i, (x, _) in enumerate(rank[u]) if x == w) + 1
    inh = w in hyp.get(u, set())
    nhyp += inh
    nshort += len(ph.get(w, "").split()) <= 3
    print(f"  {s:7.2f}  hạng {r:4d}  {w:15s} [{ph.get(w, '?')}]  Qwen chép đúng={inh}  | {text[u][:90]}")
print(f"\nQwen lần 1 chép đúng: {nhyp}/{k}; ≤3 âm vị: {nshort}/{k}")
print("độ dài âm vị của từ đúng (toàn bộ):", sorted(Counter(len(ph.get(w, '').split()) for _, _, w in gs).items())[:8])
