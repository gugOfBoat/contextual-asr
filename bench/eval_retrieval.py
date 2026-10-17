"""Đo bước tạo candidate (bias discovery), tách khỏi ASR.

  python eval_retrieval.py --ref protocol/test-clean/ref.tsv --cand cand.json --ks 10 20 50 100
  cand.json: {uid: [candidate xếp hạng giảm dần]}

Recall@K = số từ hiếm đúng nằm trong top-K / tổng số từ hiếm (micro, chỉ tính câu có từ hiếm).
MRR      = trung bình 1/rank của từng từ hiếm đúng (0 nếu không có trong list).
AvgLen   = độ dài list trung bình (list dài thì recall dễ cao nhưng bias loãng).
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--cand", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
a = ap.parse_args()

gold = {}
for line in open(a.ref, encoding="utf-8"):
    uid, _, rare = line.rstrip("\n").split("\t")
    if json.loads(rare):
        gold[uid] = json.loads(rare)
cand = json.load(open(a.cand, encoding="utf-8"))

total = sum(len(g) for g in gold.values())
hits = {k: 0 for k in a.ks}
rr = 0.0
for uid, words in gold.items():
    rank = {w: i + 1 for i, w in reversed(list(enumerate(cand.get(uid, []))))}
    for w in words:
        r = rank.get(w)
        if r:
            rr += 1 / r
            for k in a.ks:
                hits[k] += r <= k
for k in a.ks:
    print(f"Recall@{k:<5} {100 * hits[k] / total:.2f}")
print(f"MRR        {rr / total:.4f}")
print(f"AvgLen     {sum(len(cand.get(u, [])) for u in gold) / len(gold):.1f}")
print(f"gold words {total} trong {len(gold)} câu")
