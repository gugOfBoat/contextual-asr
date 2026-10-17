"""Đo bộ lọc độ liên quan, tách khỏi ASR: với mỗi ngưỡng xác suất, list giữ lại tốt cỡ nào.

  python -m inference.eval_filter --prob work/out/pxattn/test-clean/prob_N2000.json \\
      --ref ../bench/protocol/test-clean/ref.tsv --thr 0.3 0.5 0.7 0.9

  Precision = từ giữ lại đúng / từ giữ lại;  Recall = từ đúng được giữ / tổng từ đúng
  Rỗng      = % câu bị lọc hết.  Với list missing_N (không có từ đúng): Precision luôn 0,
              số đáng xem là AvgLen và Rỗng — lý tưởng là 0 từ, 100% rỗng.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--prob", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
a = ap.parse_args()

gold = {}
for line in open(a.ref, encoding="utf-8"):
    uid, _, rare = line.rstrip("\n").split("\t")
    gold[uid] = set(json.loads(rare))
prob = json.load(open(a.prob, encoding="utf-8"))

print(f"{'ngưỡng':>7} {'Precision':>10} {'Recall':>8} {'AvgLen':>7} {'Rỗng':>7}")
for thr in a.thr:
    kept = {u: {w for w, p in ps if p >= thr} for u, ps in prob.items()}
    tp = sum(len(kept[u] & gold[u]) for u in kept)
    n_kept = sum(map(len, kept.values()))
    n_gold = sum(len(gold[u]) for u in kept)
    print(f"{thr:>7.2f} {100 * tp / max(n_kept, 1):>10.2f} {100 * tp / max(n_gold, 1):>8.2f} "
          f"{n_kept / len(kept):>7.2f} {100 * sum(not v for v in kept.values()) / len(kept):>6.1f}%")
