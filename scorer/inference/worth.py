"""Điểm "đáng bias" = có trong câu (scorer) × Qwen lần 1 CHƯA chép ra (từ khó). Không cần train thêm.

  python -m inference.worth --prob work/out/pxattn/test-clean/prob_N2000.json \\
      --hyp ../bench/out/test-clean/qwen3asr_none.tsv --ref ../bench/protocol/test-clean/ref.tsv \\
      --thr 0.5 --name pxattn_N2000

Giữ từ w nếu P(w có trong câu) ≥ thr VÀ w (chuẩn hoá Whisper) không có trong hyp lần 1.
Lý do: từ Qwen đã chép đúng thì nhắc thêm không giúp gì, chỉ làm list dài ra.          [C]
Đầu ra: bench/protocol/<split>/ctx_worth_<name>_p<thr>.json (thứ tự tăng dần theo điểm, như Hou25)
In ra: so với "từ khó thật" (từ hiếm của ref mà hyp lần 1 bỏ sót) → Precision / Recall / AvgLen.
"""
import argparse, json, sys

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from common import BENCH

sys.path.insert(0, str(BENCH))
from textnorm import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prob", required=True)
ap.add_argument("--hyp", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--thr", type=float, nargs="+", default=[0.5])
ap.add_argument("--name", required=True)
a = ap.parse_args()

wn = EnglishTextNormalizer(json.load(open(BENCH / "data" / "whisper_normalizer.json", encoding="utf-8")))
in_hyp = lambda w, hw: set(wn(w).split()) <= hw
hyp = {}
for line in open(a.hyp, encoding="utf-8"):
    uid, _, text = line.rstrip("\n").partition("\t")
    hyp[uid] = set(wn(norm(text)).split())
hard = {}
for line in open(a.ref, encoding="utf-8"):
    uid, _, rare = line.rstrip("\n").split("\t")
    hard[uid] = {w for w in json.loads(rare) if not in_hyp(w, hyp[uid])}
prob = json.load(open(a.prob, encoding="utf-8"))
split = a.ref.replace("\\", "/").split("/")[-2]

n_hard = sum(len(hard[u]) for u in prob)
print(f"{a.name}: từ khó thật {n_hard} lượt / {len(prob)} câu")
print(f"{'ngưỡng':>7} {'Precision':>10} {'Recall':>8} {'AvgLen':>7} {'Rỗng':>7}")
for thr in a.thr:
    keep = {u: [w for w, p in ps if p >= thr and not in_hyp(w, hyp[u])] for u, ps in prob.items()}
    tp = sum(len(set(keep[u]) & hard[u]) for u in keep)
    n = sum(map(len, keep.values()))
    print(f"{thr:>7.2f} {100 * tp / max(n, 1):>10.2f} {100 * tp / max(n_hard, 1):>8.2f} {n / len(keep):>7.2f} "
          f"{100 * sum(not v for v in keep.values()) / len(keep):>6.1f}%")
    json.dump({u: ws[::-1] for u, ws in keep.items()},
              open(BENCH / "protocol" / split / f"ctx_worth_{a.name}_p{thr:g}.json", "w", encoding="utf-8"),
              ensure_ascii=False)
