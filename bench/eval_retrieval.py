"""Tầng (a) — đo riêng bước chọn từ (scorer), tách khỏi ASR. Metric theo BR-ASR (2505.19179, Bảng 2) / PTC-Bias (2609.28727, Bảng 3).

  python eval_retrieval.py --ref protocol/test-other/ref.tsv --scores prob_N2000.json \
      --ctx protocol/test-other/ctx_N2000.json --hyp out/test-other/qwen3asr_none.tsv --thr 0.5 0.7
  python eval_retrieval.py --ref ... --scores prob_hard_N2000.json --ctx .../ctx_hard_N2000.json --near .../near_N2000.json

--scores: {uid: [[từ, điểm], ...]} hoặc {uid: [từ, ...]}, xếp giảm dần (file prob_*.json của scorer chỉ giữ top-50).
Từ đúng = từ hiếm của reference (cột 3 ref.tsv) ∩ list (--ctx) → list missing_N cho 0 từ đúng, chỉ đo "giữ nhầm".

Recall_B#K   % từ đúng nằm trong top-K (micro)                                       [A]
Recall_B@99  số từ trung bình / câu phải giữ để đạt 99% recall: một ngưỡng điểm chung cho cả tập,
             tính trên mọi câu kể cả câu không có từ đúng. Cần điểm số. [A] tên metric; cách tính là ta diễn giải
K99          K cố định nhỏ nhất để Recall_B#K ≥ 99% (bản không cần điểm)                 [C]
Recall_H#K   % từ "nghe gần giống" lọt top-K — thấp = tốt                                 [A]
             --phones: H = từ trong list có Levenshtein âm vị ≤ 2 với một từ đúng (BR-ASR Eq. 4, khớp PTC Bảng 3)
             --near:   H cho sẵn (bản cũ của nhóm, 20 từ gần nhất trong kho)                [C]
@thr         Precision / Recall / số từ giữ mỗi câu / % câu rỗng khi giữ điểm ≥ thr     [C]
hard         Recall_B#K và Recall@thr chỉ trên từ khó = từ đúng không có trong hyp lần 1 (--hyp, chuẩn hoá Whisper) [C]
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--scores", "--cand", dest="scores", required=True)
ap.add_argument("--ctx", help="list đã đưa cho scorer; mặc định coi mọi từ hiếm của ref là có trong list")
ap.add_argument("--near", help="near_*.json: {uid: [từ nghe gần giống]}")
ap.add_argument("--phones", help="phones.tsv: tính H theo BR-ASR từ chính list (--ctx), thay cho --near")
ap.add_argument("--hyp", help="hyp ASR không context (uid \\t text) để xác định từ khó")
ap.add_argument("--ks", type=int, nargs="+", default=[10, 50])
ap.add_argument("--thr", type=float, nargs="*", default=[])
a = ap.parse_args()

ref = {}
for line in open(a.ref, encoding="utf-8"):
    uid, _, rare = line.rstrip("\n").split("\t")
    ref[uid] = json.loads(rare)
scores = json.load(open(a.scores, encoding="utf-8"))
ctx = json.load(open(a.ctx, encoding="utf-8")) if a.ctx else None
near = json.load(open(a.near, encoding="utf-8")) if a.near else None
uids = [u for u in ref if u in scores]

ranked, scored = {}, True
for u in uids:
    v = scores[u]
    if v and isinstance(v[0], str):
        scored = False
        ranked[u] = [(w, -i) for i, w in enumerate(v)]
    else:
        ranked[u] = [(w, s) for w, s in v]
gold = {u: [w for w in ref[u] if ctx is None or w in set(ctx[u])] for u in uids}

hard = None
if a.hyp:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    wn = EnglishTextNormalizer(json.load(open(Path(__file__).parent / "data" / "whisper_normalizer.json", encoding="utf-8")))
    hw = {}
    for line in open(a.hyp, encoding="utf-8"):
        uid, _, text = line.rstrip("\n").partition("\t")
        hw[uid] = set(wn(text).split())
    hard = {u: [w for w in gold[u] if not set(wn(w).split()) <= hw.get(u, set())] for u in uids}

n_gold = sum(len(g) for g in gold.values())
print(f"{a.scores}: {len(uids)} câu, {n_gold} từ đúng" + (f", {sum(len(h) for h in hard.values())} từ khó" if hard else ""))


def recall_at(sets, k):
    tot = sum(len(g) for g in sets.values())
    if not tot:
        return float("nan")
    hit = sum(sum(w in {x for x, _ in ranked[u][:k]} for w in g) for u, g in sets.items())
    return 100 * hit / tot


row = []
if n_gold:
    for k in a.ks:
        row.append(f"Recall_B#{k} {recall_at(gold, k):.2f}")
    # K99: K cố định nhỏ nhất đạt 99%
    rank = [next((i + 1 for i, (x, _) in enumerate(ranked[u]) if x == w), math.inf) for u in uids for w in gold[u]]
    rank.sort()
    k99 = rank[math.ceil(0.99 * len(rank)) - 1]
    row.append(f"K99 {k99 if k99 < math.inf else 'n/a'}")
    # Recall_B@99: một ngưỡng chung, đếm số từ giữ trung bình
    if scored:
        allv = sorted(((s, (w in set(gold[u]))) for u in uids for w, s in ranked[u]), key=lambda t: -t[0])
        need, hit = math.ceil(0.99 * n_gold), 0
        at = None
        for i, (_, g) in enumerate(allv):
            hit += g
            if hit >= need:
                at = (i + 1) / len(uids)
                break
        row.append(f"Recall_B@99 {at:.1f}" if at else "Recall_B@99 n/a (không đạt 99% trong top đã lưu)")
if a.phones:  # H theo BR-ASR (Eq. 4 + chú thích 2): từ trong list có Levenshtein âm vị ≤ 2 với 1 từ đúng
    from rapidfuzz.distance import Levenshtein
    ph = {}
    for line in open(a.phones, encoding="utf-8"):
        w, p = line.rstrip("\n").split("\t")
        ph[w] = p.split()
    near = {u: [w for w in ctx[u] if w in ph and w not in gold[u]
                and any(Levenshtein.distance(ph[w], ph[g], score_cutoff=2) <= 2 for g in gold[u] if g in ph)]
            for u in uids}
if near:
    nb = {u: [w for w in near.get(u, []) if ctx is None or w in set(ctx[u])] for u in uids}
    row.append(f"N_H {sum(len(v) for v in nb.values())}")
    for k in a.ks:
        row.append(f"Recall_H#{k} {recall_at(nb, k):.2f}")
print(" | ".join(row))

if hard and any(hard.values()):
    print("từ khó: " + " | ".join(f"Recall_B#{k} {recall_at(hard, k):.2f}" for k in a.ks))

for t in a.thr:
    kept = {u: {w for w, s in ranked[u] if s >= t} for u in uids}
    n_kept = sum(len(v) for v in kept.values())
    tp = sum(len(kept[u] & set(gold[u])) for u in uids)
    line = (f"@{t}: P {100 * tp / n_kept if n_kept else float('nan'):.1f} | R {100 * tp / n_gold if n_gold else float('nan'):.1f}"
            f" | từ/câu {n_kept / len(uids):.2f} | rỗng {100 * sum(not v for v in kept.values()) / len(uids):.1f}%")
    if hard and any(hard.values()):
        nh = sum(len(h) for h in hard.values())
        line += f" | R từ khó {100 * sum(len(kept[u] & set(hard[u])) for u in uids) / nh:.1f}"
    print(line)
