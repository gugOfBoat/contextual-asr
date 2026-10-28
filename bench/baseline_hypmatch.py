"""Baseline chọn từ KHÔNG train, kiểu CTC-Filter (Yang et al. 2411.06437): so âm vị giữa từng từ trong list
và câu ASR chép lần 1 (không context). Khác bản gốc: dùng hyp của Qwen thay cho CTC thô của WavLM,
so âm vị (g2p) thay vì chữ [C].

  python baseline_hypmatch.py --hyp out/test-clean/qwen3asr_none.tsv --ctx protocol/test-clean/ctx_N2000.json \
      --phones ../scorer/runs/2026-10-16/phones.tsv --out ../scorer/runs/<ngày>/out/hypmatch/test-clean/prob_N2000.json

điểm(từ) = max trên mọi cụm 1–3 từ liên tiếp của hyp của độ giống âm vị = 1 − Levenshtein / độ dài dài hơn ∈ [0, 1]
(= 1 khi từ có nguyên văn trong hyp). Ghi top-50 giảm dần, cùng format prob_*.json của scorer.
"""
import argparse, json
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

ap = argparse.ArgumentParser()
ap.add_argument("--hyp", required=True)
ap.add_argument("--ctx", required=True)
ap.add_argument("--phones", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=50)
ap.add_argument("--span", type=int, default=3)
a = ap.parse_args()

sym = {}
enc = lambda ps: "".join(sym.setdefault(x, chr(0x100 + len(sym))) for x in ps)
phones = {}
for line in open(a.phones, encoding="utf-8"):
    w, p = line.rstrip("\n").split("\t")
    phones[w] = enc(p.split())

ctx = json.load(open(a.ctx, encoding="utf-8"))
out = {}
n_skip = 0
for line in open(a.hyp, encoding="utf-8"):
    uid, _, text = line.rstrip("\n").partition("\t")
    if uid not in ctx:
        continue
    words = [w for w in ctx[uid] if w in phones]
    n_skip += len(ctx[uid]) - len(words)
    hw = text.split()
    spans = {phones[w] for w in hw if w in phones}
    for n in range(2, a.span + 1):
        for i in range(len(hw) - n + 1):
            if all(w in phones for w in hw[i:i + n]):
                spans.add("".join(phones[w] for w in hw[i:i + n]))
    if not words or not spans:
        out[uid] = []
        continue
    sim = cdist([phones[w] for w in words], sorted(spans), scorer=Levenshtein.normalized_similarity, workers=-1).max(1)
    s = {w: float(x) for w, x in zip(words, sim)}
    for w in set(hw) & set(s):
        s[w] = 1.0
    out[uid] = [[w, round(x, 4)] for w, x in sorted(s.items(), key=lambda t: -t[1])[:a.top]]

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(out)} câu → {a.out}; {n_skip} lượt từ trong list không có âm vị (bỏ qua)")
