"""Chấm WER / U-WER / B-WER bằng scorer IS21 chính thức, có tuỳ chọn chuẩn hoá Whisper.

  python score.py --ref protocol/test-clean/ref.tsv --hyp out/test-clean/whisper_none_sub100.raw.tsv --whisper-norm

--whisper-norm: áp EnglishTextNormalizer của Whisper cho CẢ ref, hyp và từ hiếm
(honour->honor, saint/st, số...). Không bật thì giữ nguyên ref và chỉ dùng textnorm.norm cho hyp.
Chỉ chấm các câu có trong hyp (cho phép chạy trên tập con).
"""
import argparse, json
from pathlib import Path

from score_is21_official import Code, EditDistance, WordError
from textnorm import norm

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--hyp", required=True)
ap.add_argument("--whisper-norm", action="store_true")
a = ap.parse_args()

if a.whisper_norm:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    spell = json.load(open(Path(__file__).parent / "data" / "whisper_normalizer.json", encoding="utf-8"))
    wn = EnglishTextNormalizer(spell)
    f = lambda s: wn(s)
else:
    f = lambda s: s

hyps = {}
for line in open(a.hyp, encoding="utf-8"):
    uid, _, text = line.rstrip("\n").partition("\t")
    hyps[uid] = f(norm(text)) if a.whisper_norm else norm(text)

errs = {k: WordError() for k in ["WER", "U-WER", "B-WER"]}
for line in open(a.ref, encoding="utf-8"):
    uid, text, rare = line.rstrip("\n").split("\t")
    if uid not in hyps:
        continue
    ref = f(text).split()
    bias = {w for r in json.loads(rare) for w in f(r).split()}
    hyp = hyps[uid].split()
    if not ref and not hyp:
        continue
    res = EditDistance().align(ref, hyp)
    for code, ri, hi in zip(res.codes, res.refs, res.hyps):
        if code == Code.insertion:
            errs["WER"].errors[code] += 1
            errs["B-WER" if hyp[hi] in bias else "U-WER"].errors[code] += 1
            continue
        k = "B-WER" if ref[ri] in bias else "U-WER"
        for e in (errs["WER"], errs[k]):
            e.ref_words += 1
            if code != Code.match:
                e.errors[code] += 1

print(f"{len(hyps)} câu | " + " | ".join(f"{k} {e.get_wer():.2f}" for k, e in errs.items())
      + f" | N_bias {errs['B-WER'].ref_words}")
