"""Bảng E1: WER / U-WER / B-WER (wnorm) + số từ của list N2000 bị chèn sai cho nhiều output cùng lúc.

  python e1_table.py --split test-other out/test-other/whisper_oracle_plain.tsv out/test-other/whisper_oracle_prompt.tsv ...

Chèn sai = từ thuộc list N2000 của câu xuất hiện trong hyp nhiều hơn trong ref (đếm sau textnorm.norm).
Dùng cùng list N2000 cho mọi hệ để so được với nhau (list oracle là tập con của N2000).
"""
import argparse, json
from collections import Counter
from pathlib import Path

from score import KEYS, rate, utt_errors
from textnorm import norm

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--subset")
ap.add_argument("hyps", nargs="+")
a = ap.parse_args()

proto = Path(__file__).parent / "protocol" / a.split
ref = {l.split("\t")[0]: norm(l.split("\t")[1]).split() for l in open(proto / "ref.tsv", encoding="utf-8")}
lst = {u: set(ws) for u, ws in json.load(open(proto / "ctx_N2000.json", encoding="utf-8")).items()}
keep = set(open(a.subset).read().split()) if a.subset else None
print(f"{'hệ':45s} {'câu':>5s} " + " ".join(f"{k:>6s}" for k in KEYS) + f" {'chèn sai':>8s}")
for h in a.hyps:
    errs = utt_errors(proto / "ref.tsv", h, True)
    if keep:
        errs = {u: e for u, e in errs.items() if u in keep}
    ins = 0
    for l in open(h, encoding="utf-8"):
        uid, _, text = l.rstrip("\n").partition("\t")
        if uid not in errs:
            continue
        hc, rc = Counter(norm(text).split()), Counter(ref[uid])
        ins += sum(max(0, hc[w] - rc[w]) for w in lst.get(uid, ()) if w in hc)
    print(f"{Path(h).stem:45s} {len(errs):5d} " + " ".join(f"{rate(errs, k):6.2f}" for k in KEYS) + f" {ins:8d}")
