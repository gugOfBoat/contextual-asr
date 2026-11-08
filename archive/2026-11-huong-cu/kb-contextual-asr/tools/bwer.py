#!/usr/bin/env python3
"""Tính WER / B-WER / U-WER theo đúng định nghĩa gốc của Le et al., Interspeech 2021.

Quy ước (trích bản full-text arXiv:2104.02194):
  - U-WER đo trên các từ KHÔNG có trong bias list; B-WER đo trên các từ CÓ trong bias list.
  - Lỗi thay thế/xoá phân loại theo từ ở REFERENCE.
  - Lỗi chèn tính vào B-WER nếu từ được chèn nằm trong bias list, ngược lại tính vào U-WER.
  - Cộng dồn tử/mẫu trên toàn tập rồi mới chia (micro-average), không lấy trung bình theo câu.

Cách dùng:
    python bwer.py --ref ref.tsv --hyp hyp.tsv --bias bias.txt
    python bwer.py --ref ref.tsv --hyp hyp.tsv --bias-json bias_per_utt.json

  ref.tsv / hyp.tsv : mỗi dòng "utt_id<TAB>văn bản"
  bias.txt         : mỗi dòng một cụm bias (dùng chung cho mọi câu)
  bias_per_utt.json: {"utt_id": ["cụm 1", "cụm 2", ...]}  (bias list riêng từng câu)
"""
import argparse, json, sys, unicodedata
from collections import Counter

PUNCT = '.,!?;:"“”‘’()[]{}'


def normalize(s, lower=True, strip_punct=True):
    """Chuẩn hoá PHẢI áp giống hệt cho reference, hypothesis và bias list."""
    s = unicodedata.normalize("NFC", s.replace("﻿", ""))   # NFC + bỏ BOM (file Windows)
    if lower:
        s = s.lower()
    if strip_punct:
        s = "".join(" " if c in PUNCT else c for c in s)
    return " ".join(s.split())


def align(ref, hyp):
    """Levenshtein có backtrace. Trả về danh sách op: (kind, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("cor", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    return ops[::-1]


def score(pairs, bias_of):
    """pairs: [(utt_id, ref_text, hyp_text)]; bias_of: utt_id -> set các TỪ thuộc bias list."""
    c = Counter()
    for uid, ref_t, hyp_t in pairs:
        ref, hyp = ref_t.split(), hyp_t.split()
        B = bias_of(uid)
        c["N"] += len(ref)
        c["Nb"] += sum(1 for w in ref if w in B)
        for kind, rw, hw in align(ref, hyp):
            if kind == "cor":
                continue
            if kind == "ins":
                c["Ib" if hw in B else "Iu"] += 1          # phân loại theo từ ở HYPOTHESIS
            else:                                          # sub / del -> theo từ ở REFERENCE
                c[("S" if kind == "sub" else "D") + ("b" if rw in B else "u")] += 1
    c["Nu"] = c["N"] - c["Nb"]
    err_b = c["Sb"] + c["Db"] + c["Ib"]
    err_u = c["Su"] + c["Du"] + c["Iu"]
    pct = lambda a, b: 100.0 * a / b if b else float("nan")
    return {
        "WER": pct(err_b + err_u, c["N"]),
        "B-WER": pct(err_b, c["Nb"]),
        "U-WER": pct(err_u, c["Nu"]),
        "N": c["N"], "N_biased": c["Nb"], "N_unbiased": c["Nu"],
        "S": c["Sb"] + c["Su"], "D": c["Db"] + c["Du"], "I": c["Ib"] + c["Iu"],
        "I_biased": c["Ib"],
    }


def read_tsv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, _, text = line.partition("\t")
            out[uid] = text
    return out


def phrases_to_words(phrases, **kw):
    """Bias list ở mức CỤM -> tập TỪ (mọi từ trong cụm đều là 'biased word')."""
    return {w for p in phrases for w in normalize(p, **kw).split()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--bias")
    ap.add_argument("--bias-json")
    ap.add_argument("--keep-case", action="store_true")
    ap.add_argument("--keep-punct", action="store_true")
    a = ap.parse_args()
    kw = dict(lower=not a.keep_case, strip_punct=not a.keep_punct)

    ref, hyp = read_tsv(a.ref), read_tsv(a.hyp)
    missing = set(ref) - set(hyp)
    if missing:
        print(f"[cảnh báo] {len(missing)} câu thiếu trong hyp, coi như rỗng", file=sys.stderr)

    if a.bias_json:
        per_utt = json.load(open(a.bias_json, encoding="utf-8"))
        bias_of = lambda uid: phrases_to_words(per_utt.get(uid, []), **kw)
    elif a.bias:
        words = phrases_to_words(open(a.bias, encoding="utf-8").read().splitlines(), **kw)
        bias_of = lambda uid: words
    else:
        bias_of = lambda uid: set()

    pairs = [(u, normalize(ref[u], **kw), normalize(hyp.get(u, ""), **kw)) for u in ref]
    for k, v in score(pairs, bias_of).items():
        print(f"{k:12} {v:.2f}" if isinstance(v, float) else f"{k:12} {v}")


if __name__ == "__main__":
    main()
