"""Whisper dùng bias word thế nào: từ đúng có còn trong prompt sau khi cắt 223 token cuối không,
có được chép ra không (khớp nguyên từ trong hyp, xấp xỉ), và từ nhiễu trong prompt có bị chèn vào không."""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

B = Path(sys.argv[1])          # bench/
tok = Tokenizer.from_file(sys.argv[2])
CFG = ["none", "N100", "N1000", "N2000", "pxattn_N100", "pxattn_N2000", "oracle"]


def kept(words):               # như run_whisper.py: get_prompt_ids(" w1, w2...") rồi giữ 223 token cuối
    if not words:
        return set()
    ids = tok.encode(" " + ", ".join(words), add_special_tokens=False).ids
    if len(ids) <= 223:
        return set(words)
    tail = tok.decode(ids[-223:]).split(", ")
    return set(tail[1:])        # bỏ mảnh từ đầu bị cắt dở


for sp in ["test-clean", "test-other"]:
    ref = {}
    for l in open(B / f"protocol/{sp}/ref.tsv", encoding="utf-8"):
        u, t, r = l.rstrip("\n").split("\t")
        ref[u] = (t.split(), json.loads(r))
    base = {}
    for l in open(B / f"out/{sp}/whisper_none.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        base[p[0]] = set(p[1].split()) if len(p) > 1 else set()
    print(f"== {sp}")
    print(f"{'cấu hình':14s} {'từ đúng còn trong prompt':>24s} {'chép đúng | có trong prompt':>28s} {'| không có':>10s} {'từ nhiễu bị chèn':>17s} {'cùng từ, không prompt':>22s}")
    for c in CFG:
        ctx = json.load(open(B / f"protocol/{sp}/ctx_{c}.json", encoding="utf-8")) if c != "none" else {}
        hyp = {}
        for l in open(B / f"out/{sp}/whisper_{c}.tsv", encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            hyp[p[0]] = p[1].split() if len(p) > 1 else []
        n = n_in = hit_in = hit_out = ins = base_in = 0
        for u, (rw, rare) in ref.items():
            ws = ctx.get(u, [])
            k = kept(ws)
            h = set(hyp.get(u, []))
            for w in rare:
                n += 1
                if w in k:
                    n_in += 1
                    hit_in += w in h
                    base_in += w in base[u]
                else:
                    hit_out += w in h
            ins += sum(1 for w in k if w not in rare and w in h and w not in rw)
        n_out = n - n_in
        f = lambda a, b: f"{100 * a / b:5.1f}%" if b else "   -  "
        print(f"{c:14s} {f(n_in, n):>24s} {f(hit_in, n_in):>28s} {f(hit_out, n_out):>10s} {ins:>17d} {f(base_in, n_in):>22s}")
