"""Bước 2' — đổi từng từ trong work/vocab.txt thành chuỗi âm vị ARPAbet (cho pmaxsim / pxattn, không cần TTS).

  python -m data.g2p

g2p_en: tra CMUdict, từ không có trong CMUdict thì đoán bằng model seq2seq có sẵn [B]. Bỏ số trọng âm (AH0 → AH).
Gồm cả mọi từ trong list IS21 của bench/protocol/*/ctx_N2000.json (dev chưa có trong vocab) và mọi từ trong bản chép lần 1 của Qwen (work/pass1_*.jsonl, bench/out/*/qwen3asr_none.tsv) cho hxattn.
Đầu ra: work/phones.tsv  (từ \\t âm vị cách nhau bởi dấu cách)
"""
import json, re
from multiprocessing import Pool

from common import BENCH, WORK

_g2p = None


def conv(w):
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return w, " ".join(re.sub(r"\d", "", p) for p in _g2p(w) if p.strip() and p[0].isalpha())


if __name__ == "__main__":
    words = set(open(WORK / "vocab.txt", encoding="utf-8").read().split())
    for f in WORK.glob("pass1_*.jsonl"):
        words |= {w for line in open(f, encoding="utf-8") for w in json.loads(line)["hyp"].split()}
    for f in BENCH.glob("protocol/*/ctx_N2000.json"):
        words |= {w for ws in json.load(open(f, encoding="utf-8")).values() for w in ws}
    for f in BENCH.glob("out/*/qwen3asr_none.tsv"):
        words |= {w for line in open(f, encoding="utf-8") for w in line.rstrip("\n").partition("\t")[2].split()}
    words = sorted(words)
    with Pool() as p:
        rows = p.map(conv, words, chunksize=256)
    with open(WORK / "phones.tsv", "w", encoding="utf-8") as f:
        for w, ph in rows:
            f.write(f"{w}\t{ph}\n")
    print(f"xong {len(rows)} từ, {sum(not ph for _, ph in rows)} từ không ra âm vị")
