"""Dựng protocol LibriSpeech + IS21 cho benchmark.

Đầu ra (bench/protocol/<split>/):
  ref.tsv               uid \t text \t rare_words(json)   -> đúng format score_is21_official.py
  audio.tsv             uid \t đường dẫn .flac
  ctx_none.json         {uid: []}                          không context
  ctx_oracle.json       {uid: rare words của câu}          cận trên
  ctx_N{n}.json         {uid: rare + n distractor}         đúng setting Hou25 / PTC-Bias
  ctx_missing_N{n}.json {uid: chỉ n distractor}            list KHÔNG chứa từ đúng
Kho toàn cục cho setting "không có list": data/is21/all_rare_words.txt (209.291 từ).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
IS21 = ROOT / "data" / "is21"
LIBRI = ROOT.parent / "datasets" / "original" / "LibriSpeech"
NS = [100, 500, 1000, 2000]


def load(split, n):
    rows = {}
    for line in open(IS21 / f"{split}.biasing_{n}.tsv", encoding="utf-8"):
        uid, text, rare, blist = line.rstrip("\n").split("\t")
        rows[uid] = (text, json.loads(rare), json.loads(blist))
    return rows


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)


for split in ["test-clean", "test-other"]:
    out = ROOT / "protocol" / split
    out.mkdir(parents=True, exist_ok=True)
    base = load(split, NS[0])

    with open(out / "ref.tsv", "w", encoding="utf-8") as f:
        for uid, (text, rare, _) in base.items():
            f.write(f"{uid}\t{text}\t{json.dumps(rare)}\n")

    n_audio = 0
    with open(out / "audio.tsv", "w", encoding="utf-8") as f:
        for uid in base:
            spk, chap, _ = uid.split("-")
            p = LIBRI / split / spk / chap / f"{uid}.flac"
            n_audio += p.exists()
            f.write(f"{uid}\t{p}\n")

    dump({u: [] for u in base}, out / "ctx_none.json")
    dump({u: r for u, (_, r, _) in base.items()}, out / "ctx_oracle.json")
    for n in NS:
        rows = load(split, n)
        dump({u: b for u, (_, _, b) in rows.items()}, out / f"ctx_N{n}.json")
        dump({u: [w for w in b if w not in set(r)] for u, (_, r, b) in rows.items()},
             out / f"ctx_missing_N{n}.json")
    print(f"{split}: {len(base)} câu, audio có sẵn {n_audio}/{len(base)}")
