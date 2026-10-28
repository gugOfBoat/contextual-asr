"""Bước 1 — dựng dữ liệu train/dev (Hou25 §2.2, §3.1).

  python -m data.build_train --libri /data/LibriSpeech
  # thêm tập train lớn (không ghi đè train.jsonl / dev.jsonl; near.json, vocab.txt được bổ sung từ mới):
  python -m data.build_train --libri ... --splits train-clean-100 train-clean-360 train-other-500 --name train960

Đầu ra work/:
  train.jsonl / dev.jsonl   {"uid", "path", "words": mọi từ của câu, "pos": từ hiếm của câu}
                            từ hiếm = ngoài common_words_5k (định nghĩa IS21)
  near.json                 {từ hiếm: 20 từ gần nhất theo edit distance trong kho 209k} → negative khó
  vocab.txt                 mọi từ cần TTS = kho 209k ∪ từ hiếm train/dev ∪ mọi từ trong list test IS21
"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from common import IS21, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--libri", required=True)
ap.add_argument("--n_near", type=int, default=20)
ap.add_argument("--splits", nargs="+", help="các split train gộp lại (mặc định: train-clean-100 + dev-clean như cũ)")
ap.add_argument("--name", default="train")
a = ap.parse_args()

common = set(open(IS21 / "common_words_5k.txt", encoding="utf-8").read().split())
rare = open(IS21 / "all_rare_words.txt", encoding="utf-8").read().split()
WORK.mkdir(exist_ok=True)


def read_split(split):
    items = []
    for trans in sorted((Path(a.libri) / split).rglob("*.trans.txt")):
        for line in open(trans, encoding="utf-8"):
            uid, text = line.strip().split(" ", 1)
            words = text.lower().split()
            items.append({"uid": uid, "path": str(trans.parent / f"{uid}.flac"), "words": sorted(set(words)),
                          "pos": sorted({w for w in words if w not in common})})
    return items


def near(w):
    res = process.extract(w, rare, scorer=Levenshtein.distance, limit=a.n_near + 1)
    return w, [r[0] for r in res if r[0] != w][:a.n_near]


if __name__ == "__main__":
    pos = set()
    jobs = [(a.splits, a.name)] if a.splits else [(["train-clean-100"], "train"), (["dev-clean"], "dev")]
    for splits, name in jobs:
        items = [it for s in splits for it in read_split(s)]
        with open(WORK / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        pos |= {w for it in items for w in it["pos"]}
        print(f"{name}: {len(items)} câu, {sum(len(it['pos']) > 0 for it in items)} câu có từ hiếm")

    old = json.load(open(WORK / "near.json", encoding="utf-8")) if a.splits else {}
    with Pool() as p:
        nb = old | dict(p.map(near, sorted(pos - set(old)), chunksize=64))
    json.dump(nb, open(WORK / "near.json", "w", encoding="utf-8"))

    vocab = set(rare) | pos
    if a.splits:
        vocab |= set(open(WORK / "vocab.txt", encoding="utf-8").read().split())
    for f in IS21.glob("*.biasing_*.tsv"):
        for line in open(f, encoding="utf-8"):
            _, _, r, b = line.rstrip("\n").split("\t")
            vocab |= set(json.loads(r)) | set(json.loads(b))
    open(WORK / "vocab.txt", "w", encoding="utf-8").write("\n".join(sorted(vocab)) + "\n")
    print(f"near: {len(nb)} từ, vocab TTS: {len(vocab)} từ")
