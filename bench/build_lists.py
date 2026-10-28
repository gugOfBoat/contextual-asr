"""Dựng thêm list cho đánh giá scorer (EVAL_PROTOCOL.md mục 2). Chạy sau build_is21.py.

  python build_lists.py --phones ../scorer/runs/2026-10-16/phones.tsv

1) Dev theo đúng quy tắc IS21 (IS21 chỉ công bố list cho test) -> protocol/dev-{clean,other}/
     ref.tsv, audio.tsv, ctx_none, ctx_oracle, ctx_N{100,500,1000,2000}, ctx_missing_N{n}
     từ hiếm = từ ngoài common_words_5k (đã kiểm: khớp 5559/5559 câu test của IS21)
     distractor = bốc ngẫu nhiên từ all_rare_words, bỏ từ hiếm của câu, mỗi N bốc độc lập, list xếp a-b-c (như IS21)
2) L2 "nhiễu nghe giống" cho cả 4 split -> ctx_hard_N2000.json + near_N2000.json
     near = từ hiếm có khoảng cách Levenshtein âm vị 1..2 với một từ đúng (ngưỡng ≤ 2 theo BR-ASR [A]),
            tối đa K_NEAR từ gần nhất / từ đúng (hoà thì bốc ngẫu nhiên); khoảng cách 0 (đồng âm hoàn toàn) bị loại vì audio không phân biệt được
     list = từ đúng + near + phần còn lại lấy từ list N2000 (IS21 hoặc dev) cho đủ 2000 distractor
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

ROOT = Path(__file__).parent
IS21 = ROOT / "data" / "is21"
LIBRI = ROOT.parent / "datasets" / "original" / "LibriSpeech"
NS = [100, 500, 1000, 2000]
K_NEAR = 20
SEED = 21

ap = argparse.ArgumentParser()
ap.add_argument("--phones", required=True, help="phones.tsv: từ \\t âm vị ARPAbet")
a = ap.parse_args()

common = set(open(IS21 / "common_words_5k.txt", encoding="utf-8").read().split())
pool = open(IS21 / "all_rare_words.txt", encoding="utf-8").read().split()


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)


# 1) dev
rng = random.Random(SEED)
for split in ["dev-clean", "dev-other"]:
    out = ROOT / "protocol" / split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for trans in sorted((LIBRI / split).rglob("*.trans.txt")):
        for line in open(trans, encoding="utf-8"):
            uid, text = line.strip().split(" ", 1)
            text = text.lower()
            rows.append((uid, text, sorted({w for w in text.split() if w not in common})))
    with open(out / "ref.tsv", "w", encoding="utf-8") as f:
        for uid, text, rare in rows:
            f.write(f"{uid}\t{text}\t{json.dumps(rare)}\n")
    with open(out / "audio.tsv", "w", encoding="utf-8") as f:
        for uid, _, _ in rows:
            spk, chap, _ = uid.split("-")
            f.write(f"{uid}\t{LIBRI / split / spk / chap / f'{uid}.flac'}\n")
    dump({u: [] for u, _, _ in rows}, out / "ctx_none.json")
    dump({u: r for u, _, r in rows}, out / "ctx_oracle.json")
    for n in NS:
        dis = {}
        for uid, _, rare in rows:
            s = set(rare)
            d = set()
            while len(d) < n:
                d.update(w for w in rng.sample(pool, n - len(d)) if w not in s)
            dis[uid] = d
        dump({u: sorted(set(r) | dis[u]) for u, _, r in rows}, out / f"ctx_N{n}.json")
        dump({u: sorted(dis[u]) for u, _, r in rows}, out / f"ctx_missing_N{n}.json")
    print(f"{split}: {len(rows)} câu, {sum(len(r) for _, _, r in rows)} từ hiếm, "
          f"{sum(not r for _, _, r in rows)} câu không có từ hiếm")

# 2) L2: nhiễu nghe giống
phones = {}
for line in open(a.phones, encoding="utf-8"):
    w, p = line.rstrip("\n").split("\t")
    phones[w] = p.split()
sym = {}
enc = lambda ps: "".join(sym.setdefault(x, chr(0x100 + len(sym))) for x in ps)
cand = [w for w in pool if w in phones]
cand_s = [enc(phones[w]) for w in cand]

splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
refs = {}
for split in splits:
    refs[split] = {}
    for line in open(ROOT / "protocol" / split / "ref.tsv", encoding="utf-8"):
        uid, _, rare = line.rstrip("\n").split("\t")
        refs[split][uid] = json.loads(rare)
gold = sorted({w for s in refs.values() for r in s.values() for w in r if w in phones})
miss = {w for s in refs.values() for r in s.values() for w in r if w not in phones}
print(f"L2: {len(gold)} từ đúng có âm vị, {len(miss)} không có (bỏ qua)")

near, homo = {}, Counter()
for i in range(0, len(gold), 1000):
    q = gold[i:i + 1000]
    d = cdist([enc(phones[w]) for w in q], cand_s, scorer=Levenshtein.distance,
              score_cutoff=2, dtype=np.int8, workers=-1)
    for w, row in zip(q, d):
        idx = np.nonzero(row <= 2)[0]
        homo[w] = sum(1 for j in idx if row[j] == 0 and cand[j] != w)
        idx = [j for j in idx if row[j] > 0]
        rng.shuffle(idx)  # hoà khoảng cách thì chọn ngẫu nhiên, không thiên về chữ "a"
        idx.sort(key=lambda j: row[j])
        near[w] = [cand[j] for j in idx[:K_NEAR]]

for split in splits:
    out = ROOT / "protocol" / split
    base = json.load(open(out / "ctx_N2000.json", encoding="utf-8"))
    hard, nb = {}, {}
    for uid, rare in refs[split].items():
        s = set(rare)
        h = sorted({x for w in rare for x in near.get(w, []) if x not in s})
        rest = [w for w in base[uid] if w not in s and w not in set(h)]
        rng.shuffle(rest)
        n_dis = len(base[uid]) - len(s & set(base[uid]))
        hard[uid] = sorted(s | set(h) | set(rest[:max(0, n_dis - len(h))]))
        nb[uid] = h
    dump(hard, out / "ctx_hard_N2000.json")
    dump(nb, out / "near_N2000.json")
    n_g = sum(len(r) for r in refs[split].values())
    print(f"{split}: hard_N2000 trung bình {np.mean([len(v) for v in nb.values() if v] or [0]):.1f} từ gần/câu có từ hiếm; "
          f"{sum(homo[w] > 0 for r in refs[split].values() for w in r)}/{n_g} từ đúng có từ đồng âm hoàn toàn trong kho (đã loại)")
