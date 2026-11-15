"""Phân tích đuôi khó: từ đúng KHÔNG lọt top-50 (list IS21 N2000) theo đặc điểm của từ.
python scorer/analysis/tail.py <split>   (CPU, đọc prob_N2000.json top-50 có sẵn)
"""
import json, sys, collections, random
R = "D:/Lâm/research-asr"
split = sys.argv[1] if len(sys.argv) > 1 else "test-other"
SYS = {
    "hou25": f"{R}/scorer/runs/2026-10-18/out/hou25/{split}/prob_N2000.json",
    "pmaxsim": f"{R}/scorer/runs/2026-10-18/out/pmaxsim/{split}/prob_N2000.json",
    "pxattn": f"{R}/scorer/runs/2026-10-18/out/pxattn/{split}/prob_N2000.json",
}
top = {}
for k, p in SYS.items():
    try:
        d = json.load(open(p, encoding="utf-8"))
    except FileNotFoundError:
        continue
    top[k] = {u: {w for w, _ in v} for u, v in d.items()}

ph = {}
for l in open(f"{R}/scorer/runs/2026-10-16/phones.tsv", encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t"); ph[w] = len(p.split())
vocab = set(open(f"{R}/bench/data/is21/all_rare_words.txt", encoding="utf-8").read().split())
ref = {}
for l in open(f"{R}/bench/protocol/{split}/ref.tsv", encoding="utf-8"):
    u, t, b = l.rstrip("\n").split("\t"); ref[u] = (t.split(), json.loads(b))
al = collections.defaultdict(dict); dur = collections.defaultdict(float)
for l in open(f"{R}/bench/protocol/{split}/align.tsv", encoding="utf-8"):
    u, i, w, s, e = l.rstrip("\n").split("\t"); al[u][int(i)] = (w, float(s), float(e)); dur[u] = max(dur[u], float(e))
hyp = {}
for name in ("whisper_none", "qwen3asr_none"):
    h = {}
    for l in open(f"{R}/bench/out/{split}/{name}.tsv", encoding="utf-8"):
        x = l.rstrip("\n").split("\t"); h[x[0]] = set(x[1].split()) if len(x) > 1 else set()
    hyp[name] = h


def b(x, cuts, labels):
    for c, lab in zip(cuts, labels):
        if x < c:
            return lab
    return labels[-1]


rows = []
for u, (words, rare) in ref.items():
    if u not in top.get("hou25", {}):
        continue
    for w in set(rare):
        idx = words.index(w) if w in words else None
        a = al[u].get(idx) if idx is not None else None
        f = {
            "âm vị": b(ph.get(w, 0), [4, 6, 9], ["≤3", "4-5", "6-8", "≥9"]) if w in ph else "không có g2p",
            "dài từ (s)": b(a[2] - a[1], [0.25, 0.4, 0.6], ["<0.25", "0.25-0.4", "0.4-0.6", "≥0.6"]) if a else "không căn được",
            "bắt đầu (s)": b(a[1], [10, 20, 28, 30], ["<10", "10-20", "20-28", "28-30", "≥30"]) if a else "không căn được",
            "dài câu (s)": b(dur[u], [10, 20, 30], ["<10", "10-20", "20-30", "≥30"]),
            "số từ hiếm/câu": b(len(set(rare)), [2, 5, 10], ["1", "2-4", "5-9", "≥10"]),
            "Whisper không prompt": "đúng" if w in hyp["whisper_none"].get(u, set()) else "sai",
            "Qwen không prompt": "đúng" if w in hyp["qwen3asr_none"].get(u, set()) else "sai",
            "trong kho 209k": "có" if w in vocab else "không",
            "có dấu ' hoặc -": "có" if ("'" in w or "-" in w) else "không",
        }
        miss = {k: w not in top[k].get(u, set()) for k in top}
        rows.append((u, w, f, miss))

print(f"## {split}: {len(rows)} lượt từ đúng (từ hiếm, khác nhau trong câu)\n")
print("Tỉ lệ sót (%) = không lọt top-50 của list N2000\n")
print("| hệ | tổng sót |"); print("|---|---|")
for k in top:
    n = sum(r[3][k] for r in rows); print(f"| {k} | {100*n/len(rows):.2f} ({n}) |")
allm = [r for r in rows if all(r[3].values())]
print(f"\nSót ở **cả {len(top)} hệ**: {len(allm)} ({100*len(allm)/len(rows):.2f}%). "
      f"Chỉ Hou25 sót: {sum(1 for r in rows if r[3]['hou25'] and not any(v for k, v in r[3].items() if k != 'hou25'))}.\n")
for feat in rows[0][2]:
    cnt = collections.Counter(r[2][feat] for r in rows)
    print(f"\n### {feat}\n")
    print("| nhóm | số lượt | " + " | ".join(f"sót {k} %" for k in top) + " | phần trong 'sót cả 3' % |")
    print("|---" * (len(top) + 3) + "|")
    for g in sorted(cnt, key=lambda g: str(g)):
        rs = [r for r in rows if r[2][feat] == g]
        cells = " | ".join(f"{100*sum(r[3][k] for r in rs)/len(rs):.1f}" for k in top)
        share = 100 * sum(1 for r in allm if r[2][feat] == g) / max(len(allm), 1)
        print(f"| {g} | {len(rs)} | {cells} | {share:.0f} |")
random.seed(0)
print("\n### 25 ví dụ sót ở cả 3 hệ (từ — âm vị, dài từ, bắt đầu, Whisper/Qwen không prompt)\n")
for u, w, f, _ in random.sample(allm, min(25, len(allm))):
    print(f"- {u} **{w}** — {f['âm vị']}, {f['dài từ (s)']}, {f['bắt đầu (s)']}, W:{f['Whisper không prompt']} Q:{f['Qwen không prompt']}, kho:{f['trong kho 209k']}")
