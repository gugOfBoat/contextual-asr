"""Từ nhiễu đứng trên từ đúng: nó nghe giống MỘT ĐOẠN nào đó trong câu (scorer nghe đúng âm, sai từ), hay không giống gì (báo động giả)?
Khoảng cách = edit âm vị nhỏ nhất giữa từ nhiễu và một đoạn liên tiếp bất kỳ trong chuỗi âm vị của câu ref, chia cho số âm vị từ nhiễu."""
import json, sys, collections
R = "D:/Lâm/research-asr"; split = sys.argv[1]
ph = {}
for l in open(f"{R}/scorer/runs/2026-10-16/phones.tsv", encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t"); ph[w] = p.split()
ref = {}
for l in open(f"{R}/bench/protocol/{split}/ref.tsv", encoding="utf-8"):
    u, t, b = l.rstrip("\n").split("\t"); ref[u] = (t.split(), set(json.loads(b)))
def sub_ed(q, s):  # edit nhỏ nhất của q so với một đoạn con của s (semi-global)
    prev = [0] * (len(s) + 1)
    for i, a in enumerate(q, 1):
        cur = [i] + [0] * len(s)
        for j, b in enumerate(s, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return min(prev)
for k in ("hou25", "pmaxsim", "pxattn"):
    try: d = json.load(open(f"{R}/scorer/runs/2026-10-18/out/{k}/{split}/prob_N2000.json", encoding="utf-8"))
    except FileNotFoundError: continue
    b = collections.Counter(); n = 0; ex = []
    for u, lst in d.items():
        words, gold = ref[u]
        if not gold: continue
        sent = sum((ph.get(w, []) for w in words), [])
        gi = [i for i, (w, _) in enumerate(lst) if w in gold]
        last = max(gi) if gi else 50
        for i, (w, p) in enumerate(lst):  # từ nhiễu đứng TRÊN một từ đúng và có p >= 0.5
            if w in gold: continue
            if i >= last or p < 0.5: break
            q = ph.get(w)
            if not q: continue
            r = sub_ed(q, sent) / len(q); n += 1
            g = "≤0.2 (gần như có trong câu)" if r <= 0.2 else "0.2–0.4" if r <= 0.4 else "> 0.4 (không giống đoạn nào)"
            b[g] += 1
            if r <= 0.2 and len(ex) < 8: ex.append(w)
    print(f"{k}: {n} lượt từ nhiễu p≥0.5 đứng trên từ đúng | " + " | ".join(f"{g}: {100*c/n:.0f}%" for g, c in sorted(b.items())) + f" | vd: {', '.join(ex)}")
