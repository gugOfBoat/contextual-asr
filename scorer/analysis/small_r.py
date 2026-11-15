"""Đo lại scorer với R nhỏ + chia nhóm lỗi: từ đúng bị từ nhiễu nào vượt lên, và vì sao.
python small_r.py <split>   (dùng top-50 trong prob_N2000.json của runs/2026-10-18, chạy CPU)

d(w) = số từ NHIỄU đứng trên từ đúng w (không tính các từ đúng khác, để câu nhiều từ hiếm không bị phạt oan).
"Sạch@K" = % từ đúng có d < K, tức nằm trong top-K nếu chỉ tính nhiễu. Ngoài top-50 thì d >= 50.
"""
import json, sys, collections
from rapidfuzz.distance import Levenshtein

R = "D:/Lâm/research-asr"
split = sys.argv[1] if len(sys.argv) > 1 else "test-other"
SYS = [s for s in ("hou25", "pmaxsim", "pxattn")]
top = {}
for k in SYS:
    try:
        top[k] = json.load(open(f"{R}/scorer/runs/2026-10-18/out/{k}/{split}/prob_N2000.json", encoding="utf-8"))
    except FileNotFoundError:
        pass
ph = {}
for l in open(f"{R}/scorer/runs/2026-10-16/phones.tsv", encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t"); ph[w] = p.split()
ref = {}
for l in open(f"{R}/bench/protocol/{split}/ref.tsv", encoding="utf-8"):
    u, t, b = l.rstrip("\n").split("\t"); ref[u] = set(json.loads(b))
al = collections.defaultdict(dict); dur = collections.defaultdict(float)
for l in open(f"{R}/bench/protocol/{split}/align.tsv", encoding="utf-8"):
    u, i, w, s, e = l.rstrip("\n").split("\t"); al[u][w] = (float(s), float(e)); dur[u] = max(dur[u], float(e))
wh = {}
for l in open(f"{R}/bench/out/{split}/whisper_none.tsv", encoding="utf-8"):
    x = l.rstrip("\n").split("\t"); wh[x[0]] = set(x[1].split()) if len(x) > 1 else set()


def pdist(a, b):
    if a in ph and b in ph:
        return Levenshtein.distance(ph[a], ph[b])
    return 99


def group(u, w, top_noise):
    """Nhóm lỗi của từ đúng w, dựa vào từ nhiễu cao nhất đứng trên nó."""
    s, e = al[u].get(w, (None, None))
    if s is not None and s >= 30:
        return "sau giây 30 (encoder cắt)"
    if top_noise is None:
        return "?"
    d = pdist(w, top_noise)
    if d == 0:
        return "bị từ ĐỒNG ÂM vượt"
    if d <= 2:
        return "bị từ GẦN ÂM (≤2 âm vị) vượt"
    return "bị từ KHÁC HẲN vượt"


print(f"## {split}\n")
print("### 1. Độ sạch: % từ đúng có ít hơn K từ nhiễu đứng trên\n")
print("| hệ | lượt | Sạch@1 | Sạch@3 | Sạch@5 | Sạch@10 | Sạch@50 | TB từ nhiễu p≥0.5 / câu |")
print("|---|---|---|---|---|---|---|---|")
rows = {}
for k, d in top.items():
    rs = []
    noise05 = []
    for u, lst in d.items():
        gold = ref.get(u, set())
        noise05.append(sum(1 for w, p in lst if p >= 0.5 and w not in gold))
        above, first_noise = 0, None
        pos = {}
        for w, p in lst:
            if w in gold:
                pos[w] = (above, first_noise)
            else:
                above += 1
                first_noise = first_noise or w
        for w in gold:
            a, fn = pos.get(w, (50, first_noise))
            rs.append((u, w, a, fn))
    rows[k] = rs
    c = lambda K: 100 * sum(1 for r in rs if r[2] < K) / len(rs)
    print(f"| {k} | {len(rs)} | {c(1):.1f} | {c(3):.1f} | {c(5):.1f} | {c(10):.1f} | {c(50):.1f} | {sum(noise05)/len(noise05):.1f} |")

print("\n### 2. Từ đúng KHÔNG đứng đầu (có ≥1 từ nhiễu trên nó): nhóm lỗi theo từ nhiễu cao nhất\n")
for k, rs in rows.items():
    bad = [r for r in rs if r[2] >= 1]
    cnt = collections.Counter(group(u, w, fn) for u, w, a, fn in bad)
    print(f"**{k}** ({len(bad)} lượt, {100*len(bad)/len(rs):.1f}% số từ đúng)\n")
    print("| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |")
    print("|---|---|---|---|")
    for g, n in cnt.most_common():
        sub = [r for r in bad if group(r[0], r[1], r[3]) == g]
        wrong = 100 * sum(1 for u, w, *_ in sub if w not in wh.get(u, set())) / len(sub)
        print(f"| {g} | {n} | {100*n/len(bad):.0f} | {wrong:.0f} |")
    print()

print("### 3. Từ nhiễu hay đứng đầu nhất (đứng trên từ đúng, đếm số câu)\n")
for k, rs in rows.items():
    c = collections.Counter(fn for u, w, a, fn in rs if a >= 1 and fn)
    print(f"- {k}: " + ", ".join(f"{w} ({n})" for w, n in c.most_common(15)))
print("\n### 4. Ví dụ (hou25): từ đúng ← từ nhiễu cao nhất, khoảng cách âm vị\n")
import random
random.seed(1)
for u, w, a, fn in random.sample([r for r in rows["hou25"] if r[2] >= 1], 20):
    print(f"- {u} **{w}** (d={a}) ← {fn} [{pdist(w, fn)}]")
