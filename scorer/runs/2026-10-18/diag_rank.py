"""Phân tích lỗi xếp hạng: từ đúng rớt khỏi top-K là loại gì, từ sai nào chen lên."""
import json, sys, collections
from rapidfuzz.distance import Levenshtein as L

R = "D:/Lâm/research-asr"
split, model = sys.argv[1], sys.argv[2]
K = 10
ph = {}
for l in open(f"{R}/scorer/runs/2026-10-16/phones.tsv", encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t"); ph[w] = p.split()
ref = {}
for l in open(f"{R}/bench/protocol/{split}/ref.tsv", encoding="utf-8"):
    u, t, b = l.rstrip("\n").split("\t"); ref[u] = (t.split(), json.loads(b))
hyp = {}
for l in open(f"{R}/bench/out/{split}/qwen3asr_none.tsv", encoding="utf-8"):
    x = l.rstrip("\n").split("\t"); hyp[x[0]] = x[1].split() if len(x) > 1 else []
ctx = json.load(open(f"{R}/bench/protocol/{split}/ctx_N2000.json", encoding="utf-8"))
prob = json.load(open(f"{R}/scorer/runs/2026-10-16/out/{model}/{split}/prob_N2000.json", encoding="utf-8"))


def nd(a, b):  # khoảng cách âm vị chuẩn hoá
    return L.distance(a, b) / max(len(a), len(b))


def best_span(p, words):  # khớp âm vị gần nhất với cụm 1–3 từ liên tiếp
    best = 9
    for i in range(len(words)):
        s = []
        for j in range(i, min(i + 3, len(words))):
            s = s + ph.get(words[j], [])
            if s:
                best = min(best, nd(p, s))
    return best


def sub(p, t):  # khoảng cách của p tới đoạn con gần nhất của t (căn chỉnh nửa toàn cục), chuẩn hoá theo |p|
    prev = [0] * (len(t) + 1)
    for i in range(1, len(p) + 1):
        cur = [i] + [0] * len(t)
        for j in range(1, len(t) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p[i - 1] != t[j - 1]))
        prev = cur
    return min(prev) / len(p)


cap = collections.Counter()
stat = collections.defaultdict(lambda: [0, 0])
fp_kind = collections.Counter()
n_fp = 0
ex = []
for u, (words, gold) in ref.items():
    g = [w for w in gold if w in set(ctx.get(u, []))]
    if not g:
        continue
    ranked = [w for w, _ in prob.get(u, [])]
    top = ranked[:K]
    hw = set(hyp.get(u, []))
    for w in g:
        ok = w in top
        nph = len(ph.get(w, []))
        b = "≤3" if nph <= 3 else "4-5" if nph <= 5 else "6-8" if nph <= 8 else "≥9"
        for key in [("all",), ("phone_len", b), ("qwen_pass1", "đúng" if w in hw else "sai"),
                    ("utt_len", "≤10" if len(words) <= 10 else "11-30" if len(words) <= 30 else ">30")]:
            stat[key][0] += ok; stat[key][1] += 1
    # từ sai xếp TRÊN một từ đúng bị rớt khỏi top-K: loại gì
    miss = [w for w in g if w not in top]
    if not miss:
        continue
    if len(ex) < 12 and cap["miss"] % 20 < len(miss):
        ex.append((u, " ".join(words), " ".join(hyp.get(u, [])), miss, prob[u][:6]))
    common = [x for x in words if x not in set(gold)]
    utt = [q for x in words for q in ph.get(x, [])]
    sc = dict(prob[u])
    cap["miss"] += len(miss)
    cap["miss_trong_top50"] += sum(w in sc for w in miss)
    cap["miss_điểm<0.3"] += sum(sc.get(w, 0) < 0.3 for w in miss)
    cap["từ_hạng10_điểm≥0.3"] += sum(prob[u][K - 1][1] >= 0.3 for w in miss) if len(prob[u]) >= K else 0
    cap["miss_vì_nhiều_hơn_K_từ_đúng"] += len(g) > K and all(x in g for x in top)
    for w in top:
        if w not in ph:
            continue
        n_fp += 1
        p = ph[w]
        dg = min((nd(p, ph[x]) for x in g if x in ph), default=9)
        dc = best_span(p, common)
        if w in gold:
            fp_kind["từ đúng khác (list không có / chiếm chỗ)"] += 1
        elif dg <= 0.34:
            fp_kind["giống từ đúng (d ≤ 1/3)"] += 1
        elif dc <= 0.34:
            fp_kind["giống cụm 1-3 từ thường trong câu (d ≤ 1/3)"] += 1
        elif sub(p, utt) <= 0.34:
            fp_kind["khớp 1 đoạn âm bất kỳ của câu, lệch ranh giới từ (d ≤ 1/3)"] += 1
        else:
            fp_kind["khác"] += 1

print(f"## {split} {model} N2000 — Recall#{K}")
for k, (a, b) in sorted(stat.items()):
    print(f"{' / '.join(k):24s} {100 * a / b:6.2f}  (n={b})")
print(f"từ sai trong top-{K}: {n_fp}")
for k, v in fp_kind.most_common():
    print(f"  {k:45s} {100 * v / n_fp:5.1f}%")
print(dict(cap))
for u, r, h, m, t in ex:
    print(f"\n{u}\n  REF {r}\n  HYP {h}\n  rớt {m}\n  top {[(w, round(s, 2)) for w, s in t]}")
