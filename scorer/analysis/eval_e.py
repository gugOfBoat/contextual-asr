"""Thước đo chung cho E1–E4 (scorer/PLAN-2011.md). Chạy CPU trên output của inference/rank.py.

  python eval_e.py --split test-other --list N2000 --base hou25_win \\
      hou25_win=<dir chứa prob_N2000.json [+ span_N2000.json]> pmaxsim_win=<dir> ...

Bảng ra (mọi số tính trên top-50 của list):
  1. Sạch@1/5/10 (= từ đúng có < K từ NHIỄU đứng trên) + Δ Sạch@1 so với --base, CI 95% bootstrap theo câu
  2. Báo nhầm: số từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu
  3. Sạch@1 trên nhóm từ Whisper (không prompt) nghe SAI / ĐÚNG
  4. Đúng@chỗ (cần span): tại span MFA của từ đúng, từ đúng đứng trên mọi từ nhiễu có span chồng lên, theo nhóm 1 / 2-4 / ≥5 từ hiếm
  5. Định vị (cần span): sai số tâm trung vị (s), % tâm nằm trong span MFA ± 0.1 s
  6. Sót sau giây 30: % từ đúng bắt đầu ≥ 30 s không lọt top-50
  7. Nhóm lỗi (kiểu Hoàng F12 nhưng cho scorer): từ nhiễu cao nhất đứng trên từ đúng là đồng âm / gần âm (≤2 âm vị) / khác hẳn;
     E4: ở CÙNG mức giữ 98% từ đúng (K98 nhiễu đầu tiên), mỗi câu lọt bao nhiêu từ nhiễu gần âm (≤2) với một từ đúng.
     Cần rank_<L>.json (đủ 2000 từ) trong thư mục hệ nếu K98 > 50.
"""
import argparse, collections, json, random
from pathlib import Path

R = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--list", default="N2000")
ap.add_argument("--base", help="tên hệ làm mốc cho CI")
ap.add_argument("runs", nargs="+", help="tên=thư mục")
a = ap.parse_args()

P = R / "bench/protocol" / a.split
ref = {l.split("\t")[0]: set(json.loads(l.rstrip("\n").split("\t")[2])) for l in open(P / "ref.tsv", encoding="utf-8")}
mfa, dur = collections.defaultdict(dict), collections.defaultdict(float)
for l in open(P / "align.tsv", encoding="utf-8"):
    u, _, w, s, e = l.rstrip("\n").split("\t")
    dur[u] = max(dur[u], float(e))
    if w in ref[u]:
        mfa[u].setdefault(w, (float(s), float(e)))
wh = {}
for l in open(R / "bench/out" / a.split / "whisper_none.tsv", encoding="utf-8"):
    x = l.rstrip("\n").split("\t")
    wh[x[0]] = set(x[1].split()) if len(x) > 1 else set()
DB = lambda s: "<5s" if s < 5 else "5-10s" if s < 10 else "10-20s" if s < 20 else "≥20s"
NG = lambda n: "1" if n == 1 else "2-4" if n <= 4 else "≥5"


def load(d):
    d = Path(d)
    prob = json.load(open(d / f"prob_{a.list}.json", encoding="utf-8"))
    sp = d / f"span_{a.list}.json"
    return prob, (json.load(open(sp, encoding="utf-8")) if sp.exists() else None)


def per_word_full(d, prob):
    rk = Path(d) / f"rank_{a.list}.json"
    return json.load(open(rk, encoding="utf-8")) if rk.exists() else prob


def per_word(prob):
    """→ {uid: [(từ đúng, số nhiễu đứng trên)]}; ngoài top-50 thì 50."""
    out = {}
    for u, lst in prob.items():
        above, pos = 0, {}
        for w, _ in lst:
            if w in ref[u]:
                pos[w] = above
            else:
                above += 1
        out[u] = [(w, pos.get(w, 50)) for w in ref[u]]
    return out


runs, full = {}, {}
for spec in a.runs:
    n, _, d = spec.partition("=")
    runs[n] = (*load(d), )
    full[n] = per_word_full(d, runs[n][0])
pw = {n: per_word(p) for n, (p, _) in runs.items()}
uids = sorted(set.intersection(*(set(x) for x in pw.values())))
rnd = random.Random(0)
boots = [[rnd.choice(uids) for _ in uids] for _ in range(1000)]


def sach(n, K, us, filt=lambda u, w: True):
    xs = [d < K for u in us for w, d in pw[n][u] if filt(u, w)]
    return 100 * sum(xs) / max(len(xs), 1), len(xs)


print(f"## {a.split}, list {a.list}, {len(uids)} câu\n\n### 1. Sạch@K\n")
print("| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với " + (a.base or "-") + " [CI 95%] |\n|---|---|---|---|---|")
for n in runs:
    ci = ""
    if a.base and n != a.base:
        ds = sorted(sach(n, 1, b)[0] - sach(a.base, 1, b)[0] for b in boots)
        ci = f"{sach(n, 1, uids)[0] - sach(a.base, 1, uids)[0]:+.1f} [{ds[25]:+.1f}; {ds[974]:+.1f}]"
    print(f"| {n} | " + " | ".join(f"{sach(n, K, uids)[0]:.1f}" for K in (1, 5, 10)) + f" | {ci} |")

print("\n### 2. Báo nhầm: từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu\n")
bk = ["<5s", "5-10s", "10-20s", "≥20s"]
print("| hệ | " + " | ".join(f"{b} ({sum(1 for u in uids if DB(dur[u]) == b)} câu)" for b in bk) + " |\n|---" * 1 + "|---" * 4 + "|")
for n, (prob, _) in runs.items():
    g = collections.defaultdict(list)
    for u in uids:
        g[DB(dur[u])].append(sum(1 for w, p in prob[u] if p >= 0.5 and w not in ref[u]))
    print(f"| {n} | " + " | ".join(f"{sum(g[b]) / max(len(g[b]), 1):.1f}" for b in bk) + " |")

print("\n### 3. Sạch@1 theo việc Whisper không prompt nghe đúng / sai từ đó\n\n| hệ | nghe sai | nghe đúng |\n|---|---|---|")
for n in runs:
    s1, k1 = sach(n, 1, uids, lambda u, w: w not in wh.get(u, ()))
    s2, k2 = sach(n, 1, uids, lambda u, w: w in wh.get(u, ()))
    print(f"| {n} | {s1:.1f} ({k1}) | {s2:.1f} ({k2}) |")

print("\n### 4–5. Đúng@chỗ và định vị (hệ có span)\n")
print("| hệ | Sạch@1 cả câu: 1 / 2-4 / ≥5 từ | **Đúng@chỗ**: 1 / 2-4 / ≥5 từ | nhiễu cùng chỗ TB | sai số tâm trung vị (s) | % tâm trong span ±0.1 s |")
print("|---|---|---|---|---|---|")
for n, (prob, sp) in runs.items():
    if sp is None:
        continue
    hit, tot, crowd = collections.Counter(), collections.Counter(), []
    err, inside = [], []
    for u in uids:
        rank = {w: i for i, (w, _) in enumerate(prob[u])}
        g = NG(len(ref[u]))
        for w, (b, e) in mfa[u].items():
            if w not in rank:
                continue
            here = [x for x in rank if x not in ref[u] and x in sp[u] and sp[u][x][0] < e and sp[u][x][1] > b]
            crowd.append(len(here))
            tot[g] += 1
            hit[g] += all(rank[x] > rank[w] for x in here)
            if w in sp[u]:
                c = (sp[u][w][0] + sp[u][w][1]) / 2
                err.append(abs(c - (b + e) / 2))
                inside.append(b - 0.1 <= c <= e + 0.1)
    G = ["1", "2-4", "≥5"]
    whole = " / ".join(f"{sach(n, 1, [u for u in uids if ref[u] and NG(len(ref[u])) == g])[0]:.1f}" for g in G)
    err.sort()
    print(f"| {n} | {whole} | " + " / ".join(f"{100 * hit[g] / max(tot[g], 1):.1f}" for g in G)
          + f" | {sum(crowd) / max(len(crowd), 1):.1f} | {err[len(err) // 2] if err else float('nan'):.2f} | "
          f"{100 * sum(inside) / max(len(inside), 1):.1f} |")

print("\n### 6. Từ đúng bắt đầu ≥ 30 s: % không lọt top-50\n")
for n, (prob, _) in runs.items():
    late = [(u, w) for u in uids for w, (b, _) in mfa[u].items() if b >= 30]
    miss = sum(1 for u, w in late if w not in {x for x, _ in prob[u]})
    print(f"- {n}: {miss}/{len(late)}")

print("\n### 7. Nhóm lỗi + từ gần âm ở cùng mức recall 98%\n")
PH = {}
for l in open(R / "scorer/runs/2026-10-16/phones.tsv", encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t")
    PH[w] = p.split()


def pd(x, y):
    if x not in PH or y not in PH:
        return 99
    s_, t = PH[x], PH[y]
    d = list(range(len(t) + 1))
    for i in range(1, len(s_) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(t) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (s_[i - 1] != t[j - 1]))
    return d[-1]


print("| hệ | lượt bị vượt | đồng âm % | gần âm % | khác hẳn % | K98 (số nhiễu cần lấy) | nhiễu gần âm lọt / câu ở K98 |")
print("|---|---|---|---|---|---|---|")
for n in runs:
    fw = per_word(full[n])
    ds = sorted(d for u in uids for _, d in fw[u])
    k98 = ds[min(len(ds) - 1, int(0.98 * len(ds)))] + 1
    grp = collections.Counter()
    near = 0
    for u in uids:
        lst = [w for w, _ in full[n][u]]
        noise = [w for w in lst if w not in ref[u]]
        for w, d in fw[u]:
            if 1 <= d < 50:
                x = pd(w, noise[0]) if noise else 99
                grp["đồng âm" if x == 0 else "gần âm" if x <= 2 else "khác"] += 1
        near += sum(1 for x in noise[:k98] if any(pd(x, g) <= 2 for g in ref[u]))
    t = max(sum(grp.values()), 1)
    print(f"| {n} | {t} | {100 * grp['đồng âm'] / t:.1f} | {100 * grp['gần âm'] / t:.1f} | {100 * grp['khác'] / t:.1f} | "
          f"{k98}{' (≥50: cần rank file)' if k98 > 50 and full[n] is runs[n][0] else ''} | {near / len(uids):.2f} |")
