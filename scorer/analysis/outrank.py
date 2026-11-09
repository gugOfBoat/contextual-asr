"""A4: từ đúng rớt khỏi top-K là bị từ nào vượt lên? (PLAN.md P1.1, bản chỉ dùng điểm, chưa có vị trí)

Với mỗi từ hiếm đúng có hạng > K trong list của câu, xét các từ nhiễu xếp trên nó và phân loại theo
khoảng cách âm vị (g2p) tới từ đúng: na ná (≤ 1), gần (2), khác (≥ 3).
Cận trên "tranh nhau": bỏ các từ nhiễu na ná/gần đứng trên từ đúng rồi tính lại R#K.

  python outrank.py --rank out/hou25/test-other/rank_N2000.json --ctx ../bench/protocol/test-other/ctx_oracle.json \
      --phones work/phones.tsv --k 10 --ex 15
"""
import argparse, json, random
from collections import Counter


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


ap = argparse.ArgumentParser()
ap.add_argument("--rank", required=True)
ap.add_argument("--ctx", required=True)
ap.add_argument("--phones", required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--ex", type=int, default=10, help="số ví dụ in ra")
a = ap.parse_args()

rank = json.load(open(a.rank, encoding="utf-8"))
ctx = json.load(open(a.ctx, encoding="utf-8"))
ph = {}
for l in open(a.phones, encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t")[:2]
    ph[w] = p.split()


def cls(w, t):  # loại từ nhiễu w so với từ đúng t
    if w not in ph or t not in ph:
        return "không có âm vị"
    d = lev(ph[w], ph[t])
    return "na ná (≤1)" if d <= 1 else "gần (2)" if d == 2 else "khác (≥3)"


n = hit = hit_ub = 0
alike_in_list = 0     # từ đúng mà list có ít nhất 1 từ nhiễu cách ≤ 2 âm vị (bất kể hạng)
nlen, alen = Counter(), Counter()  # số âm vị của từ bị rớt / của mọi từ đúng
ranks = []
top1 = Counter()      # loại của kẻ đứng đầu (từ nhiễu cao điểm nhất) trên mỗi từ đúng bị rớt
slots = Counter()     # loại của mọi từ nhiễu đứng trên từ đúng bị rớt
examples = []
for uid, lst in rank.items():
    true = set(ctx.get(uid, []))
    pos = {w: i for i, (w, _) in enumerate(lst)}
    for t in true:
        if t not in pos:
            continue
        n += 1
        alen[len(ph.get(t, []))] += 1
        alike_in_list += any(cls(w, t) in ("na ná (≤1)", "gần (2)") for w, _ in lst if w not in true)
        above = [w for w, _ in lst[:pos[t]] if w not in true]
        r = len([w for w, _ in lst[:pos[t]]])  # hạng (0-based), tính cả từ đúng khác như cách R#K
        if r < a.k:
            hit += 1
            hit_ub += 1
            continue
        c = [cls(w, t) for w in above]
        if c:  # rỗng khi mọi từ đứng trên đều là từ đúng khác (câu có > K từ hiếm)
            top1[c[0]] += 1
        slots.update(c)
        nlen[len(ph.get(t, []))] += 1
        ranks.append(r + 1)
        r_ub = r - sum(x in ("na ná (≤1)", "gần (2)") for x in c)
        hit_ub += r_ub < a.k
        examples.append((uid, t, r + 1, [(w, x) for w, x in zip(above[:5], c[:5])]))

miss = n - hit
print(f"{a.rank}\n  từ đúng: {n} | R#{a.k} = {100 * hit / n:.2f} | rớt khỏi top-{a.k}: {miss}")
print(f"  kẻ đứng đầu (từ nhiễu điểm cao nhất) trên từ bị rớt:")
for k, v in top1.most_common():
    print(f"    {k:15s} {v:5d}  {100 * v / miss:5.1f}%")
tot = sum(slots.values())
print(f"  mọi từ nhiễu đứng trên từ bị rớt ({tot}):")
for k, v in slots.most_common():
    print(f"    {k:15s} {v:6d}  {100 * v / tot:5.1f}%")
print(f"  cận trên nếu bỏ từ nhiễu na ná/gần đứng trên: R#{a.k} = {100 * hit_ub / n:.2f}")
print(f"  list có từ nhiễu cách ≤ 2 âm vị (bất kể hạng): {100 * alike_in_list / n:.1f}% số từ đúng")
ranks.sort()
print(f"  hạng của từ bị rớt: trung vị {ranks[len(ranks) // 2]}, p25 {ranks[len(ranks) // 4]}, p75 {ranks[3 * len(ranks) // 4]}")
for lo, hi in [(0, 3), (4, 5), (6, 8), (9, 99)]:
    m = sum(v for k, v in nlen.items() if lo <= k <= hi)
    t_ = sum(v for k, v in alen.items() if lo <= k <= hi)
    print(f"    từ {lo}-{hi} âm vị: tỉ lệ rớt {100 * m / max(t_, 1):5.1f}%  ({m}/{t_})")
random.seed(0)
for uid, t, r, ab in random.sample(examples, min(a.ex, len(examples))):
    print(f"  ví dụ {uid}: đúng '{t}' {' '.join(ph.get(t, []))} hạng {r}; trên nó: "
          + ", ".join(f"{w} [{' '.join(ph.get(w, []))}] ({x.split()[0]})" for w, x in ab))
