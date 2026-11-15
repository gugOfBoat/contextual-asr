"""B2 (b): sửa SAU khi decode theo span, kiểu PTC-Bias (arXiv 2609.28727 §2.5) nhưng dùng span thay cho đối chiếu cả câu.

Đầu vào: một lần decode KHÔNG context có ghi giây cho mỗi token (gate_boost.py --ctx none --times),
list bias của câu và span của từng từ trong list (align.tsv = oracle MFA, hoặc span scorer dự đoán).
Với mỗi từ w trong list: xét các từ hyp (1 từ hoặc 2 từ liền nhau, để sửa lỗi tách từ "mc lain")
có giây bắt đầu nằm trong span ± margin; nếu giống w về âm vị >= --theta thì thay bằng w.
Mỗi vị trí hyp chỉ bị thay một lần; tranh nhau thì cặp giống nhất thắng.

  python postcorr.py --split test-other --hyp whisper_none_times --ctx top50or_N2000 --spans protocol/test-other/pred_a1s1.tsv --theta 0.7
Ra: out/<split>/<hyp>_pc_<ctx><tag>.tsv, chấm bằng score.py / e1_table.py.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from textnorm import norm

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--hyp", required=True, help="tên output của gate_boost.py có .raw.tsv và .time.tsv")
ap.add_argument("--ctx", required=True)
ap.add_argument("--spans", help="tsv: uid, i, từ, bắt đầu, kết thúc (mặc định align.tsv)")
ap.add_argument("--margin", type=float, default=0.1)
ap.add_argument("--theta", type=float, default=0.7, help="ngưỡng giống âm vị: 1 - edit/max(len)")
ap.add_argument("--phones", default=str(ROOT.parent / "scorer/runs/2026-10-16/phones.tsv"))
ap.add_argument("--tag", default="")
a = ap.parse_args()

proto, out = ROOT / "protocol" / a.split, ROOT / "out" / a.split
ctx = json.load(open(proto / f"ctx_{a.ctx}.json", encoding="utf-8"))
spans = defaultdict(list)
for l in open(a.spans or proto / "align.tsv", encoding="utf-8"):
    uid, _, w, b, e = l.rstrip("\n").split("\t")[:5]
    spans[uid, w].append((float(b) - a.margin, float(e) + a.margin))
PH = {}
for l in open(a.phones, encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t")
    PH[w] = p.split()
g2p = None


def ph(w):
    global g2p
    if w not in PH:
        if g2p is None:
            from g2p_en import G2p
            g2p = G2p()
        PH[w] = [x.rstrip("012") for x in g2p(w) if x.strip() and x[0].isalpha()]
    return PH[w]


def sim(x, y):
    return 1 - Levenshtein.distance(x, y) / max(len(x), len(y), 1)


# token -> từ: token bắt đầu bằng khoảng trắng mở một từ mới; giây của từ = giây của token đầu
toks = defaultdict(list)
for l in open(out / f"{a.hyp}.time.tsv", encoding="utf-8"):
    uid, _, t, s = l.rstrip("\n").split("\t", 3)
    toks[uid].append((float(t), s))

n_fix = 0
fn = out / f"{a.hyp}_pc_{a.ctx}{a.tag}.tsv"
with open(fn, "w", encoding="utf-8") as f:
    for l in open(out / f"{a.hyp}.raw.tsv", encoding="utf-8"):
        uid = l.split("\t")[0]
        words = []  # [(giây, chữ đã chuẩn hoá)]
        for t, s in toks[uid]:
            if s.startswith(" ") or not words:
                words.append([t, s])
            else:
                words[-1][1] += s
        words = [(t, w) for t, s in words for w in norm(s).split()]
        cand = []  # (độ giống, i, số từ hyp bị thay, từ bias)
        for w in ctx.get(uid, []):
            for i, (t, h) in enumerate(words):
                if not any(b <= t <= e for b, e in spans[uid, w]):
                    continue
                for k in (1, 2):
                    seg = [x for _, x in words[i:i + k]]
                    if len(seg) < k:
                        continue
                    if k == 1 and seg[0] == w:
                        continue
                    sc = sim(sum((ph(x) for x in seg), []), ph(w))
                    if sc >= a.theta:
                        cand.append((sc, i, k, w))
        used, rep = set(), {}
        for sc, i, k, w in sorted(cand, reverse=True):
            if used & set(range(i, i + k)) or w in {x for _, x in rep.values()}:
                continue
            if any(x == w for _, x in words[max(0, i - 3):i + k + 3]):  # từ đã có ngay cạnh: model viết đúng rồi
                continue
            used |= set(range(i, i + k))
            rep[i] = (k, w)
        hyp, i = [], 0
        while i < len(words):
            if i in rep:
                k, w = rep[i]
                hyp.append(w)
                i += k
                n_fix += 1
            else:
                hyp.append(words[i][1])
                i += 1
        f.write(f"{uid}\t{' '.join(hyp)}\n")
print(f"thay {n_fix} chỗ -> {fn}")
