"""Phân rã lỗi từ hiếm theo NGUỒN GỐC + đếm từ nhiễu bị chép ra, cho nhiều hệ cùng lúc (S1, PLAN-chot-scorer.md) [C].

Căn chỉnh bằng scorer IS21 chính thức (như score.py) → tổng lỗi khớp B-WER của score.py.
Mỗi lượt từ hiếm SAI của ref xếp vào ô đầu tiên khớp:
  xoá            ASR bỏ mất từ
  ngoài prompt   từ đúng không có trong prompt (bước chọn từ bỏ sót)
  thay = nhiễu   ASR viết thành một từ NHIỄU có trong prompt (prompt kéo sai)
  ranh giới      viết dính/tách từ ("grave digger" ↔ "gravedigger")
  nghe giống     từ viết ra cách ≤ 2 âm vị (g2p) — ASR nghe ra nhưng viết chính tả khác
  khác
  chèn từ hiếm   ASR viết THÊM một từ hiếm của ref (thường là lặp chữ); score.py tính vào B-WER
Thêm: "nhiễu xuất hiện" = số token của hyp là từ nhiễu trong prompt (không có trong ref), kể cả chèn thêm;
"nhiễu nghe giống / prompt" = số từ nhiễu trong prompt cách một từ đúng ≤ 2 âm vị (kiểu R_H của BR-ASR).

  python error_breakdown.py --split test-other \
      --sys pxattn_N2000=out/test-other/whisper_pxattn_N2000.tsv:../scorer/runs/2026-10-16/out/pxattn/test-other/prob_N2000.json \
      --sys N100=out/test-other/whisper_N100.tsv:protocol/test-other/ctx_N100.json
Prompt: file prob_*.json (top-50 [[từ, p], ...]) hoặc ctx_*.json ({uid: [từ, ...]}); bỏ trống sau ":" = không prompt.
"""
import argparse, json
from pathlib import Path

from score_is21_official import Code, EditDistance
from textnorm import norm

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--sys", nargs="+", required=True, help="tên=hyp.tsv:prompt.json")
ap.add_argument("--phones", default=str(Path(__file__).parent.parent / "scorer/runs/2026-10-16/phones.tsv"))
ap.add_argument("--md", help="ghi bảng markdown ra file (nối thêm)")
a = ap.parse_args()

PH = {}
for l in open(a.phones, encoding="utf-8"):
    w, p = l.rstrip("\n").split("\t")
    PH[w] = [x.rstrip("012") for x in p.split()]


def pdist(x, y):
    """Khoảng cách sửa âm vị; None nếu thiếu âm vị."""
    if x not in PH or y not in PH:
        return None
    s, t = PH[x], PH[y]
    d = list(range(len(t) + 1))
    for i in range(1, len(s) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(t) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (s[i - 1] != t[j - 1]))
    return d[-1]


def load_prompt(p):
    if not p:
        return {}
    d = json.load(open(p, encoding="utf-8"))
    return {u: [x[0] if isinstance(x, list) else x for x in v] for u, v in d.items()}


ref = {}
for l in open(Path(__file__).parent / f"protocol/{a.split}/ref.tsv", encoding="utf-8"):
    u, t, r = l.rstrip("\n").split("\t")
    ref[u] = (t.split(), set(json.loads(r)))

CATS = ["xoá", "ngoài prompt", "thay = nhiễu", "ranh giới", "nghe giống", "khác", "chèn từ hiếm"]
rows = []
for spec in a.sys:
    name, _, rest = spec.partition("=")
    hp, _, pp = rest.partition(":")
    prompt = load_prompt(pp)
    hyp = {}
    for l in open(hp, encoding="utf-8"):
        u, _, t = l.rstrip("\n").partition("\t")
        hyp[u] = norm(t).split()
    n_rare = n_err = n_dist_out = n_near_prompt = n_true_in = n_prompt = 0
    cat = dict.fromkeys(CATS, 0)
    for u, (rw, rare) in ref.items():
        if u not in hyp:
            continue
        pr = set(prompt.get(u, []))
        dist = pr - set(rw)
        n_prompt += len(pr)
        n_near_prompt += sum(1 for d in dist if any((pdist(d, t) or 99) <= 2 for t in rare))
        h = hyp[u]
        n_dist_out += sum(1 for w in h if w in dist)
        res = EditDistance().align(rw, h)
        codes = list(zip(res.codes, res.refs, res.hyps))
        for k, (code, ri, hi) in enumerate(codes):
            if code == Code.insertion:
                if h[hi] in rare:                                      # score.py tính lượt chèn này vào B-WER
                    n_err += 1
                    cat["chèn từ hiếm"] += 1
                continue
            if rw[ri] not in rare:
                continue
            n_rare += 1
            n_true_in += rw[ri] in pr
            if code == Code.match:
                continue
            n_err += 1
            w = rw[ri]
            hw = h[hi] if code == Code.substitution else None
            nb = [h[codes[j][2]] for j in (k - 1, k + 1) if 0 <= j < len(codes) and codes[j][0] != Code.deletion]
            if code == Code.deletion and not any(w in x and len(x) > len(w) for x in nb):
                c = "xoá"
            elif w not in pr and prompt:
                c = "ngoài prompt"
            elif hw is not None and hw in dist:
                c = "thay = nhiễu"
            elif code == Code.deletion or (hw and (w in hw or hw in w) and abs(len(hw) - len(w)) >= 3):
                c = "ranh giới"
            elif hw and (pdist(w, hw) is not None and pdist(w, hw) <= 2):
                c = "nghe giống"
            else:
                c = "khác"
            cat[c] += 1
    n_u = len([u for u in ref if u in hyp])
    rows.append((name, n_rare, n_err, cat, n_dist_out, n_near_prompt / n_u, n_true_in / max(n_rare, 1), n_prompt / n_u))

out = [f"\n### {a.split}: lỗi từ hiếm theo nguồn gốc (plain, căn chỉnh IS21; % = phần của tổng số lượt từ hiếm)\n",
       "| hệ | B-WER (lỗi / lượt) | " + " | ".join(CATS) + " | từ nhiễu xuất hiện trong hyp | nhiễu nghe giống / prompt | từ đúng có trong prompt | cỡ prompt |",
       "|---" * (len(CATS) + 6) + "|"]
for name, n, e, cat, dout, near, tin, psz in rows:
    out.append(f"| {name} | {100 * e / n:.2f} ({e}/{n}) | " + " | ".join(f"{100 * cat[c] / n:.2f}" for c in CATS)
               + f" | {dout} | {near:.2f} | {100 * tin:.1f}% | {psz:.1f} |")
print("\n".join(out))
if a.md:
    open(a.md, "a", encoding="utf-8").write("\n".join(out) + "\n")
