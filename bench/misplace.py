"""Có list mà không có vị trí thì ASR có dùng từ ĐÚNG CHỖ không? Chia câu theo số từ hiếm khác nhau (1 / 2-4 / ≥5).

Mỗi lượt từ hiếm của ref (căn chỉnh IS21 như score.py) xếp vào:
  đúng
  sai chỗ: từ đúng KHÁC   ASR viết một từ hiếm khác của chính câu này vào chỗ này (đúng từ, sai chỗ)
  sai chỗ: từ nhiễu       ASR viết một từ nhiễu của prompt / list boost vào chỗ này
  lỗi khác                xoá, nghe giống, ranh giới…
Thêm: "chèn từ hiếm của câu" = ASR viết thêm một từ hiếm của câu ở chỗ không có (lặp / đặt thừa).

  python misplace.py --split test-other --sys none=out/test-other/whisper_none.tsv: \
      prompt_oracle=out/test-other/whisper_oracle_prompt.tsv:protocol/test-other/ctx_oracle.json ...
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from score_is21_official import Code, EditDistance
from textnorm import norm

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--sys", nargs="+", required=True, help="tên=hyp.tsv:list.json (list = từ đưa cho ASR; trống = không có)")
a = ap.parse_args()

ref = {}
for l in open(Path(__file__).parent / f"protocol/{a.split}/ref.tsv", encoding="utf-8"):
    u, t, r = l.rstrip("\n").split("\t")
    ref[u] = (t.split(), set(json.loads(r)))
grp = lambda n: "1 từ" if n == 1 else "2-4 từ" if n <= 4 else "≥5 từ"
G = ["1 từ", "2-4 từ", "≥5 từ"]
C = ["đúng", "sai chỗ: từ đúng khác", "sai chỗ: từ nhiễu", "lỗi khác", "chèn từ hiếm của câu"]

print(f"## {a.split}: từ hiếm có được dùng đúng chỗ? (% trên số lượt từ hiếm của nhóm)\n")
for g in G:
    print(f"\n### Câu có {g} hiếm ({sum(1 for _, r in ref.values() if r and grp(len(r)) == g)} câu)\n")
    print("| hệ | lượt | " + " | ".join(C) + " |")
    print("|---" * (len(C) + 2) + "|")
    for spec in a.sys:
        name, _, rest = spec.partition("=")
        hp, _, lp = rest.partition(":")
        lst = json.load(open(lp, encoding="utf-8")) if lp else {}
        hyp = {l.split("\t")[0]: norm(l.rstrip("\n").partition("\t")[2]).split() for l in open(hp, encoding="utf-8")}
        cnt, n = defaultdict(int), 0
        for u, (rw, rare) in ref.items():
            if not rare or grp(len(rare)) != g or u not in hyp:
                continue
            pr = {x[0] if isinstance(x, list) else x for x in lst.get(u, [])}
            h = hyp[u]
            res = EditDistance().align(rw, h)
            for code, ri, hi in zip(res.codes, res.refs, res.hyps):
                if code == Code.insertion:
                    cnt["chèn từ hiếm của câu"] += h[hi] in rare
                    continue
                if rw[ri] not in rare:
                    if code == Code.substitution and h[hi] in rare:   # từ hiếm của câu đè lên một từ thường
                        cnt["chèn từ hiếm của câu"] += 1
                    continue
                n += 1
                if code == Code.match:
                    cnt["đúng"] += 1
                elif code == Code.substitution and h[hi] in rare:
                    cnt["sai chỗ: từ đúng khác"] += 1
                elif code == Code.substitution and h[hi] in pr:
                    cnt["sai chỗ: từ nhiễu"] += 1
                else:
                    cnt["lỗi khác"] += 1
        print(f"| {name} | {n} | " + " | ".join(f"{100 * cnt[c] / max(n, 1):.2f}" for c in C) + " |")
