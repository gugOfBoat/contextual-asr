"""Đổi căn chỉnh MFA của LibriSpeech (Lugosch et al. 2019, zenodo 2619474; TextGrid, tầng "words") sang
protocol/<split>/align.tsv cùng định dạng với align_words.py: uid, chỉ số từ, từ, bắt đầu (s), kết thúc (s).
MFA ghi từ nó không biết là <unk> → nếu số từ khớp reference thì lấy tên từ theo reference.

  python mfa_to_align.py --split test-other --mfa /work/align
"""
import argparse, re
from pathlib import Path

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--mfa", required=True)
a = ap.parse_args()

proto = ROOT / "protocol" / a.split
ref = {l.split("\t")[0]: l.split("\t")[1].split() for l in open(proto / "ref.tsv", encoding="utf-8")}
iv = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"')
n = miss = diff = 0
with open(proto / "align.tsv", "w", encoding="utf-8") as f:
    for uid, rw in ref.items():
        spk, chap, _ = uid.split("-")
        p = Path(a.mfa) / a.split / spk / chap / f"{uid}.TextGrid"
        if not p.exists():
            miss += 1
            continue
        words = p.read_text(encoding="utf-8").split('name = "phones"')[0]  # chỉ tầng words
        ws = [(b, e, w.lower()) for b, e, w in iv.findall(words) if w]
        if len(ws) == len(rw):
            ws = [(b, e, r) for (b, e, _), r in zip(ws, rw)]
        else:
            diff += 1
        for i, (b, e, w) in enumerate(ws):
            f.write(f"{uid}\t{i}\t{w}\t{float(b):.2f}\t{float(e):.2f}\n")
        n += 1
print(f"{a.split}: {n} câu, thiếu {miss}, lệch số từ với ref {diff} -> {proto / 'align.tsv'}")
