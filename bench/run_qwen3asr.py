"""Baseline Qwen3-ASR-1.7B trên protocol IS21 (chạy trên máy có GPU).

  pip install -U qwen-asr
  python run_qwen3asr.py --split test-clean --ctx none oracle N100 N500 N1000 N2000 missing_N100
  (CPU, tập con) python run_qwen3asr.py --ctx none --subset protocol/test-clean/subset100.txt --tag _sub100 --bs 4
  python score_is21_official.py --refs protocol/test-clean/ref.tsv --hyps out/test-clean/qwen3asr_N100.tsv

Context đưa vào system prompt dạng "w1, w2, ..." (giống prompt "naive" của Hou25).
Hyp được chuẩn hoá về format LibriSpeech: chữ thường, bỏ dấu câu, giữ dấu nháy đơn.
"""
import argparse, json
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from textnorm import norm

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-clean")
ap.add_argument("--ctx", nargs="+", default=["none"])
ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--subset", help="file danh sách uid (vd protocol/test-clean/subset100.txt)")
ap.add_argument("--tag", default="", help="hậu tố tên file output, vd _sub100")
a = ap.parse_args()

proto = ROOT / "protocol" / a.split
audio = [l.rstrip("\n").split("\t") for l in open(proto / "audio.tsv", encoding="utf-8")]
if a.subset:
    keep = set(open(a.subset).read().split())
    audio = [x for x in audio if x[0] in keep]
out = ROOT / "out" / a.split
out.mkdir(parents=True, exist_ok=True)

gpu = torch.cuda.is_available()
model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16 if gpu else torch.float32,
                                      device_map="cuda:0" if gpu else "cpu",
                                      max_inference_batch_size=a.bs)
for c in a.ctx:
    ctx = json.load(open(proto / f"ctx_{c}.json", encoding="utf-8"))
    uids, paths = zip(*audio)
    res = model.transcribe(audio=list(paths), context=[", ".join(ctx[u]) for u in uids], language="English")
    with open(out / f"qwen3asr_{c}{a.tag}.tsv", "w", encoding="utf-8") as f, \
         open(out / f"qwen3asr_{c}{a.tag}.raw.tsv", "w", encoding="utf-8") as fr:
        for u, r in zip(uids, res):
            f.write(f"{u}\t{norm(r.text)}\n")
            fr.write(f"{u}\t{r.text}\n")
    print("xong", c)
