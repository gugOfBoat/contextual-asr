"""Bước 2b — nhãn "từ khó": chạy Qwen3-ASR KHÔNG context, xem từ hiếm nào của câu bị chép sai.

  python -m data.qwen_pass1 --libri /work/libri/LibriSpeech --splits dev-clean dev-other train-clean-100

Mỗi câu: từ hiếm = từ ngoài common_words_5k (như IS21).
  khó (hard) = từ hiếm của reference KHÔNG xuất hiện trong hyp lần 1   [C — xấp xỉ, không căn chỉnh từng vị trí]
              so sau chuẩn hoá Whisper (recognising = recognizing, mr = mister), giống khi chấm B-WER wnorm
  dễ  (easy) = có xuất hiện trong hyp
Đầu ra: work/pass1_<split>.jsonl  {"uid", "hyp", "hard": [...], "easy": [...]}
In ra tỉ lệ từ khó mỗi split → biết Qwen có "thuộc" tập train không.
"""
import argparse, json, sys
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from common import BENCH, IS21, WORK

sys.path.insert(0, str(BENCH))
from textnorm import norm  # noqa: E402  chuẩn hoá giống bench/run_qwen3asr.py

ap = argparse.ArgumentParser()
ap.add_argument("--libri", required=True)
ap.add_argument("--splits", nargs="+", default=["dev-clean"])
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

wn = EnglishTextNormalizer(json.load(open(BENCH / "data" / "whisper_normalizer.json", encoding="utf-8")))
common = set(open(IS21 / "common_words_5k.txt", encoding="utf-8").read().split())
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                      max_inference_batch_size=a.bs)

for split in a.splits:
    rows = []
    for trans in sorted((Path(a.libri) / split).rglob("*.trans.txt")):
        for line in open(trans, encoding="utf-8"):
            uid, text = line.strip().split(" ", 1)
            rows.append((uid, str(trans.parent / f"{uid}.flac"), text.lower().split()))
    rows = rows[:a.limit]
    res = model.transcribe(audio=[p for _, p, _ in rows], context=[""] * len(rows), language="English")
    n_hard = n_rare = 0
    with open(WORK / f"pass1_{split}.jsonl", "w", encoding="utf-8") as f:
        for (uid, _, ref), r in zip(rows, res):
            hyp = norm(r.text)
            hw = set(wn(hyp).split())
            rare = sorted({w for w in ref if w not in common})
            hard = [w for w in rare if not set(wn(w).split()) <= hw]
            n_hard += len(hard)
            n_rare += len(rare)
            f.write(json.dumps({"uid": uid, "hyp": hyp, "hard": hard, "easy": [w for w in rare if w not in hard]}) + "\n")
    print(f"{split}: {len(rows)} câu, {n_rare} lượt từ hiếm, khó {n_hard} ({100 * n_hard / n_rare:.2f}%)", flush=True)
