"""Bước 3 — audio TTS → encoder Whisper-large-v3 → feature, cắt theo độ dài thật, ghép vào 1 file.

  python -m data.encode_bias --bs 64

Đầu ra: work/bias.f16 (tổng frame × 1280, float16) + work/bias.idx (từ \\t offset \\t số frame).
Cắt khoảng lặng đầu/cuối (top_db=40) trước khi encode [B]: TTS có lặng, encoder Whisper pad tới 30s.
"""
import argparse

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import SR, WORK, Encoder, tts_path

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()

words = [w for w in open(WORK / "vocab.txt", encoding="utf-8").read().split() if tts_path(w).exists()]


def load(w):
    wav, _ = librosa.load(tts_path(w), sr=SR)
    return w, librosa.effects.trim(wav, top_db=40)[0]


if __name__ == "__main__":
    enc = Encoder()
    off = 0
    dl = DataLoader(words, batch_size=a.bs, num_workers=16, collate_fn=lambda ws: [load(w) for w in ws])
    with open(WORK / "bias.f16", "wb") as fd, open(WORK / "bias.idx", "w", encoding="utf-8") as fi:
        for i, batch in enumerate(dl):
            for (w, _), h in zip(batch, enc([x for _, x in batch])):
                fd.write(h.cpu().numpy().astype(np.float16).tobytes())
                fi.write(f"{w}\t{off}\t{len(h)}\n")
                off += len(h)
            if i % 200 == 0:
                print(f"{i * a.bs}/{len(words)}  frame trung bình {off / ((i + 1) * a.bs):.1f}", flush=True)
    print(f"xong {len(words)} từ, {off} frame, {off * 1280 * 2 / 1e9:.1f} GB")
