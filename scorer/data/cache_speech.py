"""Lưu feature encoder Whisper của câu nói train ra đĩa → train scorer không phải chạy lại encoder mỗi epoch.

  python -m data.cache_speech --train train            # work/train.jsonl → work/speech_train.f16 + .idx
Encoder đóng băng nên feature không đổi giữa các epoch; train-clean-100 ≈ 46 GB float16.
"""
import argparse, json

import numpy as np
import soundfile as sf

from common import WORK, Encoder

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="train")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

items = [json.loads(l) for l in open(WORK / f"{a.train}.jsonl", encoding="utf-8")][:a.limit]
enc = Encoder()
off = 0
with open(WORK / f"speech_{a.train}.f16", "wb") as f, open(WORK / f"speech_{a.train}.idx", "w", encoding="utf-8") as fi:
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]
        for it, h in zip(b, enc([sf.read(it["path"], dtype="float32")[0] for it in b])):
            x = h.cpu().numpy().astype(np.float16)
            f.write(x.tobytes())
            fi.write(f"{it['uid']}\t{off}\t{len(x)}\n")
            off += len(x)
        if (i // a.bs) % 50 == 0:
            print(f"{i + len(b)}/{len(items)}  {off * 1280 * 2 / 1e9:.1f} GB", flush=True)
print(f"xong {len(items)} câu, {off} frame")
