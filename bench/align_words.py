"""Mốc thời gian thật của từng từ trong reference bằng forced alignment CTC (wav2vec2-base-960h, 20 ms/frame).
Cùng cách với `scripts/align_words.py` của Hoàng (F14), tự viết Viterbi để không cần torchaudio.

  python align_words.py --split test-other --subset protocol/test-other/subset100.txt

Ra: protocol/<split>/align.tsv: uid, chỉ số từ, từ, bắt đầu (s), kết thúc (s), posterior trung bình.
Scorer không bao giờ thấy file này; chỉ dùng làm "vị trí hoàn hảo" (cận trên) và mốc chấm định vị.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).parent
FR = 0.02  # wav2vec2: 320 mẫu / frame ở 16 kHz

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--subset")
ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
a = ap.parse_args()

proto = ROOT / "protocol" / a.split
audio = dict(l.rstrip("\n").split("\t") for l in open(proto / "audio.tsv", encoding="utf-8"))
ref = {l.split("\t")[0]: l.split("\t")[1].split() for l in open(proto / "ref.tsv", encoding="utf-8")}
uids = open(a.subset).read().split() if a.subset else list(ref)

proc = Wav2Vec2Processor.from_pretrained(a.model)
model = Wav2Vec2ForCTC.from_pretrained(a.model).eval()
vocab = proc.tokenizer.get_vocab()
blank, sep = vocab["<pad>"], vocab["|"]


def viterbi(lp, y):
    """lp: (T, V) log-prob; y: nhãn ký tự. Trả về trạng thái (chỉ số trong chuỗi có blank xen kẽ) cho từng frame."""
    ext = [blank] + [v for c in y for v in (c, blank)]
    S, T = len(ext), lp.shape[0]
    skip = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    em = lp[:, ext]
    D = np.full((T, S), -np.inf)
    B = np.zeros((T, S), dtype=np.int8)  # 0: đứng yên, 1: từ s-1, 2: từ s-2
    D[0, :2] = em[0, :2]
    for t in range(1, T):
        c0 = D[t - 1]
        c1 = np.concatenate([[-np.inf], c0[:-1]])
        c2 = np.where(skip, np.concatenate([[-np.inf, -np.inf], c0[:-2]]), -np.inf)
        st = np.stack([c0, c1, c2])
        B[t] = st.argmax(0)
        D[t] = st.max(0) + em[t]
    s = S - 1 if D[-1, S - 1] >= D[-1, S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s -= int(B[t, s])
        path.append(s)
    return path[::-1]


with open(proto / "align.tsv", "w", encoding="utf-8") as f:
    for uid in uids:
        wav, sr = sf.read(audio[uid], dtype="float32")
        with torch.no_grad():
            x = proc(wav, sampling_rate=sr, return_tensors="pt").input_values
            lp = model(x).logits[0].log_softmax(-1).numpy()
        words = ref[uid]
        y, owner = [], []  # nhãn ký tự và từ chứa nó (-1 = dấu cách giữa từ)
        for i, w in enumerate(words):
            if i:
                y.append(sep), owner.append(-1)
            for ch in w.upper():
                if ch in vocab:
                    y.append(vocab[ch]), owner.append(i)
        path = viterbi(lp, y)
        span = {}
        for t, s in enumerate(path):
            if s % 2 and owner[s // 2] >= 0:
                i = owner[s // 2]
                b, e, ps = span.get(i, (t, t, []))
                span[i] = (min(b, t), max(e, t), ps + [lp[t, y[s // 2]]])
        for i, w in enumerate(words):
            if i in span:
                b, e, ps = span[i]
                f.write(f"{uid}\t{i}\t{w}\t{b * FR:.2f}\t{(e + 1) * FR:.2f}\t{np.exp(np.mean(ps)):.3f}\n")
print(f"xong {len(uids)} câu -> {proto / 'align.tsv'}")
