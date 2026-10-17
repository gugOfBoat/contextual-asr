"""Dùng chung cho data/, training/, inference/: đường dẫn, encoder Whisper, kho feature bias word."""
import hashlib, os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("SCORER_WORK", ROOT / "work"))  # mọi output nặng (tts, feature, ckpt, out) — không commit
BENCH = ROOT.parent / "bench"
IS21 = BENCH / "data" / "is21"
SR = 16000
FPS = 50                                               # encoder Whisper: 50 frame/giây, tối đa 1500 frame (30s)


def tts_path(w):
    h = hashlib.md5(w.encode()).hexdigest()
    return WORK / "tts" / h[:2] / f"{h}.mp3"


def libri_path(libri, uid):
    spk, chap, _ = uid.split("-")  # uid LibriSpeech không chứa tên split → thử lần lượt
    for s in ["test-clean", "test-other", "dev-clean", "train-clean-100"]:
        p = Path(libri) / s / spk / chap / f"{uid}.flac"
        if p.exists():
            return p
    raise FileNotFoundError(uid)


class Encoder:
    """Encoder Whisper-large-v3 đóng băng (Hou25 §3.3). Trả feature cắt theo độ dài thật của audio."""

    def __init__(self, name="openai/whisper-large-v3", dev="cuda"):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.dev = dev
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.enc = WhisperModel.from_pretrained(name, torch_dtype=torch.float16).get_encoder().to(dev).eval()

    @torch.no_grad()
    def __call__(self, wavs):
        x = self.fe(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, torch.float16)
        h = self.enc(x).last_hidden_state
        return [h[i, :max(1, min(1500, -(-len(w) * FPS // SR)))] for i, w in enumerate(wavs)]


PHONES = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH").split()
N_PHONE = len(PHONES) + 1                              # id 0 = pad; ARPAbet bỏ trọng âm


class PhoneStore:
    """Âm vị từng từ: work/phones.tsv (từ \\t "K AA R D AH M AH M") từ data/g2p.py."""

    def __init__(self, work=WORK):
        pid = {p: i + 1 for i, p in enumerate(PHONES)}
        self.idx = {}
        for line in open(work / "phones.tsv", encoding="utf-8"):
            w, ph = line.rstrip("\n").split("\t")
            ids = [pid[p] for p in ph.split() if p in pid]
            if ids:
                self.idx[w] = ids

    def __contains__(self, w):
        return w in self.idx

    def seq(self, words):
        """Bản chép (list từ) → 1 chuỗi id âm vị, ngăn từ bằng token N_PHONE. Từ không có âm vị thì bỏ qua."""
        out = []
        for w in words:
            if w in self.idx:
                out += self.idx[w] + [N_PHONE]
        return torch.tensor(out, dtype=torch.long)

    def get(self, words):
        """→ ids (K, Pmax) long đã pad 0, lb (K,) số âm vị."""
        lens = [len(self.idx[w]) for w in words]
        ids = torch.zeros(len(words), max(lens), dtype=torch.long)
        for i, w in enumerate(words):
            ids[i, :lens[i]] = torch.tensor(self.idx[w])
        return ids, torch.tensor(lens)


def make_store(model):
    return PhoneStore() if model.bias_type == "phone" else BiasStore()


class BiasStore:
    """Feature encoder của audio TTS từng từ: work/bias.f16 (tổng frame × 1280) + work/bias.idx (từ, offset, len)."""

    def __init__(self, work=WORK):
        self.idx = {}
        for line in open(work / "bias.idx", encoding="utf-8"):
            w, o, n = line.rstrip("\n").split("\t")
            self.idx[w] = (int(o), int(n))
        total = sum(n for _, n in self.idx.values())
        self.mm = np.memmap(work / "bias.f16", dtype=np.float16, mode="r", shape=(total, 1280))

    def __contains__(self, w):
        return w in self.idx

    def get(self, words):
        """→ fb (K, Tmax, 1280) float16 đã pad 0, lb (K,) độ dài thật."""
        lens = [self.idx[w][1] for w in words]
        fb = np.zeros((len(words), max(lens), 1280), np.float16)
        for i, w in enumerate(words):
            o, n = self.idx[w]
            fb[i, :n] = self.mm[o:o + n]
        return torch.from_numpy(fb), torch.tensor(lens)
