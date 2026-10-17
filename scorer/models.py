"""Các kiến trúc scorer: (câu nói, 1 bias word) → logit "từ này có trong câu".

Giao diện chung, cho MỘT câu: forward(fs, fb, lb) → (K,) logit
  fs (Ts, 1280)      feature encoder của câu nói
  fb (K, Tb, 1280)   feature của K bias word (audio TTS), đã pad 0;  lb (K,) độ dài thật

  hou25    Hou25 §2.2: cross-attention 2 chiều → 4 ma trận cosine → CNN 4 lớp → logit         [A + B]
  nocross  ablation: chỉ S1 = cos(speech, bias) → CNN (bỏ cross-attention)                      [C]
  maxsim   late-interaction kiểu ColBERT: mỗi frame của bias word tìm frame speech khớp nhất,
           lấy trung bình. Phía bias KHÔNG phụ thuộc câu → tính trước 1 lần cho cả kho 209k,
           chấm N từ chỉ còn 1 phép nhân ma trận → dùng làm tầng lọc rẻ (cascade)            [C]

Bias là chuỗi ÂM VỊ thay cho audio TTS (bias_type="phone"): fb = (K, P) id âm vị, lb = số âm vị.
Không cần TTS → từ do LLM vừa sinh chấm được ngay.                                            [C]
  pmaxsim  maxsim, phía bias = encoder âm vị
  pxattn   âm vị của từ "hỏi" vào audio qua 2 lớp cross-attention → pool → MLP → logit.
           sigmoid(logit) = xác suất từ có trong câu → lọc theo ngưỡng, list được phép rỗng
  hxattn   (A + B) như pxattn nhưng từ ứng viên nhìn CẢ audio LẪN bản chép lần 1 của Qwen (âm vị),
           ra 3 lớp: 0 không có / 1 có, Qwen đã chép đúng / 2 có, Qwen chép SAI ("từ khó" = đáng bias).
           Train thêm loss theo cả list: softmax trên [ô "không chọn"] + mọi ứng viên.          [C]
           forward(fs, fb, lb, hyp) → (K, 3); hyp = id âm vị bản chép lần 1, có token ranh giới từ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_mask(lb, T):
    return torch.arange(T, device=lb.device)[None] >= lb[:, None]          # True = pad


def cos(x, y):
    return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(-1, -2)


class Hou25(nn.Module):
    cheap, bias_type = False, "tts"

    def __init__(self, d=368, heads=8, drop=0.1, cross=True):
        super().__init__()
        self.cross = cross
        self.proj = nn.Linear(1280, d)                                     # chung cho speech và bias [B]
        if cross:
            self.ca_sb = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)   # Q = speech
            self.ca_bs = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)   # Q = bias
        ch = [4 if cross else 1, 32, 64, 128, 256]
        self.cnn = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.ReLU())])
        self.out = nn.Linear(256, 1)

    def forward(self, fs, fb, lb):
        K, Tb, _ = fb.shape
        mask = pad_mask(lb, Tb)
        s = self.proj(fs).unsqueeze(0).expand(K, -1, -1)                    # (K, Ts, d)
        b = self.proj(fb)                                                   # (K, Tb, d)
        if self.cross:
            sb, _ = self.ca_sb(s, b, b, key_padding_mask=mask, need_weights=False)
            bs, _ = self.ca_bs(b, s, s, need_weights=False)
            S = torch.stack([cos(s, b), cos(s, bs), cos(sb, b), cos(sb, bs)], 1)   # (K, 4, Ts, Tb)
        else:
            S = cos(s, b).unsqueeze(1)
        valid = (~mask)[:, None, None, :].to(S.dtype)
        h = self.cnn(S * valid) * valid
        h = h.sum((2, 3)) / (valid.sum((2, 3)) * S.shape[2])               # avg pool chỉ trên vùng thật [B]
        return self.out(h).squeeze(-1)


class PhoneEnc(nn.Module):
    """(K, P) id âm vị → (K, P, d). Id 0 = pad."""

    def __init__(self, n_phone, d=256, heads=4, layers=2, drop=0.1, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(n_phone, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True), layers,
                                        enable_nested_tensor=False)

    def forward(self, ids, lb):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device).clamp(max=self.pos.num_embeddings - 1))
        return self.tf(x, src_key_padding_mask=pad_mask(lb, ids.shape[1]))


class MaxSim(nn.Module):
    cheap = True

    def __init__(self, d=256, heads=4, drop=0.1, n_phone=None):
        super().__init__()
        self.bias_type = "phone" if n_phone else "tts"
        self.ls = nn.Linear(1280, d)
        self.ts = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
        if n_phone:
            self.pe = PhoneEnc(n_phone, d, heads, drop=drop)
        else:
            self.lb = nn.Linear(1280, d)
            self.tb = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
        self.a, self.c = nn.Parameter(torch.tensor(10.0)), nn.Parameter(torch.tensor(-5.0))

    def enc_speech(self, fs):                                               # (Ts, 1280) → (Ts, d) chuẩn hoá
        return F.normalize(self.ts(self.ls(fs).unsqueeze(0)).squeeze(0), dim=-1)

    def enc_bias(self, fb, lb):                                             # không phụ thuộc câu → cache được
        if self.bias_type == "phone":
            return F.normalize(self.pe(fb, lb), dim=-1)
        h = self.tb(self.lb(fb), src_key_padding_mask=pad_mask(lb, fb.shape[1]))
        return F.normalize(h, dim=-1)

    def score(self, es, eb, lb):
        sim = (eb @ es.T).amax(-1)                                          # (K, Tb): frame speech khớp nhất
        valid = (~pad_mask(lb, eb.shape[1])).to(sim.dtype)
        return self.a * (sim * valid).sum(1) / lb + self.c

    def forward(self, fs, fb, lb):
        return self.score(self.enc_speech(fs), self.enc_bias(fb, lb), lb)


class PhoneXAttn(nn.Module):
    cheap, bias_type = False, "phone"

    def __init__(self, n_phone, d=256, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.ls = nn.Linear(1280, d)
        self.ts = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
        self.pe = PhoneEnc(n_phone, d, heads, drop=drop)
        self.xa = nn.ModuleList(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, fs, fb, lb):
        K, P = fb.shape
        mask = pad_mask(lb, P)
        s = self.ts(self.ls(fs).unsqueeze(0)).expand(K, -1, -1)             # (K, Ts, d)
        h = self.pe(fb, lb)                                                 # (K, P, d)
        for layer in self.xa:                                               # âm vị tự nhìn nhau + nhìn vào audio
            h = layer(h, s, tgt_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).to(h.dtype)
        mean = (h * valid).sum(1) / lb[:, None]
        mx = h.masked_fill(mask.unsqueeze(-1), -1e4).amax(1)
        return self.out(torch.cat([mean, mx], -1)).squeeze(-1)


class HypXAttn(nn.Module):
    cheap, bias_type, uses_hyp = False, "phone", True

    def __init__(self, n_phone, d=256, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.ls = nn.Linear(1280, d)
        self.ts = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
        self.pe = PhoneEnc(n_phone + 1, d, heads, drop=drop, max_len=1024)    # +1 = token ranh giới từ (hyp)
        self.typ = nn.Embedding(2, d)                                       # 0 = frame audio, 1 = âm vị hyp
        self.xa = nn.ModuleList(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 3))
        self.null = nn.Parameter(torch.tensor(0.0))                         # điểm của ô "không chọn"

    def forward(self, fs, fb, lb, hyp):
        K, P = fb.shape
        mask = pad_mask(lb, P)
        mem = self.ts(self.ls(fs).unsqueeze(0)) + self.typ.weight[0]       # (1, Ts, d)
        if len(hyp):
            hh = self.pe(hyp.unsqueeze(0), torch.tensor([len(hyp)], device=hyp.device)) + self.typ.weight[1]
            mem = torch.cat([mem, hh], 1)                                   # (1, Ts + H, d)
        mem = mem.expand(K, -1, -1)
        h = self.pe(fb, lb)
        for layer in self.xa:
            h = layer(h, mem, tgt_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).to(h.dtype)
        mean = (h * valid).sum(1) / lb[:, None]
        mx = h.masked_fill(mask.unsqueeze(-1), -1e4).amax(1)
        return self.out(torch.cat([mean, mx], -1))                          # (K, 3)


def present_logit(z):                                                       # log p(có) − log p(không có)
    return torch.logsumexp(z[:, 1:], 1) - z[:, 0]


def hard_logit(z):                                                          # log p(khó) − log p(không khó)
    return z[:, 2] - torch.logsumexp(z[:, :2], 1)


@torch.no_grad()
def score_chunks(model, fs, fb, lb, chunk=32, hyp=None):
    """Chấm K từ cho 1 câu theo từng khúc (K có thể vài nghìn) → (K,) hoặc (K, 3) float32."""
    out = []
    for i in range(0, len(fb), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = (fs, fb[i:i + chunk], lb[i:i + chunk]) + ((hyp,) if hyp is not None else ())
            out.append(model(*x).float())
    return torch.cat(out)


def build(arch, n_phone=None):
    return {"hou25": lambda: Hou25(), "nocross": lambda: Hou25(cross=False), "maxsim": lambda: MaxSim(),
            "pmaxsim": lambda: MaxSim(n_phone=n_phone), "pxattn": lambda: PhoneXAttn(n_phone),
            "hxattn": lambda: HypXAttn(n_phone)}[arch]()
