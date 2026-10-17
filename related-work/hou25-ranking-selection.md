# Hou25 — Ranking and Selection of Bias Words (Interspeech 2025, pp. 5183–5187)

Hou, Gong, Zhang, Wang, Qian (SJTU). DOI 10.21437/Interspeech.2025-646.
Đọc **toàn văn PDF** ngày 03/10/2026. **Không có code public**: đã tìm lại 03/10, trang ISCA cũng không có link.

Quy ước: **[P]** = paper nói (kèm §/Bảng); **[TD]** = ta diễn giải hoặc ta tự chọn khi cài đặt.

---

## 1. Ý tưởng một câu
Trước khi đưa list vào prompt của Whisper, dùng một **scorer** để chấm điểm từng bias word (từ đó có trong câu nói hay không), rồi chỉ giữ **top-50**. Mục đích là list ngắn lại, prompt không bị "loãng". [P] §1, §2.1

## 2. Pipeline
```
bias word ──edge-tts──► bias audio ─┐
                                    ├─► encoder Whisper-large-v3 ─► Linear 1280→368
câu nói (speech audio) ─────────────┘                    │
                                                         ▼
                      Cross-attention 2 chiều (8 head, dropout 0.1)
                      F_speech→bias = CA(Q=F_speech, K=V=F_bias)
                      F_bias→speech = CA(Q=F_bias,   K=V=F_speech)
                                                         ▼
                      4 ma trận cosine:  S1 = sim(F_speech, F_bias)
                                         S2 = sim(F_speech, F_bias→speech)
                                         S3 = sim(F_speech→bias, F_bias)
                                         S4 = sim(F_speech→bias, F_bias→speech)
                                         S  = stack(S1..S4)   → 4 kênh
                                                         ▼
                      CNN 4 lớp (32, 64, 128, 256 kênh, kernel 3)
                      → AdaptiveAvgPool → vector 256 chiều
                                                         ▼
                      Linear + softmax → điểm của bias word
                                                         ▼
                      Xếp TĂNG dần (điểm cao nhất đứng CUỐI), lấy top k=50
                                                         ▼
                      Prompt "naive": "w1, w2, ..., w50" → Whisper-large-v3-turbo
```
Nguồn: [P] §2.2 (Eqn 1–2), §3.3. Thứ tự tăng dần: §4.1.2 + Bảng 3.

## 3. Train
- **Dữ liệu:** train-clean-100; validate trên dev-clean; test trên test-clean/test-other. [P] §3.1
- **Negative:** 40% chọn theo **edit distance nhỏ nhất** so với từ đúng (negative khó), 60% chọn ngẫu nhiên. [P] §2.2, §3.3
- **Learning rate:** 5e-5 cho list IS21; 1e-4 cho list NER. [P] §3.3
- **TTS:** edge-tts. **ASR:** Whisper-large-v3-turbo. **Encoder của scorer:** Whisper-large-v3. [P] §3.3
- **List NER:** chạy `xlm-roberta-large-finetuned-conll03-english` trên từng từ của LibriSpeech, thu được 4.365 từ. [P] §3.2.2

## 4. Số chính (test-clean / test-other)
**Bảng 1** — Whisper-turbo, prompt naive:

| | N=100 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|
| không prompt | WER 3.2/5.3 · B 10.4/19.7 | (như cột N=100) | (như cột N=100) | (như cột N=100) |
| + prompt (cả list) | 2.7/4.8 · **7.8/15.2** | 2.9/5.2 · 9.5/19.1 | 2.9/5.2 · 10.0/19.3 | 2.9/5.2 · **9.8/19.5** |
| + scorer, top-50 | 2.3/4.2 · **5.4/10.0** | 2.4/4.3 · 5.4/10.7 | 2.4/4.4 · 5.6/11.3 | 2.5/4.5 · **5.8/11.7** |

→ "Giảm hơn 40%" trong abstract ứng với N lớn, ví dụ N=2000: 9.8 → 5.8 (−41%). Ở N=100: 7.8 → 5.4 (−31%); paper viết "above 30%". [P] §4.1.1

**Bảng 2** (N=100, so loại prompt):
- no distractor (chỉ từ đúng): WER 4.93/7.64, B-WER **5.59/10.02**. B-WER tốt nhất, nhưng WER tăng do hallucination.
- naive: 2.67/4.78, B-WER 7.78/15.21.
- spoken: 2.91/4.95, B-WER 8.30/15.50.

**Bảng 3** (N=100, thứ tự từ trong prompt; text layout PDF bị lệch, gán hàng theo §4.1.2):
- random: 7.78/15.21
- **ascending: 5.73/10.15** (tốt nhất)
- descending: 10.03/19.44

**Bảng 4** (list NER 4.365 từ): không prompt B-WER 16.70/25.15 → top-50: 11.51/17.74 (khoảng −31%, khớp §4.2). Gán hàng của top-100 và top-200 chưa chắc do layout PDF bị lệch.

Áp dụng cho TCPGen-Whisper: giảm khoảng 13% ở N lớn. [P] §4.2

## 5. Paper KHÔNG nói, phải tự quyết khi implement [TD]
| Chỗ thiếu | Lựa chọn đề xuất | Loại |
|---|---|---|
| Encoder Whisper-large-v3 đóng băng hay train? | Đóng băng; chỉ train Linear + CA + CNN + head | B |
| `sim` tính trên gì: frame × frame cho **từng** bias word? | S_i có kích thước T_speech × T_bias cho mỗi từ → CNN chạy riêng từng từ | B |
| Softmax "relative to the entire speech input" là softmax trên các từ, hay 2 lớp có/không? | Logit nhị phân + BCE. Softmax trên các từ khó với câu có nhiều từ đúng | C, cần ghi rõ |
| Số negative cho mỗi positive, batch, số epoch | Ví dụ 1:4, chọn theo dev-clean | B |
| Audio có pad 30s cho encoder Whisper không (TTS rất ngắn)? | Cắt feature theo độ dài thật (nếu không, S toàn là padding) | B |
| Câu không có từ hiếm nào | vẫn lấy top-50 (vì list IS21 luôn có distractor) | A |
| Normalizer khi tính B-WER, xử lý prompt > 224 token | Không nêu. Ta dùng scorer IS21 + cắt 223 token cuối (`run_whisper.py`) | B |
| Chi phí: N=2000 → 2000 lần chạy encoder cho bias audio mỗi câu? | Tính trước, cache feature của **mỗi từ một lần** (từ dùng chung giữa các câu) | B |

## 6. Mã giả (nếu tự viết) [TD]
```python
# chuẩn bị (1 lần): mỗi từ trong kho 209k → edge-tts → wav → enc(wav)[:T_thật] → cache
class Scorer(nn.Module):
    proj = Linear(1280, 368)
    ca_sb = MultiheadAttention(368, 8, dropout=0.1)   # Q = speech
    ca_bs = MultiheadAttention(368, 8, dropout=0.1)   # Q = bias
    cnn = Seq(Conv2d(4,32,3), ReLU, Conv2d(32,64,3), ReLU, Conv2d(64,128,3), ReLU, Conv2d(128,256,3), ReLU)
    head = Seq(AdaptiveAvgPool2d(1), Flatten, Linear(256, 1))

    def forward(self, Fs, Fb):              # Fs: (Ts,1280) của câu; Fb: (Tb,1280) của 1 từ
        Fs, Fb = self.proj(Fs), self.proj(Fb)
        Fsb = self.ca_sb(Fs, Fb, Fb)        # (Ts,368)
        Fbs = self.ca_bs(Fb, Fs, Fs)        # (Tb,368)
        S = stack([cos(Fs,Fb), cos(Fs,Fbs), cos(Fsb,Fb), cos(Fsb,Fbs)])   # (4,Ts,Tb)
        return self.head(self.cnn(S))       # logit: "từ này có trong câu"

# suy luận: điểm cho cả N từ → sort tăng dần → lấy 50 từ cuối → ", ".join → prompt Whisper
```

## 7. Liên hệ với nhóm [TD]
- Scorer **chỉ chọn trong list cho sẵn**, và list IS21 luôn chứa từ đúng (giống PTC-Bias). Gap G1 vẫn đứng.
- Bảng 2 (no distractor: B-WER tốt nhất nhưng WER tăng) khớp quan sát của nhóm trên Qwen: oracle tốt nhất. Riêng Qwen thì WER *không* tăng (1.63 → 1.22), tức Qwen ít hallucinate hơn Whisper khi chỉ có từ đúng.
- Kết quả "điểm cao đặt cuối prompt thì tốt hơn" (Bảng 3) nên thử lại trên Qwen3-ASR. Thí nghiệm rẻ, dùng ngay protocol hiện có.
- Chi phí cao nhất khi tái hiện: TTS + encoder cho 209k từ (chỉ cần tính 1 lần), cộng với train trên clean-100.
