# Báo cáo thầy — (1) Bảng 1/2 kiểu Hou25, (2) vì sao phía audio không dùng ASR âm vị

> Soạn 01/11 từ `scorer/results.md` §7–10, `scorer/DEEPDIVE.md`, `related-work/2609.28727-ptc-bias.md`. Đổi tên thư mục theo ngày họp thật.

---

## Phần 1 — Kiến trúc + Bảng 1/2 + bước tiếp

### 1.1 Kiến trúc (pxattn) trong 30 giây
Hình: `meetings/2026-10-23/imgs/scorer_detail.png` (cột ②).

```
audio ──► Whisper-large-v3 encoder (ĐÓNG BĂNG) ──► Linear + 1 lớp Transformer ──► A (T × 256)
                                                                                     │ key/value
1 từ trong list ──► g2p (CMUdict + đoán) ──► âm vị ──► embedding + 2 lớp Transformer ──► P (#âm vị × 256)
                                                                                     │ query
                    2 lớp cross-attention: mỗi âm vị "tìm" mình trong audio
                    → gộp [mean; max] → MLP → sigmoid = p(từ được nói)
```
- Chấm **từng từ độc lập** → list 2000 từ = 2000 điểm. Lấy top-50 (như Hou25) hoặc giữ p ≥ 0.5.
- Train trên train-clean-100 (cùng dữ liệu Hou25), 4.96M tham số, ~332 ms/câu với list 2000 từ.
- Khác Hou25 **đúng một chỗ**: từ trong list biểu diễn bằng **âm vị** (g2p) thay cho **audio TTS**. Phía audio giống Hou25 (encoder Whisper).
- Biến thể rẻ hơn: pmaxsim (mỗi âm vị lấy frame giống nhất, không cross-attention).

### 1.2 Bảng 1 — B-WER theo N (Whisper-large-v3-turbo, chuẩn hoá plain, clean / other)
Cùng ASR, cùng list IS21, cùng prompt naive, top-50 xếp tăng dần như Hou25; **chỉ đổi scorer**. Số Hou25 chép từ paper.

| | N=100 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|
| không prompt (mình chạy) | 9.43 / 18.92 | ← | ← | ← |
| prompt nguyên list (mình chạy) | 8.04 / 13.98 | 8.87 / 18.22 | 9.20 / 18.32 | 9.15 / 18.69 |
| **Hou25 + scorer top-50 (paper)** | 5.4 / 10.0 | 5.4 / 10.7 | 5.6 / 11.3 | 5.8 / 11.7 |
| **pxattn top-50** | **4.57 / 8.92** | **4.89 / 9.21** | **5.16 / 9.79** | **5.29 / 10.00** |
| pmaxsim top-50 | 4.60 / 10.30 | 4.95 / 9.53 | 5.17 / 9.83 | 5.47 / 10.37 |

→ pxattn thắng Hou25 **8/8 ô** (thấp hơn 0.5–0.8 trên clean, 1.1–1.7 trên other); WER cũng thấp hơn (N2000: 2.35 / 4.23 so với 2.5 / 4.5). pmaxsim thắng 7/8.

### 1.3 Bảng 2 — kiểu prompt, N=100 (WER · B-WER, clean / other)
Bảng 2 của Hou25 (đã kiểm PDF, §4.1.2) **không dùng scorer**: đưa nguyên list N=100, chỉ đổi cách viết prompt. Có đúng 4 hàng; mình chạy lại đủ cả 4.

| Prompt | Hou25 (paper) | Mình |
|---|---|---|
| baseline (không prompt) | 3.23 / 5.33 · 10.38 / 19.72 | 2.64 / 4.87 · 9.43 / 18.92 |
| no distractor (chỉ từ đúng = oracle) | 4.93 / 7.64 · 5.59 / 10.02 | **20.08 / 20.89** · 5.12 / 26.24 |
| naive `w1, w2, ...` | 2.67 / 4.78 · 7.78 / 15.21 | 2.55 / 4.20 · 8.04 / 13.98 |
| spoken | 2.91 / 4.95 · 8.30 / 15.50 | 2.32 / 4.30 · 7.12 / 14.56 |

- **Hou25 nói:** chỉ từ đúng → B-WER tốt nhất nhưng WER tăng do hallucination; naive tốt hơn spoken. **Mình thấy:** cùng xu hướng nhưng nặng hơn nhiều; naive vs spoken không rõ bên nào hơn.
- Oracle chỉ hỏng ở câu có prompt 0–1 từ: riêng câu >1 từ hiếm thì oracle WER 2.02 / 4.56, B-WER 4.33 / 8.63.
- **Ý nghĩa:** list hoàn hảo vẫn làm Whisper hỏng → cách SpeechLLM dùng list là nút thắt riêng. Trên Whisper phải dùng top-50; lọc ngưỡng chỉ dùng được khi bọc spoken.
- Hàng thêm của mình (không có trong Bảng 2 của Hou25): pxattn top-50 naive 2.18 / 3.66 · **4.57 / 8.92**; p≥0.5 naive 19.58 / 21.03 · 8.42 / 29.35; p≥0.5 spoken 2.37 / 3.78 · 5.09 / 9.61.
- Chưa chạy: oracle + spoken; Bảng 3 (thứ tự từ), Bảng 4 (list NER) của Hou25.

- **Phát hiện:** prompt chỉ 0–1 từ làm Whisper **lặp chữ** ("…of god or god or god…"), WER vọt lên 10–23%. 32/40 câu hỏng nặng nhất có prompt đúng 1 từ. Bọc câu spoken (Hou25 §2.1) chữa được.
- Qwen3-ASR **không** bị lỗi này: trên Qwen, lọc p ≥ 0.5 là cấu hình tốt nhất (N2000: 4.78 / 9.71, oracle 4.01 / 7.73).
- Ablation câu có >1 từ hiếm (1289 / 1304 câu): top-50 cải thiện đều như cả tập (pxattn N100 4.37 / 8.47); thiệt hại của p≥0.5 naive nằm ở câu có ≤1 từ hiếm.

### 1.4 Giới hạn cần nói thẳng
- Số Hou25 là **số paper**, không có output từng câu → chưa có khoảng tin cậy khi so với họ.
- Baseline mình tự chạy lệch số paper khoảng ±1 theo cả 2 chiều (không prompt: 9.43 vs 10.4; nguyên list N100: 8.04 vs 7.8) → khả năng khác cấu hình decode (Hou25 có thể dùng fallback nhiệt độ — chưa kiểm).
- Scorer bản 100 h, 1 seed. Bản 960 h chưa đưa vào ASR.

### 1.5 Bước tiếp (đề xuất, cần thầy duyệt thứ tự)
| # | Việc | Vì sao | Chi phí |
|---|---|---|---|
| 1 | **Cùng lưới bảng trên Qwen3-ASR** (N100–2000 × top-50 / p≥0.5), cả scorer 960 h | Qwen là backbone chính; hiện Qwen mới có N2000 | ước ~5–6 h GPU (≈32 lượt × 10 phút) |
| 2 | **Khoảng tin cậy** (`bench/compare.py`, bootstrap) giữa các hàng mình chạy; kiểm cấu hình decode của Hou25 | Biến "thắng 8/8" thành kết luận có ý nghĩa thống kê | CPU |
| 3 | **Thí nghiệm đo trần nút thắt ②** (đề xuất 23/10): oracle / oracle + 5–50 từ nhiễu / + từ nghe giống / + vị trí thật | Chọn từ đã gần trần; ~½ lỗi còn lại là SpeechLLM dùng sai từ đã có trong list | ~1 ngày GPU |
| 4 | Head CTC âm vị trên encoder Whisper (xem Phần 2) | Gỡ điểm yếu "không xét thứ tự âm", đồng thời cho ra vị trí từ — nối thẳng vào #3 | 1–2 ngày code + train |
| 5 | Tập thứ hai (Earnings / ConEC) | Reviewer sẽ hỏi ngoài LibriSpeech | — |

---

## Phần 2 — Vì sao từ trong list đi qua g2p (âm vị), còn audio lại không qua một ASR âm vị?

### 2.1 Câu hỏi
Hàng dưới: chữ → **âm vị rời rạc** (g2p). Hàng trên: audio → **vector liên tục** của encoder Whisper.
Sao không cho audio qua một model nhận dạng âm vị (wav2vec2 / WavLM + CTC âm vị) để **hai bên cùng là âm vị** rồi so?

### 2.2 Có 3 cách làm hàng trên
| | Hàng trên ra gì | So với từ trong list bằng | Ai làm |
|---|---|---|---|
| (a) | **1 chuỗi âm vị tốt nhất** (quyết định cứng) | edit distance chuỗi–chuỗi | CTC-Filter / baseline hypmatch của nhóm (dùng hyp Qwen) |
| (b) | **xác suất âm vị theo từng frame** (CTC posterior, mềm) | Viterbi trên trie, có thứ tự | PTC-Bias (WavLM-Large + head CTC) |
| (c) | **vector encoder** (liên tục, không ép thành âm vị) | cross-attention học được | **nhóm (pxattn)**, Hou25 (phía audio) |

### 2.3 Vì sao nhóm chọn (c) — lý do theo thứ tự quan trọng
1. **Thí nghiệm có kiểm soát với Hou25 [C].** Hou25 dùng encoder Whisper cho audio. Giữ nguyên hàng trên, chỉ đổi hàng dưới (TTS → âm vị) thì mọi chênh lệch ở Bảng 1 là **do cách biểu diễn từ**, không lẫn do đổi encoder. Đổi cả hai thì không trả lời được câu "âm vị có tốt hơn TTS không".
2. **Quyết định cứng làm mất đúng những từ cần cứu.** Từ hiếm chính là chỗ ASR hay nghe sai. Nếu hàng trên chốt 1 chuỗi âm vị, từ bị nghe sai sẽ mất luôn trước khi kịp so. Bằng chứng gián tiếp của nhóm (test-other N2000, `scorer/results.md` §7):

   | | Recall_B#50 ↑ | Recall_H#50 ↓ |
   |---|---|---|
   | hypmatch — kiểu (a), khớp âm vị với hyp Qwen | 95.41 | 35.70 |
   | pxattn — kiểu (c) | 97.71 | **21.58** |
   | pmaxsim — kiểu (c) | **98.44** | 25.52 |

   Lưu ý: hypmatch đổi **hyp chữ** của Qwen sang âm vị, không phải model nhận dạng âm vị thật → chỉ là bằng chứng **gián tiếp** cho (a).
   Thêm 1 bằng chứng: hxattn (cho scorer nhìn thêm âm vị của hyp Qwen) **kém hẳn** (B-WER 6.58 / 14.07 so với 4.78 / 9.71) — nghi model học tắt theo hyp thay vì nghe audio.
3. **Không phải chốt bộ âm vị cho audio.** Encoder Whisper-large-v3 (train trên hàng triệu giờ, đã đóng băng) giữ cả thông tin ngoài âm vị (trọng âm, ngữ cảnh, giọng); cross-attention tự học cách khớp, train đầu–cuối bằng 1 loss.
4. **Thực tế:** checkpoint WavLM âm vị mà PTC dùng **không có link tải** (`bench/third_party/README.md`) → muốn làm (b) phải tự train head CTC.

### 2.4 Nhưng (c) có giá — và đây là chỗ (b) mạnh hơn
- (c) **không ép thứ tự âm**: pmaxsim coi *stone* `S T OW N` ≈ *notes* `N OW T S`; pxattn chỉ xét được một phần. (b) có Viterbi đơn điệu nên xét đủ.
- Đây là giả thuyết cho khoảng cách còn lại với PTC: Recall_B@99 PTC **16.9** từ/câu, pxattn 960 h **64**, pmaxsim 960 h 54 (`DEEPDIVE.md` §2 — **ta diễn giải, chưa kiểm**). Phần lớn khoảng cách đã truy ra là do dữ liệu train + lỗi tiền xử lý (brief 23/10, TN3).
- PTC tự báo cáo (2609.28727, Bảng 1, đã kiểm HTML v1): dùng posterior WavLM tốt hơn dùng encoder của chính Qwen (AuT): B-WER **4.10 / 7.76** so với 4.45 / 8.61 (Qwen3-ASR-0.6B). → **Paper nói** hàng trên chuyên âm vị có giúp; **ta diễn giải** rằng phần lợi đó đến từ thứ tự + dữ liệu âm vị.

### 2.5 Câu trả lời ngắn để nói
> "Hàng trên giữ encoder Whisper để so **công bằng** với Hou25: chỉ đổi cách biểu diễn từ, và âm vị thắng 8/8 ô. Không chốt audio thành chuỗi âm vị vì quyết định cứng làm mất đúng những từ hiếm bị nghe sai (baseline kiểu đó kém hơn: 95.4 so với 98.4 Recall_B#50). Cái giá là chưa xét thứ tự âm — chỗ PTC mạnh hơn. Bước tiếp em đề xuất gắn **thêm head CTC âm vị lên encoder Whisper** (lấy cái tốt của (b) mà không bỏ (c)): vừa có thứ tự, vừa ra **vị trí** của từ cho hướng position-aware."

### 2.6 Rủi ro của đề xuất head CTC
- Gần PTC → phần mới ít nếu chỉ dừng ở retrieval. Phần mới phải nằm ở **dùng vị trí** cho SpeechLLM (nút thắt ②).
- Chưa chạy: số ở 2.4 là giả thuyết.
