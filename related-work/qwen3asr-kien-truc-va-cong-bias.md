# Qwen3-ASR: kiến trúc thật và các "cổng" để cắm scorer (16/11/2026)

> Câu hỏi: scorer cho ra (p, t_đầu, t_cuối) cho từng từ trong list. Đưa thông tin đó vào **Qwen3-ASR-1.7B** thế nào cho "bias contextual adaptive" (tự điều chỉnh theo từng câu, từng đoạn)?
> Quy ước: **[code]** = đọc từ code `qwen_asr` 0.0.6 + `config.json` của checkpoint 1.7B; **[paper nói]** = có trích nguồn; **[đo]** = số nhóm đã chạy; **[ta]** = diễn giải của mình; **[ý tưởng]** = chưa chạy.

---

## 1. Qwen3-ASR chạy thế nào: theo đường đi của một câu 10 giây

```
wav 16 kHz
  │ 128 mel, bước 10 ms → 1000 frame
  ▼
[AuT encoder 24 lớp, d=1024]                              ← "tai"
  │ cắt thành khúc 1 s (100 frame) → 3 conv2d stride 2 → 13 vector/khúc
  │ attention chỉ trong cửa sổ 8 s (104 vector)
  │ ln_post → proj1 (1024→1024) → GELU → proj2 (1024→2048)  ← projector nằm ngay trong encoder
  ▼
130 vector "audio token", mỗi cái 2048 chiều (≈ 77 ms/vector)
  │ thay vào chỗ 130 token <|audio_pad|> trong prompt (masked_scatter)
  ▼
[Qwen3-1.7B, 28 lớp, d=2048, GQA 16/8 head]                ← "miệng"
  prompt = <|im_start|>system\n{CONTEXT}<|im_end|>
           <|im_start|>user\n<|audio_start|>[130 audio token]<|audio_end|><|im_end|>
           <|im_start|>assistant\nlanguage English<asr_text>
  │ prefill 1 lần cả prompt → KV cache
  │ rồi sinh từng token chữ (greedy), mỗi token nhìn lại TẤT CẢ audio token bằng self-attention
  ▼
"he hoped there would be stew for dinner ..."
```

Chi tiết **[code]** (`modeling_qwen3_asr.py`, `config.json`):

| Thành phần | Giá trị | Ý nghĩa cho việc bias |
|---|---|---|
| Tốc độ audio token | 13 token / 1 s khúc (`_get_feat_extract_output_lengths`); report ghi 12.5 Hz, 80 ms (arXiv 2601.21337 §2.1) | Biết **chính xác** audio token thứ k ứng với giây nào: khúc `k // 13`, vị trí `k % 13` |
| Encoder | 24 lớp, 16 head, d=1024, conv theo khúc 1 s, attention cửa sổ 8 s; positional embedding **reset mỗi khúc 1 s** | Vector audio chỉ "nhìn" trong 8 s → mang thông tin cục bộ theo thời gian |
| Projector | 2 lớp Linear + GELU, nằm cuối encoder | Không có Q-Former hay nén theo nội dung → **một audio token = một lát thời gian** |
| Ghép audio vào LLM | `inputs_embeds.masked_scatter(audio_mask, audio_features)` | Audio chỉ là một đoạn "prompt mềm" nằm giữa chuỗi; **không có cross-attention** như Whisper |
| Vị trí (RoPE) | MRoPE 3 phần nhưng cả 3 dùng cùng id 1 chiều | Audio và chữ dùng chung trục vị trí; chèn thêm token vào giữa audio là được về mặt kỹ thuật |
| LLM | Qwen3-1.7B, 28 lớp, d=2048, q-norm/k-norm, embedding và lm_head dùng chung (tied), vocab 151 936 | Logits trên vocab Qwen; một từ hiếm thường bị tách thành nhiều token BPE |
| Context | Đặt trong **system prompt, đứng TRƯỚC audio** (chat template trên HF) | Do attention nhân quả, **audio token ở các lớp LLM "đọc" được list**, còn list không đọc được audio → list dài làm nhiễu cả biểu diễn audio **[ta]** |
| Decode | vLLM `temperature=0` → greedy; HF `generate` mặc định | Can thiệp ở logits mỗi bước là dễ (LogitsProcessor) |

**Train [paper nói, arXiv 2601.21337 §2.2]:** (1) pretrain AuT bằng khoảng 40 triệu giờ nhãn giả; (2) pretrain Omni 3T token đa nhiệm; (3) SFT ASR, **có dữ liệu context biasing**: model học dùng chữ trong system prompt như kiến thức nền. Report **không** mô tả cách dựng dữ liệu này, cũng không báo số biasing; (4) RL bằng GSPO, khoảng 50k câu.
Cùng họ còn có **Qwen3-ForcedAligner-0.6B** (§3): đưa transcript có chèn ô `[time]`, model đoán chỉ số frame 80 ms cho từng ô, chạy không tự hồi quy.

**Hệ quả thực tế [đo, `bench/results.md` §3, test-other, wnorm]:** B-WER tăng đều theo độ dài list: oracle 7.73 → N100 11.61 → N500 14.88 → N2000 16.45, **tệ hơn cả không context (16.08)**. Lọc bằng scorer p ≥ 0.5 rồi mới đưa vào prompt: 9.71 (§5).
Paper độc lập cũng thấy vậy **[paper nói, Huber & Waibel, arXiv 2608.05759, Table I]**: Qwen3-ASR test-other B-WER 10.26 (0 nhiễu) → 21.06 (250 nhiễu). Họ còn thấy speech LLM **nhạy với thứ tự từ trong prompt**: từ đặt đầu prompt được lợi hơn (§IV-A). Lọc list bằng Qwen3-Omni cứu được phần lớn (Table III), nhưng mỗi cặp audio–từ tốn một lượt chạy model.

---

## 2. Có những cổng nào để đưa tín hiệu vào?

Theo đường đi ở mục 1, có 6 chỗ có thể chạm. Cột "ăn khớp với scorer" xét xem cổng đó có dùng được **p**, **vị trí**, và lựa chọn **"không từ nào"** hay không.

| # | Cổng | Ví dụ paper | Cần train? | Dùng p | Dùng vị trí | Ghi chú cho Qwen3-ASR |
|---|---|---|---|---|---|---|
| A | **Prompt** (system context) | Hou25; BR-ASR (2505.19179); CTC-assisted LLM-ASR (2411.06437); Qwen-Audio-3.0-ASR P0/P1 (2609.07549 §3.4, §5.4) | Không (Qwen đã được SFT cho kênh này) | Gián tiếp: chọn từ nào vào | Không | Kênh "chính chủ"; đã đo được lợi khi lọc bằng p. Trần thấp với lỗi gần âm (F12) |
| B | **Nhúng vào chuỗi audio** (cộng vector hoặc chèn token tại frame của từ) | Spike-triggered CB (2310.04657, cho AED+CTC); SJTU 2024 dùng embedding bias có tag | Có (adapter, LLM có thể frozen) | Có | **Có, tự nhiên nhất** | Qwen chưa từng thấy kiểu input này → không train là không chạy |
| C | **Attention** (tăng/giảm attention tới token nào đó) | Steering cho audio LLM: AVS (2510.12851), 2606.11400; chưa thấy ai dùng cho biasing | Không | Có | Có | Training-free, nhưng chưa có tiền lệ cho ASR → rủi ro |
| D | **Logits** (cộng điểm cho token của từ khi sinh) | **LOGIC** (2601.15397): cây tiền tố + cộng λ + thu hồi điểm khi lệch nhánh, Phi-4-mini frozen, list tới 2 100 | Không | Có (λ theo p) | Không (họ không gate thời gian) | Chỉ cần logits + tokenizer → cắm mọi model. Đây là **mức 1** trong `PROPOSAL.md` |
| E | **Sửa sau decode** | PTC-Bias (2609.28727, có span); GER (2307.04172) | Tùy | Có | Có | **Mức 2**; không cứu được khi decoder đã viết lệch cả câu |
| F | **Train lại / RL** | Hotword retrieval + GRPO (2512.21828); IBM vị trí bias word (2604.12398, chỉ làm loss phụ); Qwen-Audio-3.0-ASR (RL thưởng hotword) | Có, train ASR | — | — | Trái với "model frozen, cắm mọi SpeechLLM" → chỉ là đối chứng |

**Điểm trống [ta, trong phạm vi khoảng 15 lượt tra hôm nay]:** chưa thấy paper nào dùng **vị trí của bias word để gate can thiệp lúc decode** trên SpeechLLM decoder-only. LOGIC gần nhất ở cổng D nhưng boost ở mọi thời điểm. PTC-Bias có span nhưng chỉ dùng để sửa sau. Đây đúng là gap đã ghi ở `PROPOSAL.md` §0.3; hôm nay tra thêm, chưa thấy ai lấp.

**Cổng đi ngược chiều (từ Qwen ra):** decoder đang nghe giây nào? Whisper có alignment heads; Qwen thì không có cross-attention. Có 4 cách:
1. **Self-attention từ token chữ tới đoạn audio** (DOA, arXiv 2605.31432 §2.1): lấy attention của token đang sinh tới các audio token, **trung bình mọi lớp và head**, argmax ra frame. Training-free, chạy trên Qwen3-Omni và Phi-4-MM. Họ **không đo** độ chính xác so với alignment thật, chỉ đo chất lượng dịch.
2. **Căn chữ đã viết với audio** bằng aligner ngoài (G6a của mình), hoặc Qwen3-ForcedAligner.
3. **Gradient theo input** (arXiv 2607.06831): mọi model khả vi; tốn một lượt backward cho mỗi token.
4. Không cần biết: dùng cổng A hoặc E.

---

## 3. "Adaptive" nghĩa là gì ở đây

Để tránh mơ hồ, mình tách "adaptive" thành 4 thứ cụ thể, đều lấy từ output của scorer:

| Thứ tự điều chỉnh | Cách làm | Scorer cung cấp |
|---|---|---|
| 1. **Bao nhiêu từ** cho câu này | Không cố định top-50; câu không có từ hiếm thì 0 từ | p + lớp "không từ nào" |
| 2. **Đẩy mạnh bao nhiêu** | Độ mạnh tỉ lệ với p (kiểu P0/P1 của Qwen-Audio-3.0, nhưng liên tục và không cần train) | p |
| 3. **Đẩy lúc nào** | Chỉ bật khi decoder đang nghe [t_đầu, t_cuối] | span |
| 4. **Từ nào thắng tại chỗ đó** | Các từ gần âm trỏ cùng một span thì chỉ từ có p cao nhất được đẩy | p theo span (cạnh tranh) |

Ví dụ: câu *"... the **Bertolucci** film ..."*, list N2000 có `bertolucci` (p = 0.93, span 2.10–2.80 s) và `bertolini` (p = 0.41, cùng span).
- Cách Hou25: cả hai (cùng 48 từ khác) vào prompt, đẩy suốt câu.
- Cách adaptive: chỉ `bertolucci` được đẩy, chỉ trong khoảng 2.1–2.8 s, mạnh theo 0.93; 48 từ còn lại không đụng tới.

---

## 4. Đề xuất cách cắm cho Qwen3-ASR [ý tưởng, chưa chạy]

### Đề xuất 1. Prompt thích ứng + boost logits có gate (cổng A + D)
- Prompt: chỉ đưa các từ có p ≥ θ_cao (thường 0–3 từ). Kênh này Qwen đã được SFT nên "an toàn".
- Logits: kiểu LOGIC (cây tiền tố, thu hồi điểm khi lệch nhánh), nhưng λ = f(p) và **chỉ bật khi decoder nằm trong span**.
- Vị trí decoder: thử cả (i) attention tới audio kiểu DOA và (ii) căn chữ đã viết (G6a).
- Vì sao: tách đúng 2 việc. Prompt cho biết **có** từ gì (Qwen giỏi việc này, oracle 7.73). Gate logits sửa lỗi **viết chính tả quen hơn** mà prompt không sửa được (F12).
- Rủi ro: boost lặp từ (đã thấy trên Whisper, +2.02 chèn lặp). Cần luật "mỗi từ chỉ đẩy một lần trong span".

### Đề xuất 2. Gate attention tới chữ trong prompt (cổng A + C) — mới hơn, rủi ro hơn
- Đặt list nhỏ (p ≥ θ_thấp, vài chục từ) vào system prompt như bình thường.
- Mỗi bước decode: nếu decoder đang ở span của từ w, **cộng một lượng vào attention logit** từ token đang sinh tới các token prompt của w (và trừ cho các từ khác). Ngoài span, list gần như "tắt".
- Vì sao hợp với Qwen: dùng đúng kênh mà Qwen đã học (đọc chữ trong context để chép), chỉ điều khiển **lúc nào** được đọc. Không đổi vocab, không đổi trọng số.
- Rủi ro: chưa có tiền lệ; độ nhạy theo lớp/head chưa biết (DOA thấy 6 lớp đơn lẻ làm hỏng hoàn toàn). Phải đo trước khi tin.

### Đề xuất 3. Chèn "hint" vào đúng chỗ trong chuỗi audio (cổng B) — cần train adapter
- Sau audio token thuộc span của w, chèn vài token embedding của w (hoặc cộng một vector học được vào các audio token trong span), kèm p.
- Train adapter nhỏ, Qwen frozen (LoRA tùy chọn).
- Vì sao: vị trí được đưa vào **trực tiếp**, không cần đoán decoder đang nghe đâu.
- Rủi ro: phá vỡ mục tiêu "train một lần, cắm mọi SpeechLLM", vì adapter gắn với không gian embedding của Qwen. Hợp làm **trần tham khảo** hơn là phương pháp chính.

**Mình nghiêng về Đề xuất 1 làm chính, Đề xuất 2 làm ablation.** Lý do: 1 khớp với mức 1 đã chốt (chỉ cần logits + tokenizer), và trên Whisper đã có bằng chứng gate thắng prompt (B-WER 7.53 so với 9.69). Đề xuất 2 là điểm mới riêng cho SpeechLLM decoder-only, nhưng phải chờ thí nghiệm bước 0.

---

## 5. Thí nghiệm rẻ nhất cần làm trước (không train, 1 GPU)

**Bước 0. Qwen3-ASR có "biết" mình đang nghe giây nào không?**
- Teacher-forcing transcript đúng; lấy attention từ mỗi token chữ tới 130 audio token; đổi chỉ số audio token sang giây (khúc 1 s × 13).
- So với `protocol/<split>/align.tsv` (MFA đã có): sai số trung vị, % nằm trong span ±0.2 s. Làm theo từng lớp × head để tìm "alignment heads" của Qwen (giống Simul-Whisper đã làm với Whisper).
- So thêm với Qwen3-ForcedAligner và với G6a.
- Quyết định: nếu có head lệch ≤ 0.1 s → đề xuất 1 và 2 đều làm được với chi phí gần như bằng 0. Nếu không → dùng G6a.

**Bước 1. Trần của gate trên Qwen:** từ đúng + span MFA, boost logits có gate so với prompt oracle (7.73). Giống E1 đã làm trên Whisper.

**Bước 2. Thay oracle bằng scorer** (p, span thật), N2000 và `missing_N`, so với prompt lọc p ≥ 0.5 (9.71) và PTC-Bias (7.76, bản 0.6B).

---

## 6. Nguồn (mức đã đọc)

| Nguồn | Mức đọc | Dùng cho |
|---|---|---|
| Code `qwen_asr` 0.0.6 (`modeling_qwen3_asr.py`, `inference/qwen3_asr.py`), `config.json` Qwen3-ASR-1.7B, chat template trên HF | đọc trực tiếp | Mục 1 |
| [Qwen3-ASR Technical Report, arXiv 2601.21337](https://arxiv.org/html/2601.21337) §2.1, §2.2, §3 | HTML, qua tóm tắt | Kiến trúc, train, ForcedAligner |
| [Huber & Waibel, arXiv 2608.05759](https://arxiv.org/abs/2608.05759) Table I–III | toàn văn PDF | Qwen3-ASR nhạy nhiễu, nhạy thứ tự |
| [Qwen-Audio-3.0-ASR, arXiv 2609.07549](https://arxiv.org/abs/2609.07549) §3.4, §5.4 | trích text PDF các đoạn hotword | P0/P1, CTC hypothesis để lấy hotword |
| [LOGIC, arXiv 2601.15397](https://arxiv.org/html/2601.15397) §3, Table 1, 2, 4 | HTML, qua tóm tắt | Cổng D |
| [DOA, arXiv 2605.31432](https://arxiv.org/html/2605.31432) §2.1, §4 | HTML, qua tóm tắt | Vị trí decoder từ self-attention |
| [CTC-assisted LLM-ASR, arXiv 2411.06437](https://arxiv.org/abs/2411.06437) | abstract | Cổng A |
| [BR-ASR, arXiv 2505.19179](https://arxiv.org/abs/2505.19179) | abstract | Cổng A, list 200k |
| [Gradient-based alignment, arXiv 2607.06831](https://arxiv.org/abs/2607.06831) | abstract | Vị trí decoder, cách 3 |
| [Hotword retrieval + RL, arXiv 2512.21828](https://arxiv.org/abs/2512.21828) | kết quả tìm kiếm | Cổng F |
| [AVS, arXiv 2510.12851](https://arxiv.org/html/2510.12851v1); [2606.11400](https://arxiv.org/pdf/2606.11400) | kết quả tìm kiếm | Cổng C (steering) |
| [TRADE, arXiv 2606.08486](https://arxiv.org/html/2606.08486) | HTML, qua tóm tắt | Ví dụ thêm transducer vào speech LLM để có đầu ra theo frame; cần train |
| [Survey 2502.19548](https://arxiv.org/pdf/2502.19548), [2410.18908](https://arxiv.org/html/2410.18908v5) | chưa đọc | Đọc nền về SpeechLLM nếu cần |
| HotBias (HF `coming245/HotBias`) | model card | Bias có gate cho ASR frozen, chuyển được giữa model; **không rõ cắm ở đâu, chưa có paper** → cần đọc code nếu dùng |
