# Mổ xẻ Qwen3-ASR-1.7B và phân tích sâu 4 paper gần nhất

16/11/2026. Tài liệu đọc từ trên xuống, gồm 3 phần:

- **Phần I:** Qwen3-ASR bên trong. Câu hỏi: dữ liệu chạy qua những khối nào, ở mỗi khối thông tin thời gian còn hay mất.
- **Phần II:** 4 paper gần nhất (LOGIC, PTC-Bias, DOA, Qwen-Audio-3.0-ASR). Mỗi paper được phân tích theo cùng một khung: bài toán → cách làm → số liệu → giới hạn → mình lấy được gì.
- **Phần III:** ghép lại thành chỗ trống, kèm nguồn.

Bản ngắn hơn, có slide: `qwen3asr-kien-truc-va-cong-bias.md`, `qwen3asr-slide/qwen3asr-cong-bias.pdf`.

**Nhãn:**

| Nhãn | Nghĩa |
|---|---|
| **[code]** | đọc trực tiếp từ `qwen_asr` 0.0.6 (`modeling_qwen3_asr.py`) và `config.json` của Qwen3-ASR-1.7B |
| **[paper]** | paper viết vậy, có ghi Section/Table |
| **[đo]** | nhóm đã chạy (`bench/results.md`) |
| **[ta]** | mình diễn giải |
| **[ý tưởng]** | chưa chạy |

Cả 5 nguồn đều đã đọc **toàn văn** bằng PDF tải từ arXiv ngày 16/11.

---

# Phần I. Mổ xẻ Qwen3-ASR-1.7B

## I.1 Toàn cảnh: một câu 10 giây đi qua model

```
 audio 10 s, 16 kHz
   │  log-mel 128 kênh, 10 ms/khung                         [1000 × 128]
   ▼
┌──────────────────────── ENCODER AuT (chạy 1 lần / câu) ─────────────────────────┐
│ cắt khúc 1 s (100 khung)                               [10 khúc × 128 × 100]     │
│ 3 × Conv2d 3×3 stride 2, 480 kênh, GELU                [10 × 480 × 16 × 13]     │
│ gộp kênh×tần số 7680 → Linear 1024 + vị trí sin        [130 × 1024]             │
│ 24 lớp Transformer encoder (attention trong khối 8 s)  [130 × 1024]             │
│ Projector: LN → Linear 1024→1024 → GELU → Linear→2048  [130 × 2048]             │
└──────────────────────────────────────────────────────────────────────────────────┘
   │  130 vector, mỗi vector là một lát thời gian ≈ 77 ms
   ▼
┌──────────────────────── GHÉP CHUỖI ─────────────────────────────────────────────┐
│ system prompt (list) + user + 130 ô <|audio_pad|> + assistant → tokenizer → id   │
│ id → bảng embedding [151936 × 2048]                                              │
│ thay 130 ô pad bằng 130 vector audio (masked_scatter)  [L × 2048]               │
└──────────────────────────────────────────────────────────────────────────────────┘
   ▼
┌──────────────────────── LLM Qwen3-1.7B (chạy 1 lần / token sinh ra) ────────────┐
│ 28 × [RMSNorm → self-attn GQA causal → +res → RMSNorm → SwiGLU → +res]           │
│ RMSNorm → LM head (chung trọng số với embedding) → 151936 logits → greedy argmax │
└──────────────────────────────────────────────────────────────────────────────────┘
   │  token mới nối vào cuối chuỗi, lặp tới <|im_end|>
   ▼
 "language English<asr_text>he hoped there would be stew for dinner ..."
```

Ba điều cần nhớ:

1. Thời gian được giữ nguyên tới cửa LLM. **[code]**
2. Audio đi vào LLM như một đoạn "chữ" nằm giữa prompt. Không có cross-attention. **[code]**
3. List nằm trước audio, nên mask causal không cho list nhìn thấy audio. **[code]**

## I.2 Encoder AuT, từng bước

**Thông số [code]:**

| Thông số | Giá trị |
|---|---|
| `num_mel_bins` | 128 |
| `n_window` | 50 |
| `n_window_infer` | 800 |
| `downsample_hidden_size` | 480 |
| `d_model` | 1024 |
| `encoder_layers` | 24 |
| `encoder_attention_heads` | 16 |
| `encoder_ffn_dim` | 4096 |
| `activation_function` | gelu |
| `output_dim` | 2048 |
| `max_source_positions` | 1500 |

**Bước 1. Cắt khúc 1 s.** Chuỗi mel được cắt thành khúc 100 khung (= 2 × `n_window`). Phần lẻ cuối câu thành một khúc ngắn hơn.

**Bước 2. Ba lớp conv.**

- Mỗi khúc là một "ảnh" 1 kênh, kích thước 128 (tần số) × 100 (thời gian), đi qua `conv2d1..3`. Mỗi lớp: kernel 3×3, stride 2, padding 1, theo sau là GELU.

  | | Kênh × tần số × thời gian |
  |---|---|
  | Đầu vào | 1 × 128 × 100 |
  | Sau conv 1 | 480 × 64 × 50 |
  | Sau conv 2 | 480 × 32 × 25 |
  | Sau conv 3 | 480 × 16 × 13 |

- Mỗi khúc ra đúng **13 vector**. Công thức trong `_get_feat_extract_output_lengths`: `out = 13 × (số khúc đủ 100 khung) + f(phần lẻ)`.

**Bước 3. Chiếu và đánh vị trí.**

- `conv_out` là Linear(480 × 16 = 7680 → 1024).
- Cộng positional embedding hình sin: lấy `positional_embedding[:13]` rồi cộng cho **mọi khúc**. Nghĩa là số vị trí **đếm lại 0..12 ở mỗi khúc**.
- Encoder vì thế không có "đồng hồ tuyệt đối" ở đầu vào. Vector thứ 3 của khúc 0 và vector thứ 3 của khúc 7 nhận cùng một vector vị trí. **[ta]:** thứ tự giữa các khúc chỉ còn suy ra được từ nội dung, cộng với việc LLM đặt chúng liên tiếp trong chuỗi.

**Bước 4. 24 lớp encoder, attention theo khối.**

- Mỗi lớp: LayerNorm → self-attention 16 head → +res → LayerNorm → FFN 1024→4096→1024 (GELU) → +res.
- Attention **không** trải cả câu. `cu_seqlens` chia chuỗi 130 vector thành các khối cố định, mỗi khối `13 × (800 / 100) = 104` vector (= 8 s). Trong khối, attention hai chiều; giữa hai khối thì không nhìn nhau.
- Câu 10 s có 2 khối: vector 0–103 và 104–129. Khối **không trượt**, nên một từ nằm vắt qua mốc 8 s bị cắt đôi ngữ cảnh.
- Report §2.1 viết cửa sổ "dynamic 1 s – 8 s". Code infer offline dùng 8 s.

**Bước 5. Projector.** `ln_post` → `proj1` (1024→1024) → GELU → `proj2` (1024→2048). Projector nằm trong lớp encoder. Nó **không nén theo nội dung** (không có Q-Former, không pooling), nên 130 vector vào thì vẫn 130 vector ra.

**Đổi giây sang chỉ số vector [code, ta]:** `k ≈ 13 × floor(t) + floor((t − floor(t)) × 13)`.

- Ví dụ: một từ ở 3,2–3,9 s ứng với vector 41–50.
- Chiều ngược lại, vector k nằm quanh giây `floor(k / 13) + (k mod 13) / 13`.

**Report nói khác code ở 2 chỗ:**

| | Report 2601.21337 | Code / config | Cách xử lý |
|---|---|---|---|
| Tốc độ vector | §2.1: "12.5 Hz" (80 ms) | 13 vector / khúc 1 s ≈ 77 ms, do làm tròn lên khi stride 2 ba lần (100→50→25→13) | Đổi giây ↔ vector phải dùng **13/khúc**, không dùng 12,5 |
| Số lớp encoder | Figure 2 vẽ "32×" self-attention layer cho AuT; §2.1 ghi "AuT encoder with 300M parameters and 1024 hidden size" cho bản 1.7B | `encoder_layers = 24` | Tin config cho bản đang dùng |

## I.3 Ghép chuỗi: list, audio và chữ nằm chung một dãy

**Prompt thật** (chat template HF; `bench/run_qwen3asr.py` ép language English):

```
<|im_start|>system
{context: "bertolucci, bertolini, ..."}<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|> × 130<|audio_end|><|im_end|>
<|im_start|>assistant
language English<asr_text>
```

Model viết tiếp sau `<asr_text>`.

**Các id đặc biệt [code]:**

| Token | Id |
|---|---|
| `audio_token_id` | 151676 |
| `audio_start` | 151669 |
| `audio_end` | 151670 |

**Cách ghép [code]:**

1. Mọi id đi qua bảng embedding 151936 × 2048.
2. `inputs_embeds.masked_scatter(mask_audio, audio_features)` ghi đè đúng 130 hàng của ô pad bằng 130 vector từ projector.

**Vị trí:**

- Config khai báo MRoPE (`mrope_section [24, 20, 20]`, interleaved).
- Với ASR, ba trục vị trí nhận **cùng một số**, nên trên thực tế đây là RoPE 1 chiều: 0, 1, 2, … chạy liên tục từ token đầu tới token cuối. **[code]**

**Độ dài [ta]:**

- List 2000 từ hiếm khoảng ≥ 4000 token (tên lạ thường tách nhiều subword, cộng dấu phẩy).
- Như vậy phần list dài gấp hơn 30 lần 130 token audio. Toàn bộ list phải prefill trước khi sinh chữ đầu tiên.

## I.4 LLM decoder Qwen3-1.7B

**Thông số [code]:**

| Thông số | Giá trị |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` | 6144 |
| `hidden_act` | silu |
| `rope_theta` | 1 000 000 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | true |

**Một lớp [code]:**

```
x ─► RMSNorm ─► q_proj (16 head × 128) ─► RMSNorm(q) ─► RoPE ─┐
            ├─► k_proj ( 8 head × 128) ─► RMSNorm(k) ─► RoPE ─┼─► softmax(q·kᵀ/√128 + mask causal) · V ─► o_proj ─► + x
            └─► v_proj ( 8 head × 128) ───────────────────────┘
  ─► RMSNorm ─► down( SiLU(gate(x)) ⊙ up(x) )   (2048 → 6144 → 2048) ─► + residual
```

- **GQA:** 2 head Q dùng chung một cặp K/V, vì vậy KV cache nhỏ hơn một nửa.
- `q_norm`, `k_norm` là RMSNorm áp lên từng head **trước** RoPE.

**Ma trận attention** (mỗi lớp, mỗi head; hàng là token đang tính, cột là token được nhìn):

```
               cột:  list        audio       chữ
hàng: list        [ ◣ tam giác ]  [  ✗  ]     [  ✗  ]
      audio       [ nhìn list ]  [ ◣    ]     [  ✗  ]
      chữ         [ nhìn list ]  [ ★ nhìn audio ] [ ◣ ]
```

- Ô ★ là nơi duy nhất chữ đang viết "nghe" audio. Nếu có head mà ở token chữ thứ j attention dồn vào đúng các vector audio của từ j, ta đọc được "decoder đang nghe giây nào". Đây là **bước 0** cần đo. **[ta]**
- Ô ✗ ở hàng list, cột audio: list không thể "tự lọc" theo audio. Mọi việc chọn từ trong list dồn cho các token chữ. **[ta]**

**Đầu ra:** RMSNorm cuối → LM head. Ma trận LM head dùng chung với bảng embedding (tied), cho ra 151936 logits. Decode greedy (vLLM temperature 0, giống cách nhóm chạy).

**Chi phí [ta]:**

- Prefill tính một lần cả chuỗi (list + audio + phần mở đầu).
- Mỗi bước decode chỉ tính 1 token mới, dùng KV cache.
- List dài làm tăng chi phí prefill và làm mỗi bước attention phải trải trên thêm vài nghìn khóa.

## I.5 Train 4 giai đoạn [paper, 2601.21337 §2.2]

| Giai đoạn | Làm gì | Ghi chú cho biasing |
|---|---|---|
| (1) AuT pretraining | Train encoder AuT theo khung AED trên khoảng **40 triệu giờ** nhãn giả, phần lớn tiếng Trung và tiếng Anh | Encoder ổn định với cửa sổ attention động |
| (2) Omni pretraining | Từ Qwen3-Omni, đa nhiệm audio/ảnh/chữ, **3T token** | — |
| (3) ASR SFT | Dữ liệu nhỏ hơn, tách rời pretraining; có non-speech, streaming, **context biasing data** | Model "học dùng context token trong system prompt như kiến thức nền". Report **không** nói list dựng ra sao, dài bao nhiêu, có distractor không, và **không** báo số biasing |
| (4) RL (GSPO) | Khoảng **50k câu**: 35% Trung–Anh, 35% đa ngữ, 30% "functional" | Report không nói có phần nào dành cho context |

**Paper nói (§2.2):** ở SFT, model được train để chỉ làm ASR, không làm theo chỉ dẫn ngôn ngữ tự nhiên trong prompt (để chống instruction injection).

**[ta]:** vì vậy list trong system prompt là "kênh chính chủ" của Qwen3-ASR. Nhưng model chưa từng được dạy chịu list 2000 từ nhiễu. Số đo ở I.7 khớp với điều này.

## I.6 Qwen3-ForcedAligner-0.6B (cùng họ) [paper, §3]

**Cách hoạt động:**

- Input là audio + transcript, trong đó sau mỗi từ chèn ô `[time]` cho mốc đầu và mốc cuối.
- Một lớp tuyến tính dự đoán **chỉ số khung 80 ms** cho mọi ô `[time]` cùng lúc. Đây là cách giải không tự hồi quy (NAR). Tối đa 3750 lớp, tức 300 s.
- Train bằng nhãn giả MFA. Loss chỉ tính tại các ô `[time]`, chuỗi output và label không dịch lệch nhau.
- RTF khoảng 0,001.

**[ta]:** đây là công cụ có sẵn để trả lời "từ đã viết nằm tới giây nào" (cách 2 trong 4 cách định vị decoder). Nó không đọc bên trong Qwen3-ASR, nên cắm được cho mọi model.

## I.7 Hệ quả đo được trên chính Qwen3-ASR-1.7B

**Số của nhóm [đo]:** `bench/results.md`, test-other, wnorm, scorer IS21 chính thức.

| Context | B-WER | WER |
|---|---|---|
| none | 16.08 | 3.41 |
| oracle | **7.73** | 2.43 |
| N100 | 11.61 | 3.17 |
| N500 | 14.88 | 3.33 |
| N1000 | 15.92 | — |
| N2000 | **16.45** | 3.55 |
| lọc p ≥ 0.5 bằng scorer pxattn | 9.71 | 2.68 |

**Huber & Waibel, arXiv 2608.05759, Table I–III [paper]:**

- Qwen3-ASR test-other B-WER tăng từ 10.26 (N = 0) lên 21.06 (N = 250).
- Model nhạy với thứ tự từ trong prompt.
- Định nghĩa từ hiếm của họ khác IS21, nên không đặt cạnh số của nhóm.

**Đọc bảng [ta]:** prompt rất mạnh khi list ngắn và sạch (oracle). List 2000 từ còn tệ hơn không có list (16.45 so với 16.08). Bài toán là **chọn và đưa đúng lúc**, không phải "có prompt hay không".

## I.8 Các chỗ chạm được, gắn vào sơ đồ I.1

| Cổng | Chạm vào khối nào | Cần train? | Dùng được vị trí? |
|---|---|---|---|
| A. Prompt | phần system của chuỗi (I.3) | không | gián tiếp (chỉ chọn từ) |
| B. Vector audio | 130 vector sau projector (I.2 bước 5) | **có** (adapter) | có, trực tiếp theo chỉ số k |
| C. Attention | ma trận ★ hoặc cột list (I.4) | không | có, nếu đọc được "đang nghe đâu" |
| D. Logits | 151936 logits (I.4 đầu ra) | không | có, nếu đọc được "đang nghe đâu" |
| E. Sửa sau | text đã ra | không | có (như PTC) |
| F. Train lại | cả model | có | — |

---

# Phần II. Bốn paper gần nhất

Mỗi paper được phân tích theo cùng một khung:

1. Thông tin
2. Bài toán họ giải
3. Cách làm, kèm ví dụ
4. Setup
5. Số chính
6. Paper nói / ta diễn giải
7. Giới hạn
8. Mình lấy gì
9. So được với số của nhóm không

## II.1 LOGIC: boost logits theo cây tiền tố, không cần train

**1. Thông tin.**

- Tên đầy đủ: "LOGIC: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration".
- Tác giả: Peidong Wang, Jian Xue, Jinyu Li (Microsoft).
- arXiv 2601.15397v4, ngày 24/09/2026.
- Paper không nhắc tới code công bố.

**2. Bài toán.**

- Prompt không có "núm vặn" độ mạnh biasing, và list dài làm tốn chi phí.
- GEC (dùng LLM viết lại transcript) có thể chèn tên không ai nói.
- Mục tiêu: đưa bias vào **lúc decode, ở logits**, có trọng số rõ ràng. §1, §2.

**3. Cách làm (§3).**

- **Biểu diễn list.**
  - Mỗi entity được tokenize vài kiểu (có hoặc không có dấu cách đầu) rồi chọn cách ngắn nhất ("best-of-K").
  - Có thêm bảng tương đương token chỉ khác dấu cách (`_the` ≈ `the`).
  - Tất cả xếp thành tensor P [N × L_max].
- **Trạng thái khớp.**
  - Với mỗi hypothesis b và mỗi phrase i, lưu μ = số token liên tiếp đã khớp.
  - Token tiếp theo khớp thì μ+1. Lệch thì kiểm tra xem có khớp token đầu của phrase không: có thì μ = 1, không thì μ = 0 ("first-token restart").
  - Thế khớp của hypothesis là φ_t = max_i μ.
- **Cộng vào logits (Eq. 2).** Với mỗi token ứng viên k, tính φ̂_{t+1}(k) nếu chọn k, rồi `z'[k] = z[k] + λ·Δ_k` với `Δ_k = φ̂_{t+1}(k) − φ_t`.
  - Token đi tiếp trên cây nhận +λ.
  - Token giữ nguyên trạng thái thì không đổi.
- **IPB (Immediate Prefix Boosting, §3.2.1).** Cộng đủ λ ngay từ token đầu (chuyển từ μ = 0 lên 1). Lý do: token đầu thường mang nhiều nghĩa, ví dụ họ trong tên tiếng Trung.
- **RSR (Retroactive Score Rectification, §3.2.2).**
  - Token làm thế khớp **giảm** thì bị trừ λ·φ_t (Eq. 4). Đây là cách chống "sunk cost": đã cộng cho "Alex" thì đừng ép ra cho bằng được "Alexander" khi audio là "Algorithm".
  - Paper nói rõ: RSR chỉnh logits **hiện tại**, không sửa token đã phát và không trừ chính xác xác suất chuỗi đã cộng.
- **Cài đặt (§3.3).**
  - Một `LogitsProcessor` trên vLLM, vector hoá bằng PyTorch.
  - Chỉ tính trên "vocab thưa" gồm các token có mặt trong list (thường 100–1000 token, so với hơn 150k).

**Ví dụ [ta]:** list có "Mira".

- Khi model đang phân vân giữa "mirror" và token đầu của "Mira", token đó được +λ.
- Nếu bước sau model rời khỏi cây, token rời cây bị trừ λ·φ.

**4. Setup (§4).**

- Model Phi-4-mini, frozen.
- Tập test nội bộ, chủ đề tên người (PNAME), 11 locale. Mỗi locale có khoảng 460 (de-DE) tới hơn 2100 (it-IT) entity.
- Metric:
  - WER
  - EWER (lỗi trên vùng entity)
  - Recall
  - FAR (chèn entity sai trên mỗi câu)
  - RTF
- Hai mức λ: 0,4 ("robust") và 1,0 ("aggressive").

**5. Số chính [paper].**

| Kết quả | Nguồn |
|---|---|
| Robust: EWER giảm trung bình **9% tương đối**, FAR tăng trung bình **0,3 điểm %**, WER gần như không đổi (lệch ≤ 0,1) | Table 1 |
| en-US: EWER 22.36 → 19.83, Recall 75.00 → 77.53 | Table 1 |
| Aggressive: giảm trung bình **17%** (fr-FR 29%, es-MX 28%); bảng này không báo FAR | Table 3 |
| Prompt + LOGIC tốt hơn prompt-only **5% tương đối** EWER; không có số tuyệt đối | Table 2 |
| RTF 0.0990 → 0.1018 (**+2,8%**) với list **2210 phrase**, A100 | Table 5 |

**6. Paper nói / ta diễn giải.**

- *Paper nói:* LOGIC "bổ sung" cho prompting và cho λ điều khiển được.
- *Ta diễn giải:* boost **ở mọi bước decode**, không cần biết audio đang ở đâu. Thông tin audio duy nhất là logits gốc của model. Cơ chế "có nên đẩy không" chỉ dựa vào việc token có nằm trên cây hay không.

**7. Giới hạn [ta].**

- Không có tín hiệu thời gian, nên list càng lớn thì càng nhiều tiền tố "mở cửa" ở mọi chỗ. FAR chỉ được đo trên tập PNAME nội bộ.
- Không có ablation tắt riêng IPB hoặc RSR. Không có thí nghiệm list nhiễu kiểu thêm N distractor.
- Không thử trên benchmark công khai như LibriSpeech/IS21.
- Lợi 9% tương đối là khiêm tốn so với khoảng cách prompt-oracle trên Qwen (16.08 → 7.73).

**8. Mình lấy gì.**

- Lấy nguyên **khung cài đặt**: trie trên token, trạng thái μ, vocab thưa, RSR. Đây là "mức 1" (chỉ cần logits + tokenizer) đúng như đã chốt.
- **Thêm gate thời gian:** chỉ mở cây cho từ w khi decoder đang nghe trong span(w), và λ = f(p) lấy từ scorer.
- Đó là đề xuất 1. **[ý tưởng]**

**9. So được không.** Không so số trực tiếp được: khác model, khác dữ liệu (nội bộ), khác metric (EWER).

## II.2 PTC-Bias: dò âm vị theo thời gian, cho các từ cạnh tranh, rồi sửa sau decode

**1. Thông tin.**

- Tên đầy đủ: "PTC-Bias: Phoneme-Level Temporal Competition for Bias Retrieval and Post-Decoding Correction in SpeechLLMs".
- Tác giả: Zhiqi Ai và cộng sự (Shanghai University, XJTLU).
- arXiv 2609.28727v1, ngày 23/09/2026.
- Code: github.com/aizhiqi-work/PTC-Bias.

**2. Bài toán.**

- Nhét list lớn vào prompt thì đắt và gây "lexical interference".
- Các cách chọn từ trước đây có hai nhược điểm:
  - Embedding thì thiếu định vị thời gian.
  - Dựa trên hypothesis thì bỏ sót từ không có trong lần decode đầu.
- Hơn nữa, các cách đó chấm từng từ **độc lập**, nên các từ gần đồng âm cùng lọt vào.
- Ngay cả khi từ đúng đã có trong prompt, SpeechLLM vẫn có thể viết thành từ phổ biến nghe giống. §1.

**3. Cách làm (§2).**

- **Nhánh âm vị (§2.3).**
  - Gắn vào encoder frozen một nhánh nhỏ: trộn có trọng số tất cả L lớp encoder (softmax(g)), qua một Transformer nhẹ (2 lớp), rồi một head CTC 71 lớp âm vị ARPAbet có dấu nhấn (g2pE).
  - Chỉ train phần trộn, Transformer nhẹ và head CTC. Kết quả là posterior âm vị theo từng khung.
- **PTC Retrieval (§2.4, Algorithm 1).**
  - List đổi sang âm vị và dựng thành **trie**. Quét từng khung trên posterior; mỗi từ có thể bắt đầu ở bất kỳ khung nào.
  - Với từ v_k, lấy đường CTC tốt nhất cục bộ và khoảng [a_k, b_k] (Eq. 2): `S_k = max_{a,b} (1/U_k)·Σ log P + β·log U_k`, với β hiệu chỉnh theo độ dài.
  - Giữ M = 100 sự kiện có điểm cao nhất. Các sự kiện **chồng khoảng** được coi là các cách giải thích **cạnh tranh** cho cùng một đoạn audio.
  - Xét mọi tập con không chồng nhau (gồm cả tập rỗng, tức "không từ nào"). Trọng số của một tập là `exp(Σ(S_j − δ)/τ)` (Eq. 3).
  - Xác suất biên q_k của từng sự kiện (Eq. 4) tính bằng forward–backward trên khoảng.
  - Giữ các sự kiện có q_k ≥ θ, xếp theo q_k, lấy tối đa **K = 10 từ** đưa vào prompt, kèm khoảng thời gian.
- **PTC Correction (§2.5).**
  - Sau decode, căn transcript với posterior đã cache, tìm đoạn chữ ŷ_{i:j} gần âm với v_k và chồng lên [a_k, b_k].
  - So hai chuỗi âm vị có cùng ngữ cảnh hai bên: z⁺ (thay bằng v_k) và z⁻ (giữ nguyên), trong cửa sổ mở rộng thêm 5 khung mỗi bên.
  - Lề âm học Δ_k = F(z⁺) − F(z⁻), với F là log forward CTC. Thay thế chỉ khi Δ_k vượt ngưỡng 2,0 và qua thêm kiểm tra từ vựng và biên.
  - Không cần forward lại SpeechLLM.

**Ví dụ (Fig. 1 của paper):**

- Output ban đầu là "... from plaza taylor ...".
- Retrieval đã tìm thấy "PLEAS" ở 5,8–6,8 s.
- Trong cửa sổ đó so "PLAZA" (0,03%) với "PLEAS" (99,97%), nên thay thành "pleas".

**4. Setup (§3).**

- LibriSpeech test-clean (2620 câu) / test-other (2939 câu), **theo protocol Rare5k của Le et al. 2021**, tức chính các list IS21 nhóm đang dùng. N ∈ {100, 500, 1000, 2000}.
- Hai backbone frozen:
  - Prompt-SLAM-ASR-7B
  - **Qwen3-ASR-0.6B**
- Ba front-end âm vị:

  | Front-end | Tốc độ | Tham số |
  |---|---|---|
  | DS-KWS | 25 Hz | 3,61M |
  | WavLM | 50 Hz | 317M tổng, 1,86M train |
  | AuT của Qwen3-ASR | **12,5 Hz** | 186M tổng, 1,83M train |

- Train nhánh âm vị trên LS-460 / LS-960.
- Siêu tham số: M = 100, K = 10, β = 1,1, δ = 0, τ = 0,25, θ = 0,05.

**5. Số chính [paper].**

| Kết quả | Nguồn |
|---|---|
| Prompt-SLAM-ASR-7B, N2000: B-WER 3.38 / 7.63 (clean/other), CTC-Filter 4.41 / 10.02, tức **−23,4% / −23,9%**; U-WER gần như không đổi | Table 1, §4.1 |
| **Qwen3-ASR-0.6B + front-end AuT, N2000:** Retrieval 5.08 / 9.24, thêm Correction **4.45 / 8.61** | Table 1, §4.1 |
| **Qwen3-ASR-0.6B + front-end WavLM (hàng ‡), N2000:** CTC-Filter 4.81 / 9.45, PTC 4.10 / **7.76** (−14,8% / −17,9%). Đây là số "7.76" nhóm hay trích | Table 1 |
| PER (LS-960): WavLM 1.13 / 2.27, AuT 2.06 / 5.13, DS-KWS 4.45 / 11.80 | Table 2 |
| Retrieval, test-other N2000: Recall_B@99 = **16,9** ứng viên (BR-ASR acoustic 42,2), Recall_B#50 = 99,3%, Recall_H#50 = **22,0%** (BR-ASR 69,3%) | Table 3 |
| Tốc độ CPU, câu 5,09 s: N = 2000 mất 7,61 ms (4 worker); N = 50.000 mất 391 ms (1 worker) hoặc 34,7 ms (16 worker), không tính phần tính posterior | Fig. 3 |

**6. Paper nói / ta diễn giải.**

- *Paper nói:* "retrieving the correct bias word does not always guarantee correct generation" (§4.1). Bước Correction lấy thêm 0,5–0,6 điểm B-WER.
- *Ta diễn giải:*
  - Đây là bằng chứng độc lập cho điều nhóm thấy trên Whisper: **đưa đúng từ vào prompt chưa đủ**, cần một tác động thứ hai dựa trên vị trí.
  - PTC đặt tác động thứ hai đó **sau** decode. Đề xuất 1 đặt nó **trong lúc** decode.

**7. Giới hạn [ta].**

- Correction là sửa text cứng (thay chuỗi). Model không được "nghe lại" với từ mới, nên không sửa được lỗi kéo theo (ví dụ tách từ sai làm lệch các từ sau).
- Từ đồng âm hoàn toàn chỉ xử lý được trong điều kiện hẹp (§2.5).
- Số "7.76" dùng WavLM 317M bên ngoài. Bản chỉ dùng AuT của Qwen là 8.61.
- Paper không nói normalizer hay cách chấm B-WER. Bảng 1 trích lại số của các paper khác: DB-NNLM, CB-QwenAudio.
- Hàng baseline Qwen3-ASR-0.6B (không bias) bị vỡ cột khi trích text, mình **chưa chắc** số nào thuộc điều kiện nào. Phải xem PDF gốc trước khi trích.

**8. Mình lấy gì.**

- (a) Là **đối thủ chính trên cùng benchmark IS21**, cùng họ model.
- (b) Ý "cạnh tranh giữa các từ chồng khoảng" chính là việc thứ 4 của adaptive ("từ nào thắng"). Có thể mượn cách tính biên q_k.
- (c) Front-end AuT của Qwen đủ phân biệt âm vị (PER 5,13 trên test-other), nên scorer của nhóm có thể đọc thẳng feature AuT.
- (d) Recall_B@99 = 16,9 là mốc tầng retrieval để so với `eval_retrieval.py`.

**9. So được không.**

- Gần nhất trong 4 paper: cùng list, cùng split.
- Nhưng họ dùng **0.6B**, nhóm dùng **1.7B**, và có thể khác normalizer.
- Muốn so công bằng: chạy code của họ (đã công bố) với Qwen3-ASR-1.7B và scorer IS21 của nhóm, hoặc ít nhất ghi rõ khác biệt khi đặt cạnh.

## II.3 DOA: đọc self-attention để biết SpeechLLM đang nghe tới đâu

**1. Thông tin.**

- Tên đầy đủ: "DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs".
- Tác giả: Sara Papi, Luisa Bentivogli (FBK).
- arXiv 2605.31432v1, ngày 29/05/2026.
- Code: github.com/hlt-mt/simulstream (Apache 2.0).

**2. Bài toán.**

- Dịch đồng thời (SimulST) cần một "policy" quyết định đọc thêm audio hay viết chữ.
- Ở model encoder–decoder, policy tốt nhất dựa trên cross-attention (AlignAtt, StreamAtt).
- SpeechLLM decoder-only **không có cross-attention**. Câu hỏi nghiên cứu của paper: self-attention có chứa tín hiệu căn chỉnh đủ ổn định không. §1.

**3. Cách làm (§2.1).**

- Lấy `A^(l,h) ∈ R^{T×(S+T)}` từ self-attention của decoder. T là số token chữ đã sinh, S là số token audio.
- Paper giả định **audio là phần đầu chuỗi** và cắt `Ã = A[:, :S]` thành một ma trận "proxy cross-attention" [T × S].
- Lấy trung bình trên một tập lớp L và head H, rồi với mỗi token chữ t lấy `a_t = argmax_s Ā[t, s]`. Đó là vị trí audio mà token "dựa vào".
- **Policy:** chỉ phát token nếu `a_t < S − f`, tức không dựa vào f khung audio mới nhất (còn chưa ổn định).
- **Long-form (§2.2):**
  - Lịch sử chữ chỉ giữ đoạn sau dấu câu mạnh gần nhất.
  - Lịch sử audio bỏ các khung chỉ căn với phần chữ đã bỏ, cũng dựa trên alignment này.

**Ví dụ [ta]:**

- Audio có 50 khung, f = 10.
- Token "Haus" có argmax attention ở khung 45 > 40, nên chưa phát và chờ thêm audio.
- Token "das" ở khung 30 thì phát.

**4. Setup (§3).**

- Model: Phi4-Multimodal (chọn siêu tham số và phân tích) và Qwen3-Omni (chỉ kết quả cuối).
- Tác vụ: dịch en→de, en→it. Test MCIF, dev ACL 60/60.
- Metric: BLEU, COMET, LongYAAL / LongLAAL.
- Baseline: StreamAtt trên SeamlessM4T.

**5. Số chính [paper].**

| Kết quả | Nguồn |
|---|---|
| Lấy lịch sử chữ theo dấu câu ổn định hơn giữ cố định N từ | Fig. 1 |
| **Theo lớp:** biến động mạnh hơn theo head; **6 lớp dùng riêng hỏng hẳn** (latency tệ đi khoảng 1 s hoặc hơn); **không lớp nào** vừa nhanh hơn vừa tốt hơn trung bình. **Theo head:** vài head (2, 10) tốt hơn chút ít ở cả hai trục, lợi rất nhỏ. Vì vậy chọn trung bình mọi lớp và head | Fig. 2, §4 |
| Qwen3-Omni: LongYAAL từ 400 ms tới 3,8 s, COMET cao nhất. Phi4: mỗi +10 khung f ứng với +800 ms latency | Fig. 3 |
| Cả hai SpeechLLM thắng StreamAtt (SeamlessM4T) về trade-off latency–chất lượng | Fig. 3 |

**6. Paper nói / ta diễn giải.**

- *Paper nói:* self-attention "provides sufficiently stable information to drive effective streaming decisions" (§5).
- *Ta diễn giải:*
  - Đó là bằng chứng **gián tiếp**: policy chạy tốt. Paper **không đo** độ lệch của a_t so với mốc thời gian thật (không có MFA, không có ms sai số).
  - Ngoài ra, policy chỉ cần biết "token có dựa vào phần cuối chưa ổn định không", là một câu hỏi thô hơn nhiều so với "từ này nằm ở 3,2–3,9 s".

**7. Giới hạn với Qwen3-ASR [ta].**

- (a) DOA giả định audio đứng đầu chuỗi. Trong Qwen3-ASR, **system prompt (list) đứng trước audio**, nên phải cắt theo chỉ số thật của `<|audio_pad|>`, không cắt `[:S]`.
- (b) List dài trước audio có thể hút attention ("attention sink"), làm argmax bị nhiễu. Chưa ai đo.
- (c) Kết quả "6 lớp hỏng" cho thấy chọn lớp sai thì tín hiệu vô dụng. Cần quét từng lớp × head, đúng như bước 0.
- (d) Tác vụ dịch có đảo trật tự từ. ASR thì đơn điệu, nên **có lẽ** dễ hơn. **[ta, chưa đo]**

**8. Mình lấy gì.**

- Là bằng chứng gần nhất rằng cách 1 (đọc self-attention) khả thi trên SpeechLLM decoder-only, kể cả Qwen3-Omni, tức "cha" của Qwen3-ASR.
- Bước 0 của nhóm làm phần DOA bỏ ngỏ: **đo độ chính xác thời gian** theo từng lớp × head so với MFA.

**9. So được không.** Không. Khác tác vụ và khác metric. DOA chỉ là tiền đề phương pháp.

## II.4 Qwen-Audio-3.0-ASR: hotword hai tầng P0/P1, dạy bằng SFT và RL

**1. Thông tin.**

- Tên đầy đủ: "Qwen-Audio-3.0-ASR Technical Report".
- Tác giả: Qwen-Audio ASR Team (Alibaba).
- arXiv 2609.07549v2, ngày 09/09/2026.
- Mình chưa kiểm model có công bố trọng số không.

**2. Bài toán.** Đây là report sản phẩm: ASR đa ngữ/phương ngữ cho production. Biasing là một tính năng, gồm hotword phân cấp và ngữ cảnh lịch sử. Nó không phải đóng góp phương pháp chính.

**3. Kiến trúc và cách làm biasing.**

- **Topology giữ từ Fun-ASR (§2):**
  1. encoder SenseVoice
  2. adapter Transformer 2 lớp
  3. **nhánh CTC trên encoder** cho ra hypothesis sơ bộ, dùng để **lấy ứng viên hotword** và làm alignment thô cho ngữ cảnh audio dài
  4. decoder LLM, nay là **Qwen MoE**
- **Dữ liệu SFT (§3.4):**
  - Mỗi câu kèm 2 tầng ứng viên: **P0** (chắc chắn, ưu tiên cao: tên riêng quan trọng) và **P1** (tập rộng).
  - Một LLM chia ứng viên vào P0/P1 bằng cách **so với reference**: khớp rõ thì vào P0, còn lại vào P1.
  - Cả hai tầng nằm trong instruction. Model học "ưu tiên P0 khi audio ủng hộ, dùng P1 như nguồn rộng".
  - Thêm dữ liệu tổng hợp có ngữ cảnh đúng, mâu thuẫn, nhiễu hoặc rỗng cho các từ đồng âm, để model không chép vô điều kiện.
- **RL (§3.5):**
  - Có reward riêng cho hotword: chấm việc nhận đúng P0/P1, **trọng số dương/âm bất đối xứng**.
  - Có reward phạt chèn sai từ ngữ cảnh và phạt hallucination.
- **Lúc chạy (§5.4):** "the two sets are assigned different conditioning strengths, with P0 receiving a larger weight than P1". Report **không nói** "weight" được cài thế nào: ở prompt, ở logits hay đã học sẵn. Phần mình đọc không có công thức.

**4. Setup.**

- Tập test hotword là **nội bộ**, theo loại: tên người, thuật ngữ, từ trend, thực thể AI, thương hiệu, sản phẩm.
- Trong §3.4, §3.5, §5.4, mình **không thấy** mô tả kích thước list lúc test hay số distractor.

**5. Số chính [paper, Table 3, recall %, không hotword → có hotword].**

| Loại | Doubao-ASR | Fun-ASR-Flash | Qwen-Audio-3.0-ASR |
|---|---|---|---|
| Tên người (P0) | 32.27 → 82.64 | 37.33 → 95.26 | 62.12 → **99.43** |
| Thuật ngữ (P1) | 84.36 → 93.04 | 86.02 → 96.73 | 88.46 → 95.31 |
| Từ trend (P1) | 51.24 → 89.48 | 53.92 → 87.00 | 74.00 → 88.34 |

Report tự nhận xét: P1 tăng ít hơn P0, khớp với việc P1 được gán trọng số yếu hơn.

**6. Paper nói / ta diễn giải.**

- *Paper nói:* cơ chế hai tầng "improves long-tail hotword recall without forcing acoustically unsupported candidates into the transcript" (§3.4). Họ chỉ báo **recall**, không báo chèn sai trên câu không chứa hotword.
- *Ta diễn giải:* đây là phiên bản **rời rạc 2 mức** của "mạnh bao nhiêu" (việc thứ 2 của adaptive). Nó còn cho thấy nhóm Qwen chọn **dạy model** (SFT + RL) thay vì can thiệp lúc decode.

**7. Giới hạn [ta].**

- Cần train lại cả model. Ngoài ràng buộc "frozen, cắm mọi SpeechLLM" của nhóm.
- P0/P1 lúc train dựa vào reference, lúc test dựa vào CTC hypothesis. Báo cáo không nói P0 lúc test chính xác cỡ nào.
- Số nội bộ, không có list nhiễu kiểu IS21.

**8. Mình lấy gì.**

- (a) Ủng hộ ý **độ mạnh theo độ tin cậy**. Scorer của nhóm cho p liên tục, nên thay P0/P1 bằng λ = f(p) là bước tổng quát hoá tự nhiên.
- (b) Cũng dùng **nhánh CTC trên encoder** để lấy ứng viên và alignment thô. Cùng tinh thần với scorer + span của nhóm, khác ở chỗ họ dùng hypothesis chữ còn nhóm chấm từ theo audio.
- (c) Khi viết paper, nêu đây là cách "train-in" của chính họ Qwen, để làm rõ vì sao nhóm chọn "plug-in".

**9. So được không.** Không. Model khác, dữ liệu nội bộ, metric là recall.

---

# Phần III. Ghép lại

## III.1 Bảng so 4 paper

| | LOGIC | PTC-Bias | DOA | Qwen-Audio-3.0 |
|---|---|---|---|---|
| Model | Phi-4-mini | Prompt-SLAM-7B, Qwen3-ASR-0.6B | Phi4-MM, Qwen3-Omni | Qwen MoE + SenseVoice |
| Train thêm? | không | nhánh âm vị khoảng 1,8M | không | cả model (SFT + RL) |
| Tác động ở đâu | logits, mọi bước | prompt (top-10) + sửa text sau decode | policy đọc/viết | prompt P0/P1 (đã học) |
| Có vị trí thời gian? | **không** | **có** ([a, b] từ CTC âm vị) | **có** (từ self-attention) | thô (CTC) |
| Dùng vị trí lúc decode? | — | **không** (dùng sau) | có, nhưng cho dịch, không cho bias | không rõ |
| Độ mạnh thích ứng | λ cố định | top-K, ngưỡng q_k | — | 2 mức P0 > P1 |
| Cạnh tranh từ gần âm | không | **có** (biên q_k, lề Δ) | — | không |
| List lớn / nhiễu | ~2200, không có distractor | IS21 N ≤ 2000 | — | không rõ |
| Benchmark công khai | không | **LibriSpeech IS21** | MCIF | không |

## III.2 Chỗ trống

Mỗi paper có một mảnh:

- **PTC:** có vị trí, nhưng chỉ dùng **sau** decode.
- **LOGIC:** tác động **trong** decode, nhưng mù vị trí.
- **DOA:** biết decoder đang nghe đâu, nhưng không dùng cho bias, và chưa đo độ chính xác thời gian.
- **Qwen-Audio-3.0:** độ mạnh theo độ tin cậy, nhưng phải train lại.

**[ta]:** chưa paper nào trong nhóm này ghép đủ bốn mảnh trên một SpeechLLM frozen:

1. scorer cho (p, span)
2. biết decoder đang nghe đâu
3. boost logits chỉ trong span, mạnh theo p
4. chỉ một từ thắng trên mỗi span

Đây là đề xuất 1. Kết luận dựa trên khoảng 15 lượt tra trong một buổi, **chưa đủ để khẳng định "chưa ai làm"**.

## III.3 Ba câu phải trả lời bằng thí nghiệm (theo thứ tự)

1. **Bước 0:** trên Qwen3-ASR-1.7B, có lớp × head nào mà argmax attention từ token chữ tới 130 vector audio khớp mốc MFA không, và sai số bao nhiêu? (DOA bỏ ngỏ câu này.)
2. **Bước 1:** với span oracle (MFA), boost có gate đạt B-WER bao nhiêu so với prompt oracle 7.73 và N2000 16.45?
3. **Bước 2:** thay oracle bằng scorer thật. So với lọc p ≥ 0.5 (9.71), PTC-Bias (8.61 bản AuT / 7.76 bản WavLM, trên **0.6B**, phải chạy lại trên 1.7B) và Hou25 top-50.

## III.4 Nguồn

| Nguồn | Mức đọc | Dùng cho |
|---|---|---|
| `qwen_asr` 0.0.6 `modeling_qwen3_asr.py`, `config.json` Qwen3-ASR-1.7B | đọc trực tiếp | Phần I |
| Qwen3-ASR Technical Report, arXiv 2601.21337v2, §2.1–2.2, §3 | toàn văn | I.2, I.5, I.6 |
| LOGIC, arXiv 2601.15397v4, §3, Table 1–5 | toàn văn | II.1 |
| PTC-Bias, arXiv 2609.28727v1, §2–4, Table 1–3, Fig. 3 | toàn văn (Table 1 trích từ text bị vỡ cột; các số dùng ở đây đã đối chiếu với câu chữ §4.1 và phép tính % tương đối) | II.2 |
| DOA, arXiv 2605.31432v1, §2–5, Fig. 1–3 | toàn văn (phụ lục số liệu không trích được) | II.3 |
| Qwen-Audio-3.0-ASR, arXiv 2609.07549v2, §2, §3.4–3.5, §5.4, Table 3 | đọc các phần liên quan biasing; chưa đọc hết phần benchmark đa ngữ | II.4 |
| Huber & Waibel, arXiv 2608.05759, Table I–III | toàn văn (phiên trước) | I.7 |
| `bench/results.md` §3, §5 | số của nhóm | I.7 |
