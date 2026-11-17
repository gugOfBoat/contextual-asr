# Scorer cắm vào ASR ở đâu: prompt, lúc decode, hay sau decode (17/11/2026)

> Câu hỏi của user (17/11): đi sâu vào **scorer + SpeechLLM + AED**. Can thiệp bằng prompt, còn novelty nằm ở **lúc decode**, **sau khi có kết quả**, hoặc cả ba. Hướng đọc self-attention bên trong Qwen để biết decoder đang nghe giây nào: **để sau** (không khả quan).
> Nhãn: **[paper]** = paper nói, có số mục/bảng · **[abs]** = mới đọc abstract · **[đo]** = nhóm đã chạy · **[ta]** = mình diễn giải · **[ý tưởng]** = chưa chạy.
> Loại setup: **A** = theo nguồn · **B** = thông lệ · **C** = nhóm tự thiết kế.

---

## 0. Tóm tắt 5 dòng

1. Hầu hết paper chỉ dùng scorer/retriever để **chọn từ rồi nhét vào prompt** (Hou25, CB-Whisper, CLAR, GLCLAP, BR-ASR). Không bài nào dùng **vị trí** của từ ở bước sau.
2. Ở lúc decode, các bài cộng điểm theo cây tiền tố: LOGIC, trie cho Whisper 2508.17796, Huang & Wang 2510.23849. **Không bài nào có gate theo thời gian.** Riêng Huang & Wang đã dùng điểm scorer (có lớp "không từ nào") để lọc list và đặt mức thưởng: phải trích, và không được claim phần đó là mới.
3. Ở sau decode: PTC-Bias (có Qwen3-ASR) và CTC-WS/DVPA (CTC) đã **thay chữ theo đoạn thời gian**. Novelty ở bước sau decode chỉ còn khi **xác minh lại bằng chính ASR**, chứ không chỉ thay theo điểm của scorer.
4. Đề xuất chính [ý tưởng]: **Đề → Định vị → Kiểm (Propose–Localize–Verify)**. Scorer đề xuất (w, p, span). Lần decode 1 dùng prompt rất ngắn. Forced aligner căn chữ đã viết với thời gian. Chỉ tại span đáng ngờ, **decode lại cục bộ** với prompt/boost của đúng từ w. Cuối cùng **ASR tự chấm** bản cũ và bản sửa bằng teacher forcing.
5. Cách này chỉ cần **prompt, logits + tokenizer, và một forced aligner ngoài**. Không đọc attention, không train ASR, cắm được cả Whisper (AED) lẫn Qwen3-ASR (SpeechLLM). Khớp chốt 11/11 trong `GOAL.md`.

---

## 1. Khung: 3 vị trí × 2 họ model

```
             audio ──► [SCORER] ──► (w, p, t_đầu, t_cuối) cho từng từ trong list
                                │
          ┌─────────────────────┼──────────────────────────┐
          ▼                     ▼                          ▼
   (P) PROMPT             (D) LÚC DECODE             (S) SAU DECODE
   chọn từ nào,           đẩy logits token của w     sửa hypothesis tại span,
   bao nhiêu từ           (có/không gate thời gian)  có/không kiểm lại bằng ASR
```

Hai họ model khác nhau ở những điểm ảnh hưởng tới cách cắm:

| | AED (Whisper) | SpeechLLM (Qwen3-ASR) |
|---|---|---|
| Prompt nghĩa là gì | Ô `<\|startofprev\|>` = **chữ của đoạn trước**, không phải list; tối đa 224 token. B-Whisper 2502.11572 nói đây là lệch format khi nhét list [abs] | System prompt **được dạy ở SFT** để chứa context (report 2601.21337) [paper] |
| List dài thì sao | Prompt bị cắt (ta cắt 223 token cuối) | Không cắt, nhưng N2000 làm B-WER **tệ hơn không list**: 16.45 so với 16.08 [đo] |
| Có thời gian của chữ không | Có token timestamp, có word timestamp (DTW trên cross-attention) | Có **Qwen3-ForcedAligner-0.6B** chính chủ (RTF ≈ 0.001 theo report) [paper] |
| Logits + tokenizer | Có | Có |
| Teacher forcing để chấm một câu cho trước | Có, rẻ | Có, rẻ (1 lượt prefill) |

**Ta:** dòng 3–5 giống nhau ở cả hai họ, nên phần D và S có thể viết **một lần cho cả hai** nếu không đọc attention. Chỉ phần P phải xử lý riêng: Whisper không được dạy đọc list.

---

## 2. Paper theo từng vị trí

### 2.1 (P) Prompt: scorer/retriever chọn từ rồi nhét vào prompt

| Paper | ASR | Scorer/retriever | Ra gì | Dùng vị trí? | Số chính |
|---|---|---|---|---|---|
| **Hou25** (Interspeech 2025) | Whisper, frozen | TTS audio + encoder + CNN đọc ma trận tương đồng | top-50 cố định | Không (vệt chéo có thời gian nhưng bị pool) | test-other N2000 prompt: B-WER 9.69 [đo, tái hiện] |
| **CB-Whisper** 2309.09552 (LREC-COLING 2024) | Whisper, frozen | TTS + Whisper encoder + KWS | từ phát hiện được → **prompt dạng nói** (bắt chước chữ đoạn trước) | Không [abs] | recall +20% trung bình [abs] |
| **Massive OV-KWS** 2606.11279 (Priberam) | Whisper large-v2 (WhisperX) | KWS: Whisper encoder + ResNet-50 trên ma trận cosine, nén embedding 128× | từ phát hiện → `hotwords` của WhisperX (prompt) | Không (§2.1, §2.5) [paper] | Internal: MER 29.5 → **32.4** (tệ hơn không bias, §4.2 do từ nhiễu) [paper] |
| **KG-Whisper** 2406.02649 (Interspeech 2024) | Whisper, **fine-tune** decoder hoặc học prefix | KWS trên encoder | sinh prompt | không rõ [abs] | WER −5.1% ở ngôn ngữ chưa thấy [abs] |
| **CLAR** 2603.25460 (nộp Interspeech 2026) | Step-Audio2-mini, **LoRA** (§3.2) | dual encoder + CIF, khớp theo cửa sổ cùng độ dài (Eq. 5–6) | top-K vào prompt (§2.4) | Định vị chỉ để **xếp hạng**, không dùng lúc decode [paper] | AISHELL-1-NE test: B-WER 12.92 → 2.78 (Bảng 3) [paper] |
| **BR-ASR** 2505.19179 (Interspeech 2025) | SpeechLLM **train** | contrastive speech–bias | top-k vào prompt | Không | test-other N2000 B-WER 7.1 [abs] |
| **Kong et al.** 2512.21828 | LLM-ASR **fine-tune + GRPO** | GLCLAP mở rộng | top-k vào prompt | Không rõ [abs] | Không có số trong abstract |
| **Huang & Wang** 2510.23849 (U. Iowa) | AED + CTC tự train, 116M | decoder phụ chấm log P(phrase \| X)/L, **có phrase rỗng p₀ = "không bias"** (Eq. 1) | **lọc** (giữ nếu s_i − s₀ + tol ≥ 0) + **mức thưởng** cho shallow fusion (Eq. 3–4) | Không (chấm trên cả encoder output) [paper] | IS21 test-other N2000: WER 5.0, B-WER ≈ 9.3–9.6 (Bảng IV) [paper] |

**Rút ra [ta]:**
- "Scorer chọn ít từ → prompt" đã đông người làm. "Lớp không từ nào + chỉ giữ vài từ" **đã có** ở Huang & Wang, và họ chạy đúng IS21 N100–2000. Phần P của ta **không còn là điểm mới**; nó chỉ là tầng đầu bắt buộc.
- Massive OV-KWS cho bằng chứng ngoài nhóm: trên dữ liệu thật, **prompt từ phát hiện sai còn làm tệ hơn không bias**. Điều này khớp quan sát của ta: N2000 trên Qwen tệ hơn không list.

### 2.2 (D) Lúc decode: cộng điểm cho token của từ trong list

| Paper | ASR | Cách | Gate thời gian? | Số chính |
|---|---|---|---|---|
| **LOGIC** 2601.15397 (Microsoft) | Phi-4-mini SpeechLLM, frozen | cây tiền tố trên token, z' = z + λ·Δ, trừ lại khi rời nhánh (Eq. 2, 4) | Không | EWER −9% tương đối, FAR +0.3 [paper] |
| **Trie Whisper** 2508.17796 (APSIPA 2025) | Whisper large-v3, frozen, beam 10 | trie + biến thể phát âm từ TTS, +1 mỗi token khớp | Không (§3.2) [paper] | IS21 **N1000** test-other: B-WER 17.55 → 9.90; U-WER 3.39 → 3.14 (Bảng 2) [paper] |
| **Huang & Wang** 2510.23849 | AED tự train | KMP shallow fusion, mức thưởng lấy từ điểm scorer | Không | như trên |
| **TCPGen cho Whisper** 2306.01942 | Whisper + module train | pointer generator trên cây | Không | — |
| **Gated contextual adapter** (Amazon) | Transducer | gate học được **theo từng frame**: có cần bias không | Có, nhưng học bên trong model, cần train [abs] | — |
| **Ta, Whisper** | turbo, frozen | boost có gate theo giây decoder đang nghe (alignment heads) | **Có** | oracle + gate MFA 7.90; scorer 2 nhánh + gate **7.53**; Hou25 prompt 9.69 [đo] |

**Rút ra [ta]:**
- Gate theo thời gian lúc decode trên model frozen **vẫn chưa thấy ai làm** sau khoảng 20 lượt tra.
- Bản của ta trên Whisper cần alignment heads, mà SpeechLLM không có. User đã chọn bỏ đường đọc attention của Qwen.
- Vì vậy ở mục 3 cần một cách gate **không cần biết decoder đang nghe đâu ở từng bước**.

### 2.3 (S) Sau decode: sửa hypothesis

| Paper | ASR | Định vị? | Ai quyết định thay | Số chính |
|---|---|---|---|---|
| **PTC-Bias** 2609.28727 | **Qwen3-ASR-0.6B**, Prompt-SLAM | Có: sự kiện CTC âm vị có đoạn frame, các sự kiện chồng nhau tranh nhau (Eq. 3–4) | margin Δ_k > 2.0 trên điểm CTC âm vị | Qwen 0.6B N2000 test-other: retrieval 9.24 → correction **8.61** (front-end AuT), **7.76** (WavLM) [paper] |
| **CTC-WS** 2406.07096 | CTC/Transducer | Có: ghi đè đoạn frame trong greedy output | điểm spotter | — |
| **DVPA** 2505.23077 | CTC (WeNet), module train | Có: đỉnh CTC của bias token đánh dấu cửa sổ (§2.3) | xác suất đường khớp > ngưỡng × độ dài | IS21 test-other **N100**: B-WER 34.3 → 9.59 (Bảng 2) [paper] |
| **PMF-CEC** 2506.11064 (TASLP 2025) | sửa text của nhiều ASR, có Whisper, SLAM-ASR | Không (chỉ trên text + âm vị) | bộ phát hiện K/D/C + ngưỡng giữ nguyên 0.5 (§III-H) | Whisper medium.en test-clean B-WER 15.08 → 9.61 (Bảng VIII) [paper] |
| **Generative annotation** 2508.20700 (Huawei) | sửa output Whisper, model sửa **được train** | Không: chỉ nói đoạn audio **có** chứa entity | model sửa đọc audio + ứng viên + transcript (§2.3) | AISHELL NE recall 70.85 → 87.31 (Bảng 2) [paper] |
| **LLM rescoring** 2309.00723 | N-best + prompt có list | Không | LLM text | [abs] |

**Rút ra [ta]:**
- Sửa theo **đoạn thời gian** đã có (PTC-Bias, CTC-WS, DVPA). PTC-Bias còn làm đúng trên Qwen3-ASR và đúng IS21. Nếu bản S của ta chỉ là "scorer nói ở đây có w thì thay", nó **chỉ là PTC-Bias đổi scorer**.
- Điểm chưa ai làm: **cho chính ASR kiểm lại** bản sửa. Teacher-force hai câu (cũ và đã thay w) trên ASR frozen, so log-prob, kết hợp với p của scorer.
  - Kang et al. (Interspeech 2024) dùng teacher forcing để chấm N-best, nhưng không gắn với bias list hay vị trí [paper, mới đọc lướt].
  - Lý do cần: sửa thuần text thì "không đo được bản sửa có giống audio không" (2506.06117) [abs]. PTC-Bias đo bằng nhánh CTC riêng, còn ta đo bằng **chính model ASR**, vốn đã nghe tốt hơn.

---

## 3. Đề xuất: Đề → Định vị → Kiểm (Propose–Localize–Verify)

### 3.1 Sơ đồ

```
audio ─► SCORER ─► (w, p, span) cho list N2000
             │
   (P)  p ≥ θ_cao (0–3 từ) ─► prompt ngắn ─► ASR lần 1 ─► hyp₁
             │                                   │
             │                         forced aligner (Whisper: word ts / Qwen3-ForcedAligner)
             │                                   ▼
   (L)  với mỗi (w, span) có p ≥ θ_thấp:   chữ của hyp₁ nằm trong span = h
             │     h == w ? → bỏ qua
             │     h ≠ w  → đoạn đáng ngờ
             ▼
   (D)  DECODE LẠI CỤC BỘ: teacher-force hyp₁ tới trước span,
        rồi sinh tiếp vài token với boost cho w (± prompt có w),
        dừng khi qua hết span ─► ứng viên hyp₂
             ▼
   (S)  KIỂM: Δ = logP_ASR(hyp₂) − logP_ASR(hyp₁) + α·logit(p)
        Δ > 0 → nhận hyp₂, ngược lại giữ hyp₁
```

### 3.2 Vì sao cách này lách được vấn đề "không biết decoder đang nghe đâu"

- Trên Whisper ta gate ở **từng bước decode**, nên mới cần alignment heads.
- Ở đây ta **decode một lần trước**, căn chữ với thời gian **sau khi đã có chữ**, rồi mới quay lại đúng chỗ. Thời gian đến từ forced aligner ngoài, không đến từ attention bên trong. GOAL 11/11 đã ghi đúng ý này: "giây decoder đang nghe được suy ra bằng cách căn chữ đã viết với audio".
- Boost chỉ bật trong vài token sau tiền tố của span, nên **gate là gate cứng theo vị trí trong chữ**, không phải theo giây ở từng bước. Không cần alignment heads.

### 3.3 Mỗi khối mới ở đâu so với paper

| Khối | Gần nhất | Khác ở đâu | Mức mới [ta] |
|---|---|---|---|
| P: prompt 0–3 từ theo p | Huang & Wang (lọc theo p₀), CLAR, Hou25 | dùng cho cả AED lẫn SpeechLLM frozen | **thấp**, chỉ là nền |
| L: căn hyp với thời gian rồi so với span của scorer | PTC-Bias (sự kiện CTC), CTC-WS | span lấy từ scorer cắm ngoài, chữ lấy từ ASR bất kỳ | trung bình |
| D: decode lại cục bộ, chỉ trong span, có boost | LOGIC / trie (boost cả câu); 3 lượt tra không thấy ai "decode lại một cửa sổ" | gate bằng **vị trí trong chữ** sau khi căn, không cần attention | **cao** |
| S: ASR tự chấm bản sửa (teacher forcing) + p | Kang 2024 (teacher forcing chấm N-best), PTC (margin CTC) | kiểm bằng chính ASR, ứng viên do scorer + vị trí sinh ra | **cao** |
| Cả 4 khối, một cơ chế cho Whisper và Qwen3-ASR | không thấy | — | đóng góp hệ thống |

**Cần nói thật:**
- Nếu S làm hết việc (D không thêm gì), câu chuyện thu về "PTC-Bias + verifier". Vẫn có điểm mới, nhưng hẹp hơn.
- Novelty check mới dựa trên khoảng 20 lượt tra web, **chưa phải tra hệ thống**. Cần tra lại "local re-decoding" và "self-verification ASR biasing" trước khi viết paper.

### 3.4 Vì sao kỳ vọng cách này thắng [ta, chưa đo]

- **Không chèn bừa** khi `missing_N`. Bước L chỉ mở khi scorer có span; bước S còn chặn thêm nếu ASR thấy bản sửa không giống audio. Hiện oracle prompt trên test-other chèn 419 bias word [đo].
- **Sửa được lỗi "nghe đúng nhưng viết chính tả quen hơn"**, loại lỗi F12 cho là chiếm 3.1/5.6 điểm B-WER [đo, Whisper]. Prompt không sửa được loại này; boost cục bộ thì sửa được.
- **Không tốn prompt dài.** Cả hai model đều tệ đi khi list dài (Qwen N2000 16.45 > none 16.08 [đo]).

### 3.5 Rủi ro

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Forced aligner căn sai quanh từ hiếm (vì chữ sai) | span của hyp lệch span scorer > 0.2 s | cửa sổ ± 1 từ; đo trước bằng MFA |
| Teacher-force log-prob ưu tiên chính tả quen (đúng thứ cần sửa) | S từ chối cả bản sửa đúng | α·logit(p) bù lại; học α trên dev |
| Chi phí: 1 lượt decode + aligner + vài lượt cục bộ | RTF tăng | chỉ mở ở span h ≠ w; đo RTF |
| Whisper: decode lại giữa chừng phải giữ cả token timestamp | lệch format | dùng chế độ không timestamp, teacher-force tiền tố chữ |

---

## 4. Thử khả thi đề nghị (cần duyệt code + GPU)

Thứ tự sắp theo điểm dừng sớm nhất. Đều không train.

| # | Thí nghiệm | Model | Cần | Câu trả lời | Dừng nếu |
|---|---|---|---|---|---|
| X0 | Forced aligner trên **hyp lần 1** (không phải ref): span của chữ sai lệch bao nhiêu so với MFA của ref | Qwen3-ASR + Qwen3-ForcedAligner; Whisper word ts | `align.tsv` test-other có sẵn trên server | Bước L có đủ chính xác không | lệch trung vị > 0.2 s |
| X1 | **Trần** (oracle): từ đúng + span MFA → L + D, không S | Whisper turbo, Qwen3-ASR-1.7B | `gate_boost.py` sửa nhỏ | D cục bộ có tiến gần oracle prompt không (Qwen 7.73, Whisper 7.90) | không thắng prompt top-50 |
| X2 | Chỉ S: ứng viên oracle và ứng viên nhiễu (`missing_N2000`) → ASR chấm | cả hai | teacher forcing | ASR có phân biệt bản sửa đúng / sai không: AUC | AUC < 0.8 |
| X3 | Đầy đủ P + L + D + S với scorer thật (pdual) | cả hai | output scorer test-other | so Hou25 9.69, PTC 8.61/7.76, pxattn 9.71, gate Whisper 7.53 | — |

Loại setup:
- θ_cao, θ_thấp, α: **C**, chọn trên dev.
- IS21 N2000, B-WER bằng scorer chính thức: **A**.
- Greedy: **B**.

---

## 5. Câu hỏi để chốt với user

1. Đề xuất chính là **P + L + D + S**, hay tách thành hai paper: D cục bộ, và S tự kiểm?
2. Chạy Whisper trước (đã có code, có số mốc) rồi mới sang Qwen3-ASR, hay chạy song song?
3. GPU: lúc 16/11 cả 4 card đều đã dùng khoảng 19/24 GB. Cần chọn card và giờ chạy.

---

## Nguồn

**Đọc toàn văn (HTML/PDF):**
- 2510.23849 Huang & Wang: §III, Bảng I, II, IV
- 2603.25460 CLAR: §2.2–2.4, §3.1–3.2, Bảng 3–4
- 2606.11279 Massive OV-KWS: §2.1–2.5, Bảng 2–3
- 2506.11064 PMF-CEC: §III, Bảng VI, VIII–X
- 2508.20700 Generative annotation: §2, Bảng 2–3, 6
- 2505.23077 DVPA: §2.3, Bảng 2
- 2509.04357 PARCO: §III, Bảng II, IV
- 2508.17796 Trie Whisper: §3, Bảng 2
- 2609.28727 PTC-Bias, 2601.15397 LOGIC, 2601.21337 report Qwen3-ASR (xem `phan-tich-4-paper-va-qwen3asr.md`)

**Mới đọc abstract / tóm tắt:**
- 2309.09552 CB-Whisper
- 2406.02649 KG-Whisper
- 2505.19179 BR-ASR
- 2512.21828 Kong et al.
- 2502.11572 B-Whisper
- Spot and Merge (Interspeech 2025, Agrawal et al.)
- Gated contextual adapters (Amazon)
- Kang et al. Interspeech 2024 (rescoring + rewriting)
- 2506.06117 phonetic rescoring
- 2310.04657 spike-triggered
- 2406.07096 CTC-WS

Số [đo] của nhóm: `bench/results.md`, `scorer/analysis/REPORT-1115.md`, `GOAL.md` mục 6.
