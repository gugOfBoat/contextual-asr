# Đề xuất: gap chung của Hou25 và PTC-Bias → scorer của nhóm (bản nháp 11/11/2026)

> Thứ tự đọc: **gap → giả định → scorer suy ra từ gap → kế hoạch chứng minh**. So với từng khối của 2 paper chỉ làm ở cuối (mục 5).
>
> Ký hiệu nguồn:
> - **[Hou25]** = Hou et al., Interspeech 2025, đọc toàn văn.
> - **[PTC]** = PTC-Bias, arXiv 2609.28727v1, đọc toàn văn.
> - **[Hoàng]** = repo `itshoang2024/hou25_interspeech`, NOTES F1–F16.
> - **[ta]** = nhóm đã đo.
> - **[diễn giải]** = suy luận của nhóm, chưa phải số đo.
> - **[ý tưởng]** = chưa chạy.

---

## 0. Sau khi tra tài liệu (11/11): quan sát → gap → điểm sáng → giả thuyết

> **Mục tiêu đã chốt (user, 11/11):** *một scorer cắm vào mọi SpeechLLM, train một lần, đẩy từ đúng lúc mà không cần biết bên trong model.*
>
> - Scorer có encoder riêng, đóng băng; không dùng encoder của ASR.
> - Từ ASR chỉ cần: logits và tokenizer (mức 1), hoặc text đầu ra (mức 2). **Không dùng prompt** làm phương pháp chính (chốt 13/11, xem §0.3b).
> - Biết decoder đang nghe giây nào bằng cách **căn chữ đã viết với audio**, không dùng attention bên trong model.

> Mục này thay thế cách định vị novelty ở các mục sau. Phạm vi: khoảng 10 lượt tìm kiếm và đọc HTML của 3 paper. **Chưa đủ để khẳng định "chưa ai làm"**, phải tra tiếp trước khi viết paper.

### 0.1 Ai đã làm gì với "thời gian của bias word"

| Paper | Có vị trí từ? | Dùng vị trí để làm gì | Loại ASR | Phải train? | Mức đã đọc |
|---|---|---|---|---|---|
| CTC-WS, Andrusenko et al., Interspeech 2024 (arXiv 2406.07096) | Có, khoảng frame | So với kết quả CTC greedy, thay nếu điểm cao hơn | CTC, transducer | Không | tóm tắt |
| Spike-triggered CB, Huang et al., ASRU 2023 (arXiv 2310.04657) | Có, các frame "spike" của CTC | **Chỉ bias tại các frame đó** | AED Conformer có nhánh CTC, tiếng Trung | Có: train mô-đun bias, ASR gốc đóng băng | HTML |
| WCTC-Biasing, Nakagome & Hentschel, Interspeech 2025 (arXiv 2506.01263) | Có, dò ở lớp giữa encoder | Bias các lớp encoder phía sau | CTC | Không | tóm tắt |
| PTC-Bias (arXiv 2609.28727) | Có, khoảng [a, b] | Lọc list, rồi sửa **sau khi decode xong** | SpeechLLM | Chỉ train đầu âm vị | toàn văn |
| CB-Whisper (2309.09552), KG-Whisper (2406.02649), CTC-assisted LLM-ASR (2411.06437) | Bộ dò từ khoá có thể có vị trí | Chỉ đưa **chữ** vào prompt, bỏ vị trí | Whisper, LLM-ASR | KG-Whisper fine-tune; còn lại không | abstract |
| Novitasari et al., IBM (arXiv 2604.12398) | "Vị trí" là nhãn trên **chữ** của transcript, dùng làm loss phụ (§2.3) | **Bỏ khi chạy** (§2.3) | Granite-Speech 8B, list 200 | Có | HTML |
| Hou25; BR-ASR (2505.19179) | Không | Top-k → prompt | Whisper, SpeechLLM | Không | toàn văn / abstract |
| PAC (2509.12647) | Không | Train LLM phân biệt đồng âm bằng RL | LLM-ASR | Có | abstract |

Một paper TCPGen-cho-Whisper năm 2025 (kết quả tìm kiếm, **chưa đọc**, 2509.09197 hoặc 2508.17796) nêu lý do: Whisper không sinh dự đoán theo frame, nên các cách bias theo frame khó áp dụng.

### 0.2 Quan sát

- **Q1 [tài liệu]:** Bias **đúng thời điểm** đã có và hiệu quả (Spike-triggered, CTC-WS, WCTC), nhưng chỉ cho ASR có đầu ra **theo frame** (CTC, transducer, AED có CTC). Với decoder tự hồi quy kiểu Whisper hoặc SpeechLLM, mọi phương pháp đã tra hoặc đưa bằng chứng vào dưới dạng **chữ** (prompt), hoặc sửa **sau** khi viết. Vị trí, nếu có, đều bị bỏ trước khi decoder dùng.
- **Q2 [ta]**, `bench/results.md`, test-other, N2000:

  | Hệ | WER | B-WER |
  |---|---|---|
  | Prompt Hou25 | 3.39 | 9.69 |
  | Từ đúng + gate theo vị trí MFA | 3.19 | **7.90** (U-WER giữ nguyên 2.63) |
  | Bỏ gate | — | 22.26 |
  | Vị trí đoán bằng S1 | — | 8.42 |
  | List thật (top-50) + gate S1 | 15.76 | 14.45 |

  Ngay cả bản oracle có gate cũng chèn sai nhiều hơn prompt: 115 so với 39.

  Các dòng gate ở bảng trên chạy **không prompt** (`bench/out/run1108_logs/final.sh`). So với không context (R4, `related-work/novelty-gate.md` §4): lỗi nghe giống 7.23 → 2.37, đồng âm 3.01 → 0.60, **ngang hoặc hơn prompt top-50** (2.88 / 0.80). Phần chèn thêm chủ yếu là chèn lặp từ hiếm (+2.02).
- **Q3 [Hoàng F12]:** 95% lỗi còn lại khi dùng prompt là từ **đã có trong prompt**; 64–70% là gần âm hoặc đồng âm.

### 0.3 Research gap

**Bias đồng bộ thời gian cho decoder tự hồi quy, đóng băng, khi list lớn và nhiễu.**

Trong phạm vi đã tra, chưa thấy ai dùng vị trí của bias word để can thiệp *đúng lúc* SpeechLLM hay Whisper đang sinh token mà không phải train lại ASR. Cản trở kỹ thuật: decoder không đồng bộ với frame, nên không biết lúc nào nó đang "nghe" đoạn có từ.

### 0.3b Gap thứ hai: adapter dựa trên prompt phụ thuộc vào từng model (chốt 13/11)

| Hiện tượng | Nguồn |
|---|---|
| Whisper chỉ giữ 223 token cuối của prompt, phần đầu bị cắt | Hoàng F1 |
| Đổi thứ tự từ là đổi B-WER; bảng tăng/giảm dần của Hou25 thực chất là do bị cắt | Hoàng F1, F13; Hou25 Table 3 |
| Prompt kiểu "nói" khác prompt liệt kê | Hou25 Table 2 |
| Prompt 1–2 từ làm Whisper lặp chữ, ảo giác (oracle prompt WER 14.29) | Hoàng F6; `bench/results.md` |
| Qwen3-ASR ngược lại: prompt ngắn tốt nhất | `bench/results.md` §3 |

→ Cách đưa từ vào prompt (thứ tự, độ dài, cách viết, có dùng hay không) phải chỉnh lại cho từng model, nên **không** "cắm vào mọi SpeechLLM" được.

**Quyết định:** adapter chính **không dùng prompt**:
- mức 1 đẩy token của từ khi decoder ở đúng đoạn, chỉ cần logits + tokenizer;
- mức 2 sửa text sau khi viết theo span, chỉ cần text.

Đổi model chỉ đổi tokenizer. Prompt chỉ còn là đường lui cho model đóng gọi qua API.

**Hệ quả cho scorer:** không còn prompt "che" giúp, nên scorer phải tự trả lời đúng ba câu: từ nào thật sự có (p đáng tin), nằm ở đâu (span), đoạn này có "không từ nào" không. Đó là S1–S3 ở §0.6. Đẩy nhầm một từ nhiễu là chèn sai ngay (list thật top-50 + gate: WER 15.76).

### 0.4 Điểm sáng

1. **Có cầu nối, không cần train.** Alignment heads của Whisper cho biết decoder đang nghe giây nào (Simul-Whisper §2.1). Thử 8 câu: lệch so với span 0.0–0.06 s.
2. **Trần đo được và đáng làm.** Biết đúng từ và đúng vị trí thì B-WER từ 9.69 xuống 7.90, U-WER không đổi. Khi đó nhóm lỗi "đã có trong prompt mà vẫn viết sai" (Q3) có kênh để sửa.
3. **Chỗ hỏng chỉ ra scorer phải làm gì.** Với list thật, từ nhiễu cũng "có vị trí", nên bị boost bừa: WER 15.76. Scorer cần trả lời *"ở đoạn này có từ nào trong list không, nếu có thì từ nào"*, chứ không chỉ "top-50 cả câu". Đây là lý do scorer **bắt buộc** phải có thời gian và lựa chọn "không từ nào".

→ **Đóng góp dự kiến:**
- (i) Gate lúc decode cho SpeechLLM đóng băng, **không cần biết bên trong model**: vị trí decoder suy ra từ chữ đã viết (G6a).
- (ii) Scorer thiết kế cho gate: bằng chứng theo thời gian, có "không từ nào", các từ gần âm đấu nhau tại cùng chỗ.
- (iii) Phân tích: vị trí cần chính xác đến đâu.

### 0.5 Giả thuyết và thí nghiệm

| # | Giả thuyết | Thí nghiệm | Thắng khi | Thua thì |
|---|---|---|---|---|
| G1 | Gate chịu được vị trí lệch tới một mức, và mức đó đo được | Từ đúng + span MFA dịch ngẫu nhiên ±0 / 50 / 100 / 200 / 500 ms, và cả câu. Không train | Đường B-WER có "gối": dưới gối thì gần 7.90. Gối = mục tiêu sai số vị trí cho scorer | Nếu chỉ cần ±0 ms, scorer khó đạt → đổi sang cách gate mềm |
| G2 | Adapter **không prompt** sửa được lỗi viết ngang hoặc hơn prompt, và gate hơn boost không gate | Cùng list (top-50 ∪ từ đúng) + span: (a) prompt (mốc), không prompt: (b) sửa sau kiểu PTC, (c) boost không gate (Liu 2508.17796), (d) gate. Chia lỗi theo nhóm: gần âm, đồng âm hoàn toàn, ranh giới từ, bỏ sót, chèn lặp | (d) ≤ (a) ở nhóm gần âm / đồng âm, (d) < (c). Chèn sai ≤ prompt sau khi chặn chèn lặp (mỗi từ boost 1 lần, dừng khi đã viết) | (d) thua (b): đóng góp chuyển sang sửa sau theo span |
| G3 | Thiếu bước "đoạn này có từ nào không" là lý do list thật hỏng | (i) Rẻ: điểm Hou25 + ngưỡng / top-k nhỏ + span S1. (ii) Scorer mới có "không từ nào" theo thời gian. Chạy N2000 và `missing_N` | B-WER < 9.69, WER ≤ 3.41, `missing_N` chèn ≈ prompt | Cần thêm bước xác nhận sau khi decode |
| G4 | **Đọc vệt sáng** tốt hơn CNN + lấy trung bình | Cùng bản đồ giống nhau kiểu Hou25, thay CNN + pool bằng đường chéo sáng nhất (đơn điệu, train được). Đo recall theo câu ngắn/dài, từ ngắn/dài, và sai số vị trí | Recall ≥ Hou25, hơn rõ ở câu dài và từ ngắn. Sai số vị trí ≤ gối của G1 | Giữ CNN cho điểm, chỉ dùng đường chéo để lấy vị trí |
| G5 | Cho các từ gần âm đấu nhau tại cùng chỗ thì ít chọn nhầm | Bật/tắt phần cạnh tranh + từ nhiễu gần âm đặt tại đúng chỗ khi train; test `near_N2000`. Đếm riêng đồng âm hoàn toàn và báo dạng **nhóm** | Chọn nhầm giảm. Đo được tỉ lệ lỗi đồng âm hoàn toàn: phần audio không thể sửa | Bỏ phần cạnh tranh |
| G6a | Đoán giây decoder đang nghe **từ chữ đã viết** (căn đoạn text đã sinh với audio bằng encoder của scorer hoặc một CTC aligner nhỏ, căn lại mỗi khi xong một từ) tốt gần bằng alignment heads | Trên Whisper, có cả hai cách: so độ lệch với MFA, so B-WER khi gate, đo thời gian chạy | Lệch ≤ gối của G1; B-WER gần mức của alignment heads; chậm thêm ≤ 2× | Mức 1 chỉ dành cho model có cách lấy vị trí riêng; các model còn lại dùng mức 2 |
| G6b | Cùng một scorer, không train lại, không chỉnh prompt, giúp được nhiều SpeechLLM | Whisper-turbo, Qwen3-ASR, thêm ≥ 1 SpeechLLM mở. Mỗi model chạy (1) gate lúc decode bằng G6a, (2) sửa sau theo vị trí; mốc = prompt top-50 Hou25 của chính model đó | Mọi model đều thắng mốc ở ít nhất 1 mức; mức 1 tốt nhất | Báo cáo model nào hỏng và vì sao |

**Các mức cắm (adapter không prompt, chốt 13/11):**

| Mức | ASR phải cho gì | Dùng được cho |
|---|---|---|
| 1. Gate lúc decode | Logits + tokenizer | Mọi model mở |
| 2. Sửa sau theo vị trí | Text đầu ra | Mọi model |
| (đường lui) Prompt vài từ chắc chắn | Ô nhập prompt | Chỉ model đóng gọi qua API; không phải phương pháp chính, xem §0.3b |

**Thứ tự:**
1. G1, G2, G6a: chỉ decode Whisper, không train, cần 1 GPU.
2. G3(i).
3. G4, G5: train scorer.
4. G6b.

**Novelty còn phải kiểm:**
- Đọc toàn văn CTC-WS, Spike-triggered, CB-Whisper, KG-Whisper, xem có ai gate theo alignment heads của Whisper chưa.
- Tìm thêm theo hướng "attention-guided shallow fusion" và "timestamp-conditioned biasing".
- Kỹ thuật "tìm đường chéo trong bản đồ giống nhau" (DTW, tìm từ bằng mẫu âm thanh) đã cũ. Điểm mới của G4 chỉ có thể nằm ở chỗ nó được học end-to-end và phục vụ gate, không ở bản thân thuật toán.

### 0.6 Gap riêng của khâu chọn từ (scorer), IS21 LibriSpeech (13/11)

Nguồn: Hoàng F7–F16; `scorer/results.md` §6–9; paper BR-ASR Table 2, PTC Table 3.

Số trên test-other, N2000, gồm 3 metric:
- **@99** = số từ phải giữ để đủ 99% từ đúng, càng thấp càng tốt;
- **#50** = % từ đúng lọt top-50, càng cao càng tốt;
- **H#50** = % từ nghe giống lọt top-50, càng thấp càng tốt.

| Scorer | @99 ↓ | #50 ↑ | H#50 ↓ | Ghi chú |
|---|---|---|---|---|
| Hou25 (nhóm tái hiện, 2 epoch) | 181 | 95.4 | **17.6** | TTS 21 GB, 922 ms/câu |
| pmaxsim (âm vị, nhóm) | 100 | 98.4 | 25.5 | — |
| BR-ASR acoustic (paper) | 42 | **99.7** | 69.3 | — |
| PTC (paper, 960h + WavLM) | **17** | 99.3 | 22.0 | — |

| Gap | Bằng chứng | Chỗ trống cho scorer mới |
|---|---|---|
| **S1. Bắt đủ ↔ không bị lừa: chưa ai có cả hai** | Hou25 H#50 tốt nhất nhưng @99 tệ nhất. BR-ASR ngược lại. Đỉnh vệt tách từ đúng/nhiễu tốt hơn output (AUC 0.992 so với 0.975, F14), nghĩa là average pool làm loãng. Câu dài: R#10 92.6 → 77.4 | Giữ bản đồ frame × frame (phân biệt tốt), đọc **cục bộ**, **train lại** (đổi lúc test đã thất bại, A1) |
| **S2. Đuôi khó khoảng 1% từ đúng, chưa ai phân tích** | @99 phình lên 100–181. 960h chỉ giảm còn 54–65 | Phân tích đuôi trước khi thiết kế (§0.7) |
| **S3. Số từ cố định, p không đáng tin** | Top-50 cho cả câu không có từ hiếm (F11). Hạng 11–50 sai nhiều gấp khoảng 8 lần (F12). Khoảng 2 từ chắc chắn tốt hơn 50 từ (F16 −0.41). Qwen: p ≥ 0.5 tốt hơn top-50 (−1.86), nhưng precision chỉ 48% | p hiệu chỉnh + "không từ nào" → số từ thay đổi theo câu |
| **S4. Thời gian có nhưng bị vứt, hoặc dùng sai chỗ** | Vệt định vị 0.05 s (F14). Xếp cả 50 từ theo thời gian thì tệ hơn (F15). Chỉ xếp từ chắc chắn thì tốt hơn (F16) | Xuất **có/không + vị trí cùng nhau** |
| **S2 đã phân tích (13/11, `scorer/analysis/TAIL.md`)** | (1) 174 / 242 từ Hou25 sót (other) thì scorer âm vị vẫn bắt được → lỗi riêng của mô hình. (2) Từ sau giây 30 bị mất 83–100% ở mọi hệ (encoder Whisper cắt 30 s), chiếm 16% đuôi chung. (3) 94% đuôi chung là từ Whisper không prompt cũng nghe sai. (4) Từ < 0.25 s, từ có `'` hoặc `-`, từ ngoài kho: sót gấp 2.5–3 lần | Cửa sổ trượt cho câu > 30 s (sửa trước); đọc cục bộ; nhiều cách đọc cho mỗi từ; báo recall trên nhóm "ASR nghe sai" |
| **S5. Đắt và phụ thuộc** | Hou25: TTS mọi từ, 2.7% từ không có TTS (F8). PTC: g2p + đầu âm vị theo encoder | Encoder riêng; biểu diễn từ quyết ở D4 |

---

## 1. Hai paper giống nhau ở đâu (chứ không phải khác nhau ở đâu)

Hai paper khác nhau ở gần như mọi chi tiết:
- biểu diễn từ: TTS so với âm vị;
- có train hay không: train scorer so với chỉ train bộ nhận âm vị;
- cách chấm: CNN so với Viterbi;
- cách chọn: top-50 so với cạnh tranh.

Nhưng cả hai **cùng một khung**:

```
audio + list ──► SCORER ──► vài từ (dạng CHỮ) ──► ASR đọc prompt ──► transcript
                                                              (PTC: ──► sửa sau khi viết xong)
```

Giả định ngầm của khung này: **chọn được đúng từ thì ASR sẽ viết đúng từ.** Vì vậy cả hai đều đo và tối ưu khâu *chọn từ* (Hou25: top-k; PTC: Table 3 về retrieval).

## 2. Gap chung

### Gap 1: chọn đúng từ ≠ viết đúng từ

Ví dụ: câu có từ "bessy". Scorer chọn đúng "bessy", prompt có "bessy". Whisper vẫn viết "bessie", vì đó là chính tả quen hơn.

| Bằng chứng | Nguồn |
|---|---|
| Lỗi còn lại khi dùng prompt top-50: 95% là từ **đã có trong prompt**. Oracle (prompt chỉ chứa từ đúng) chỉ sửa thêm 2–3% so với top-50. | [Hoàng] F12 |
| Trong các lỗi đó, 64–70% là nghe giống hoặc đồng âm; ~12% là ranh giới từ ("grave digger" → "gravedigger"). | [Hoàng] F12 |
| PTC tự nêu ở §1 rằng từ đúng đã ở trong prompt mà SpeechLLM vẫn viết ra từ gần âm, phổ biến hơn → họ phải thêm bước sửa sau decode. | [PTC] §1 |
| Scorer tốt hơn không giảm được nhóm lỗi nghe giống: 4.69 so với oracle 4.62. | [ta], họp 06/11 |
| Đưa thứ tự/thời gian vào prompt chỉ lời tối đa 0.7 điểm. | [Hoàng] F13, F16 |
| Cùng list đúng: prompt 9.69; đẩy từ đúng lên **lúc decode, đúng chỗ**: 7.90. | [ta] E1, test-other |

Hai paper xử lý khâu này thế nào [diễn giải]:
- **Hou25:** không xử lý. Chỉ có prompt.
- **PTC:** sửa **sau** khi ASR viết xong, bằng so sánh âm học. Nhưng:
  - Chính paper nói đồng âm tuyệt đối là thứ "phoneme-CTC cannot distinguish" (§2.5). Mà "bessy" và "bessie" đọc y hệt nhau. Thứ phân xử được là **chính tả trong list**, không phải âm.
  - Sửa sau là sửa trên chuỗi chữ đã viết xong. Khi từ bị tách hoặc dính, PTC phải có luật riêng cho ranh giới từ, và chỉ sửa khi vượt ngưỡng 2.0 (§3.2), tức bảo thủ.
  - Lời thực tế ~0.6 B-WER (Table 1, Qwen3 AuT, N2000: 9.24 → 8.61 trên test-other).

→ **Gap 1:** khâu nghẽn nằm ở **lúc ASR chọn chữ**. Cả hai paper đều để scorer đứng ngoài khâu đó, một bên đứng trước (prompt), một bên đứng sau (sửa).

### Gap 2: chấm từ bằng một mẫu phát âm chuẩn duy nhất [phụ, cần đo thêm]

| Bằng chứng | Nguồn |
|---|---|
| Hou25: mỗi từ chỉ một giọng TTS; từ không có TTS thì điểm −∞ (2.7% từ đúng của test). | [Hou25] §3.3, [Hoàng] F8 |
| PTC: mỗi từ chỉ một phát âm g2p; kết quả phụ thuộc bộ nhận âm vị (WavLM 4.10 → AuT 4.45). | [PTC] §2.3, Table 1 |
| 11–16% từ hiếm có phát âm g2p khác phát âm MFA. | [ta], họp 06/11 (lệch có thể do hai từ điển khác nhau) |
| Từ ngắn (0–3 âm vị) rớt top-10 nhiều nhất: 19%. | [ta] A4 |

→ **Gap 2:** từ hiếm (tên riêng, phương ngữ) chính là loại từ mà "phát âm chuẩn" không chắc đúng, vậy mà cả hai đều dựa vào nó.

## 3. Từ gap suy ra scorer phải làm gì (yêu cầu, chưa phải kiến trúc)

Nếu chỗ nghẽn là lúc ASR chọn chữ (Gap 1), thì scorer phải đưa được bằng chứng vào **từng bước decode**. Từ đó ra 4 yêu cầu:

| # | Yêu cầu | Vì sao (từ gap / số đo) |
|---|---|---|
| R1 | Bằng chứng phải **theo thời gian**: "ở đoạn audio này, từ X có đang được nói không". Không phải một điểm cho cả câu. | Decoder viết từng chữ tại từng chỗ; điểm cả câu không biết đặt vào chữ nào. E1: không có vị trí thì boost hỏng (22.26). |
| R2 | Tại mỗi thời điểm phải có lựa chọn **"không từ nào trong list"**, và đó là trường hợp phổ biến nhất. | Phần lớn thời gian không ai nói bias word. E1: boost cả top-50 theo vị trí thì từ nhiễu cũng được đẩy → WER 15.8. |
| R3 | Các từ trong list phải **tranh nhau** tại cùng một thời điểm (chỉ một từ được nói ở một chỗ). | Hệ quả trực tiếp của R1 và R2: câu hỏi "đang nói từ nào" có đáp án là một phân phối trên (list ∪ không từ nào). |
| R4 | Khớp từ với audio phải **chịu được phát âm lệch chuẩn**. | Gap 2. |

Đầu ra suy ra: một **phân phối theo thời gian** q(w | t) trên (list ∪ "không từ nào").
- Nếu cần p cho cả câu (để dùng prompt), lấy max của q theo t.
- Nếu cần vị trí, lấy đoạn mà q(w | ·) cao.

p và vị trí **không phải hai đầu ra ghép thêm**; chúng là hai cách đọc của cùng một q(w | t).

## 4. Giả định cần chứng minh

| # | Giả định | Căn cứ hiện có | Cách kiểm | Tiêu chí | Nếu sai |
|---|---|---|---|---|---|
| H1 | Gap 1 có thật trên **cùng pipeline** của nhóm, với **cả hai** hệ: phần lớn lỗi còn lại là từ đã được chọn đúng | F12 (chỉ Hou25, pipeline Hoàng) | chạy Hou25 và PTC trên cùng Whisper-turbo + IS21; phân loại lỗi từng từ (đã chọn hay chưa; nghe giống / đồng âm / ranh giới / khác) | ≥ 70% lỗi còn lại của **cả hai** hệ là từ đã được chọn | gap 1 yếu → quay lại bài toán chọn từ |
| H2 | Đưa bằng chứng vào **lúc decode** sửa được nhóm lỗi mà sửa-sau-decode không sửa được (đồng âm, ranh giới) | E1 7.90 so với 9.69; PTC §2.5 | **cùng thông tin oracle** (từ đúng + đoạn MFA): (a) prompt, (b) sửa sau kiểu PTC, (c) đẩy lúc decode; so B-WER **theo từng nhóm lỗi** | (c) < (b) rõ ở nhóm đồng âm và ranh giới; U-WER không tăng | gap 1 có, nhưng kênh decode không hơn → đóng góp chuyển sang cách sửa sau |
| H3 | Có "không từ nào" theo thời gian thì đẩy lúc decode không chèn bừa, kể cả list toàn nhiễu | E1 top-50 hỏng (15.8) | scorer mới, list N2000 và `missing_N`; đếm từ chèn sai, câu lặp | chèn sai ≤ prompt Hou25; `missing_N` gần như không chèn | phải thêm ngưỡng / bước xác nhận |
| H4 | Scorer theo thời gian vẫn **chọn từ** không kém hai baseline và định vị ≤ 0.05 s | F14 (thông tin vị trí có trong audio); PTC Table 3 | R_B#10, R_B@99, R_H#50, sai số vị trí (mốc MFA), cùng điều kiện | R_B@99 ≤ PTC chạy lại; vị trí ≤ 0.05 s | dùng retrieval của baseline, chỉ giữ phần decode |
| H5 | Khớp học được chịu phát âm lệch tốt hơn mẫu chuẩn (Gap 2) | 11–16% g2p ≠ MFA; từ ngắn rớt 19% | tách nhóm từ g2p ≠ MFA, từ ngắn, từ ngoài CMUdict; so recall của 3 hệ | mức rớt ở nhóm này thấp hơn cả hai baseline | Gap 2 bỏ khỏi paper |
| H6 | Dùng được cho ASR khác (Qwen3-ASR) | — | cùng scorer, đổi ASR | có lời so với prompt | ghi là giới hạn |

**H1 và H2 quyết định paper**, và kiểm được **không cần scorer mới** (dùng oracle + code PTC). Làm trước.

## 5. Scorer đề xuất [ý tưởng, chờ duyệt]

Mục tiêu: **vừa chấm điểm tốt hơn cả hai** (chọn ít từ mà đủ, ít bị từ nghe giống lừa), **vừa cho vị trí đúng** (≤ 0.05 s), và đầu ra là q(w | t) để dùng lúc decode (R1–R4).

### 5.1. Vì sao hai scorer hiện có chấm chưa tốt (chẩn đoán, có số)

| Triệu chứng | Số | Ở hệ nào | Nguyên nhân nghi ngờ |
|---|---|---|---|
| Câu dài thì tụt | R#10: câu < 15 từ 92.6 → câu ≥ 50 từ 77.4 | Hou25 [ta] A4 | điểm cả câu = trung bình cả bản đồ → vệt ngắn bị loãng |
| Từ ngắn bị rớt | 0–3 âm vị: rớt 19%, gấp 2–3 lần từ dài | Hou25 [ta] A4 | ít frame → ít bằng chứng; dễ khớp nhầm |
| Từ đúng bị rớt vì **điểm thấp**, không phải vì từ nghe giống vượt lên | kẻ vượt là từ nghe giống chỉ 1.7–5% | Hou25 [ta] A4 | scorer "không nghe ra" từ |
| Phải giữ nhiều từ mới đủ 99% | Hou25 ~256 [GOAL]; PTC 16.9 | cả hai | PTC tốt hơn hẳn → có điều gì đó ở PTC làm tốt việc này |
| Chất lượng phụ thuộc bộ nhận âm vị | WavLM 4.10 so với AuT 4.45; PER other 2.27 so với 5.13 | PTC Table 1–2 | phần khớp là luật cố định, chỉ tốt bằng posterior đầu vào |
| Vị trí bị vứt | S1 92% trong span → sau CNN 45% | Hou25 [ta] A1 | kiến trúc không giữ trục thời gian tới đầu ra |

Nhận xét then chốt [diễn giải]:
- PTC tìm từ tốt vì bộ nhận âm vị của họ được **dạy từng frame** (CTC).
- Hou25 chỉ được dạy **một nhãn cho cả câu** ("từ có trong câu không").
- Cả hai triệu chứng nặng nhất của Hou25 (câu dài, từ ngắn) đều là chỗ mà một nhãn cho cả câu cho ít tín hiệu nhất.

→ Giả định trung tâm của scorer: **dạy theo thời gian** (từ nào ở frame nào) thì vừa chấm điểm tốt hơn, vừa có vị trí. Không phải đánh đổi giữa hai thứ.

### 5.2. Kiến trúc

```
                    ┌──────────────── A. ĐẦU AUDIO ────────────────┐
audio ─► encoder frozen (Whisper-large-v3; ablate WavLM) ─► trộn các lớp (học trọng số) ─► Transformer 2 lớp ─► h_t   (20 ms/frame)

                    ┌──────────────── B. ĐẦU TỪ ───────────────────┐
từ w ─► các phát âm (g2p + biến thể CMUdict) ─► embedding âm vị + Transformer nhỏ ─► e_1..e_U
         (lúc train: thay/xoá âm vị ngẫu nhiên để chịu lệch chuẩn)                          (Gap 2, R4)

                    ┌──────────────── C. ĐỊNH VỊ (cho MỌI từ trong list, rẻ) ──┐
S_w[u,t] = sim(e_u, h_t)  ─►  căn chỉnh ĐƠN ĐIỆU: âm vị đi trái → phải, mỗi âm vị ≥ 1 frame,
                               bắt đầu ở đâu cũng được (forward mềm, khả vi)
                         ─►  a_w(t) = điểm "w vừa được nói xong tại frame t"  + đoạn [bắt đầu, t]

                    ┌──────────────── D. KIỂM CHỨNG (chỉ top-M ứng viên, đắt) ──┐
với M đoạn (w, [a,b]) điểm cao nhất: cross-attention âm vị ↔ frame trong cửa sổ [a−δ, b+δ]
                         ─►  điểm tinh v_w ; chỉnh lại ranh giới a, b

                    ┌──────────────── E. CẠNH TRANH THEO THỜI GIAN ──┐
tại mỗi t: softmax trên {các w có đoạn phủ t} ∪ {KHÔNG TỪ NÀO}  ─►  q(w | t)

ĐẦU RA:  q(w | t)   ─►  p(w) = max_t q(w | t)   ;   [a_w, b_w] = đoạn có q cao nhất
```

Train (một lần, đầu-cuối):

| Loss | Dạy cái gì | Nhãn lấy từ đâu |
|---|---|---|
| L_time: cross-entropy theo frame trên (list ∪ không từ nào) | **từ nào đang được nói ở frame nào** | forced alignment của train-clean-100 (word span) |
| L_span: lệch ranh giới [a, b] so với span thật | vị trí chính xác | cùng forced alignment |
| L_utt: BCE cho p(w) | "có trong câu không" (giống Hou25) | transcript |
| (tuỳ chọn) L_ctc: CTC âm vị trên h_t | đầu audio nghe được âm vị (giống PTC) | g2p của transcript |

- List khi train: từ đúng + từ nhiễu. 40% là từ gần theo edit distance, 60% ngẫu nhiên, như Hou25 §3.3. Thêm **từ nhiễu cùng chỗ**: từ nghe giống đặt vào đúng đoạn của từ đúng, để khối E học cạnh tranh.
- Dữ liệu: train-clean-100 cho bảng chính (công bằng với Hou25); 960h ghi riêng (công bằng với PTC).

Vì sao mỗi khối có mặt:

| Khối | Giải quyết triệu chứng nào (5.1) | Giả định | Lý do thiết kế |
|---|---|---|---|
| A. trộn lớp + Transformer | phụ thuộc chất lượng đầu audio | S-A | PTC §2.3 làm vậy; lớp giữa của encoder thường giữ âm vị tốt hơn lớp cuối [cần kiểm] |
| B. nhiều phát âm + nhiễu âm vị | 11–16% g2p ≠ MFA; từ ngắn | S-B (Gap 2) | không đặt cược vào một phát âm chuẩn |
| C. căn chỉnh đơn điệu | vị trí bị vứt; câu dài loãng | S-C | vị trí có **theo cấu tạo**; điểm là của một đoạn, không phải trung bình cả câu |
| D. kiểm chứng cục bộ | từ nghe giống; ranh giới chưa đủ chính xác | S-D | rẻ cho 2000 từ ở C, đắt chỉ cho M từ ở D |
| E. cạnh tranh có "không từ nào" | câu không có từ hiếm; từ nghe giống cùng chỗ | S-E | R2, R3; cũng là đầu ra cho decode |
| L_time (dạy theo thời gian) | câu dài, từ ngắn, phải giữ nhiều từ | **S-T (trung tâm)** | nhiều tín hiệu hơn một nhãn cho cả câu |

### 5.3. Giả định của scorer và thí nghiệm chứng minh

Thiết lập chung cho mọi thí nghiệm scorer:
- Train trên train-clean-100; chọn checkpoint trên dev-other (list IS21, N2000).
- Test: test-clean, test-other; N ∈ {100, 500, 1000, 2000}; thêm `ctx_hard_N2000`, `near_N2000` (list nghe giống) và `missing_N` (list không có từ đúng).
- Baseline chạy **cùng điều kiện**:
  - Hou25 (checkpoint Hoàng).
  - PTC Retrieval (code của họ, đầu âm vị train lại trên cùng encoder Whisper + 100h; bản WavLM 960h của họ ghi riêng).
- Mỗi số chính có CI bootstrap theo câu; cấu hình cuối chạy 3 seed.

Metric:

| Nhóm | Metric | Nguồn |
|---|---|---|
| Chấm điểm | R_B#10, R_B#50, **R_B@99** (số từ cần để đủ 99% từ đúng), **R_H#50** (từ nghe giống lọt top-50) | PTC §3.3, BR-ASR |
| Chọn bằng p | số từ/câu và % từ đúng giữ lại tại ngưỡng θ (θ chọn trên dev); trên `missing_N`: số từ/câu (càng gần 0 càng tốt) | nhóm (C) |
| Hiệu chỉnh | ECE của p (p = 0.8 có đúng 80% không) | thông lệ (B) |
| Vị trí | sai số tâm, đầu, cuối (trung vị); % tâm trong span; % đúng thứ tự; riêng nhóm từ ASR chép sai | [Hoàng] F14 / Table 9 |
| Tách nhóm | theo độ dài câu, độ dài từ, từ g2p ≠ MFA, từ ngoài CMUdict | [ta] A4 |

| # | Giả định | Thí nghiệm (chỉ đổi đúng 1 thứ) | Đo gì | Tiêu chí thành công | Nếu sai |
|---|---|---|---|---|---|
| **S-T** | Dạy theo thời gian làm **chấm điểm** tốt hơn dạy một nhãn cả câu, nhất là câu dài và từ ngắn | cùng kiến trúc; (a) chỉ L_utt, (b) L_utt + L_time, (c) + L_span | R_B#10, R_B@99 theo nhóm độ dài; vị trí | (b) > (a), CI không chứa 0; mức tụt ở câu ≥ 50 từ giảm một nửa | điểm chấm không nhờ thời gian → giữ L_time chỉ cho vị trí |
| S-C | Căn chỉnh đơn điệu cho vị trí chính xác **và** không làm điểm kém đi | thay C bằng (a) attention tự do không ràng buộc thứ tự, (b) gộp cả câu kiểu Hou25 | vị trí; R_B#10; R_H#50 | vị trí ≤ 0.05 s, ≥ 98% trong span; R_H#50 tốt hơn (a) | dùng attention + L_span |
| S-D | Kiểm chứng cục bộ trên top-M giảm từ nghe giống và lỗi ranh giới | bật / tắt D; M ∈ {20, 50, 100} | R_H#50, R_B#10, sai số đầu/cuối, thời gian chạy | R_H#50 ≤ 17.3 (mốc tốt nhất hiện có, Hou25); thời gian tăng < 2 lần | bỏ D |
| S-E | Cạnh tranh có "không từ nào" cho p dùng được làm ngưỡng | (a) BCE từng từ độc lập, (b) cạnh tranh chỉ lúc chạy (như PTC), (c) cạnh tranh có train | ECE; số từ/câu tại ngưỡng; `missing_N`; R_H#50 | (c) tốt nhất ở `missing_N` và ECE, không mất R_B#10 | dùng (b). **Lưu ý:** 10/2026 nhóm từng thử loss trên cả list và thấy kém, nhưng lần đó lẫn với việc cho scorer nhìn hyp của Qwen (`hxattn`) → chưa kết luận được |
| S-B | Nhiều phát âm + nhiễu âm vị giúp nhóm từ lệch chuẩn (Gap 2) | bật / tắt | R_B#10 ở nhóm g2p ≠ MFA, ngoài CMUdict, từ ngắn | tăng ở các nhóm này, không giảm ở nhóm còn lại | Gap 2 ghi là giới hạn |
| S-A | Đầu audio quyết định trần | Whisper lớp cuối / trộn lớp; WavLM | R_B@99 | — (để hiểu, không cần thắng) | — |

Mục tiêu của scorer đầy đủ (test-other, N2000, train 100h):

| | Hou25 | PTC | Mục tiêu |
|---|---|---|---|
| R_B#10 | 93.8 [Hoàng] | đo lại | ≥ 97.4 |
| R_B@99 | ~256 | 16.9 (paper, 960h, WavLM) | ≤ PTC chạy lại cùng điều kiện |
| R_H#50 | 17.3 | 22.0 (paper) | ≤ 17.3 |
| Vị trí: trung vị / % trong span | 0.05 s / 98.8% (S1, F14) | đo lại | ≤ 0.05 s / ≥ 98% |
| `missing_N`: từ/câu tại θ | 50 (top-k) | đo lại | ≈ 0 |

### So với hai paper (chỉ để định vị, không phải nguồn của thiết kế)

| | Hou25 | PTC | Nhóm mình |
|---|---|---|---|
| Đầu ra | 1 điểm / từ / câu | sự kiện (từ, đoạn, q) / câu | **q(w \| t)**: phân phối theo thời gian, có "không từ nào" |
| Bằng chứng vào ASR lúc nào | trước (prompt) | trước (prompt) + sau (sửa chữ) | **trong lúc** ASR chọn chữ |
| Train cho cái gì | có/không, cả câu | nhận âm vị (phần chọn từ là luật) | **có/không theo thời gian** |
| Đồng âm ("bessy"/"bessie") | không xử lý | không phân biệt được (§2.5) | chính tả trong list được đẩy đúng lúc |

Điểm giống PTC, phải nói thẳng trong paper: PTC cũng có cạnh tranh theo thời gian và "không từ nào" (Eq. 3–4). Khác ở chỗ: của họ là luật với τ, ρ chỉnh tay, dùng để **lọc danh sách**; của mình là phân phối **học được**, dùng **trực tiếp lúc decode**.

## 6. Kế hoạch chứng minh

| Bước | Việc | Kiểm | Cần scorer mới? | Điểm dừng |
|---|---|---|---|---|
| 0 | Chạy Hou25 và PTC (code của họ) trên cùng pipeline Whisper-turbo + IS21 + `score.py --whisper-norm` | baseline | không | có số 2 hệ cùng điều kiện |
| 1 | Phân loại lỗi còn lại của cả hai | **H1** | không | < 70% lỗi là từ đã chọn → xem lại gap |
| 2 | Oracle: cùng (từ đúng + đoạn MFA), 3 cách dùng: prompt / sửa sau kiểu PTC / đẩy lúc decode; B-WER theo nhóm lỗi | **H2** | không | đẩy lúc decode không hơn sửa sau → đổi hướng đóng góp |
| 3 | Train scorer theo mục 5.3, theo thứ tự S-T → S-C → S-E → S-D → S-B (mỗi bước thêm 1 khối, khối nào không lời thì bỏ) | H4, H5, S-* | có | thua PTC ở R_B@99 → dùng retrieval của PTC, giữ phần decode |
| 4 | Scorer + decode; N100–2000, 2 tập, `missing_N`, list nghe giống | H3 | có | chèn sai tăng → thêm ngưỡng |
| 5 | Qwen3-ASR, 960h, 3 seed | H6 | có | — |

Bước 0–2 chỉ cần GPU để decode, không cần train. Đây là cách rẻ nhất để biết gap có đủ mạnh cho paper không, **trước khi** bỏ công làm scorer.

## 7. Rủi ro
- **H2 có thể thua:** sửa sau decode của PTC đã nhắm đúng nhóm lỗi nghe giống. Nếu đẩy lúc decode không hơn ở nhóm này, gap 1 vẫn đúng nhưng lời giải không phải của mình.
- **Đẩy lúc decode cần biết decoder đang nghe frame nào.** Whisper có alignment heads (đã dùng trong E1). Qwen3-ASR chưa có cách.
- **Chọn từ:** PTC rất mạnh (R_B@99 16.9). Scorer mới chưa chắc thắng ở khâu này; paper không nên đặt cược vào đó.
- Các số [ta] về Hou25 hiện là checkpoint 2 epoch của nhóm; phải chạy lại trên checkpoint của Hoàng.
