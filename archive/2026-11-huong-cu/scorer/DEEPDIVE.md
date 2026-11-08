# Scorer: bản chất, đang hỏng ở đâu, tối ưu thế nào (18/10/2026)

Mục tiêu và cách đo: `../EVAL_PROTOCOL.md` §7. Số trong file này: `runs/2026-10-18/log_diag_rank.txt` (script `diag_rank.py`, CPU, dùng top-50 có sẵn), kèm `results.md` §6.
Mọi số dưới đây đo trên **test** bằng checkpoint cũ → chỉ để **hiểu**, không dùng để chọn. Từ nay chọn trên dev-other.

## 1. Bản chất bài toán
Cho câu nói + 1 từ ứng viên (chuỗi âm vị, vd *artful* = `AA R T F AH L`). Muốn biết từ này có được nói không. Một scorer tốt phải trả lời đủ 3 câu:

| # | Câu hỏi | Ví dụ khi trả lời sai |
|---|---|---|
| **①** Bằng chứng âm thanh | Trong audio có đoạn nào nghe ra đúng chuỗi âm này không? | giọng khó / ồn → từ đúng chỉ được 0.1 điểm |
| **②** Căn chỉnh | Các âm có **liền nhau, đúng thứ tự, đủ dài, đúng ranh giới từ** không? | *stone* `S T OW N` và *notes* `N OW T S` có cùng tập âm → nếu không xét thứ tự thì điểm bằng nhau |
| **③** Cạnh tranh, hiệu chỉnh | Đoạn đó có cách giải thích tốt hơn không (một từ thường trong câu, một ứng viên khác)? Điểm có so được giữa 2000 từ không? | *pawley* lọt top vì nghe giống *polly* trong câu — nhưng đoạn đó đã là *polly* |

Với list 2000 từ mà chỉ ~2 từ đúng, chỉ cần mỗi từ nhiễu có xác suất nhỏ được điểm cao là top-10 đã bị chen.

## 2. Mỗi hệ xử lý 3 câu đó thế nào
| | ① bằng chứng | ② căn chỉnh | ③ cạnh tranh / hiệu chỉnh | Train trên |
|---|---|---|---|---|
| **pmaxsim** | encoder Whisper đóng băng + 1 lớp | **không**: mỗi âm vị tự tìm frame giống nhất ở *bất kỳ đâu*, lấy trung bình → "túi âm vị" | không; điểm = a·sim + c, mỗi từ độc lập | clean-100 |
| **pxattn** | như trên | yếu: âm vị nhìn nhau (self-attn) và nhìn audio (cross-attn), không ép đơn điệu | không; sigmoid độc lập, BCE với tỉ lệ 1 đúng : 4 sai | clean-100 |
| hypmatch (baseline) | **mượn ASR** (hyp Qwen lần 1) | có: khớp chuỗi âm với cụm 1–3 từ của hyp | ngầm: từ đã chép ra chiếm đoạn đó | không train |
| **PTC Retrieval** | WavLM-Large + head CTC âm vị | **có**: CTC Viterbi trên trie, đơn điệu, chuẩn hoá theo độ dài | **có**: các từ cạnh tranh theo thời gian, có lựa chọn "không có gì" | LibriSpeech 460 / 960 h |

→ PTC có đủ cả ①②③ và train trên nhiều dữ liệu hơn. Nhóm mới có ①, lại chỉ có 100 h giọng sạch. Đó là giả thuyết vì sao thua trên test-other. **Ta diễn giải, chưa kiểm.**

## 3. Đang hỏng ở đâu (test-other N2000, Recall_B#10)
| Nhóm từ đúng | pmaxsim | pxattn |
|---|---|---|
| tất cả | 95.4 | 93.0 |
| Qwen lần 1 chép **đúng** / **sai** | 98.1 / **81.9** | 96.0 / **78.0** |
| ≤ 3 âm vị (*im, da, neo*) / ≥ 9 âm vị | **87.2** / 97.2 | **87.4** / 95.3 |
| câu ≤ 10 từ / > 30 từ | 96.3 / **93.8** | 93.6 / **89.9** |

Từ đúng bị rớt khỏi top-10 (pmaxsim: 240 lượt; pxattn: 368):
- **77% (pmaxsim) / 85% (pxattn) có điểm < 0.3** → scorer gần như **không nghe ra** từ đúng (lỗi ①). Trên test-clean chỉ 48%.
- Do câu có > 10 từ hiếm (top-10 không chứa đủ): chỉ 6 / 240 lượt → không phải nguyên nhân chính.
- Các từ sai xếp trên từ bị rớt, loại trừ từ đúng khác (pmaxsim):
  - không giống gì theo khoảng cách âm vị: ~75%;
  - giống một cụm từ thường trong câu: ~12%;
  - khớp một đoạn âm lệch ranh giới từ: ~7%;
  - giống chính từ đúng (gần đồng âm): ~5%.

  Khoảng cách tính trên ARPAbet với ngưỡng 1/3, nên chỉ là ước lượng.

Ví dụ (pmaxsim):
- *"asked the **artful** young man laying a trap into which polly…"* → top: *urg'd, **pawley**, **trappeme**, …*. Các từ chen lên là mảnh của *polly*, *trap me* (lỗi ③).
- *"…he had an extra pipe and **joshua** the clerk…"* → top có *jostle* (lỗi ②/③).

**Kết luận:**
1. Lỗi chính trên test-other là **① bằng chứng yếu**: những từ Qwen nghe sai thì scorer cũng nghe kém. Kế đến là **nền nhiễu**: hàng trăm từ ngẫu nhiên được 0.3–0.6 điểm, câu càng dài càng nhiều.
2. Từ gần đồng âm chỉ chiếm ~5% trên list IS21. Negative gần (curriculum BR-ASR) quan trọng cho Recall_H và list `hard`, **không phải** đòn bẩy chính cho Recall_B.
3. **Hiệu chỉnh:** train với tỉ lệ đúng/sai 1:4 (20% là từ đúng), test chỉ ~0.1%. Vì vậy ở p ≥ 0.5, pxattn giữ 3,3 từ/câu, precision 48% trên other. pmaxsim có điểm dạng tuyến tính ràng buộc nên ít bị lệch hơn. **Ta đoán, chưa kiểm.**

## 4. Kế hoạch tối ưu — mỗi lần đổi **một** thứ, đo trên dev-other N2000
Chỉ số quyết định: **Recall_B#10** (chính), K99, recall trên từ Qwen chép sai. Giữ thay đổi nếu Recall_B#10 tăng ≥ 0.5 điểm (~25 lượt từ) và K99 không xấu đi.

| # | Thí nghiệm | Kiểm giả thuyết | Chi phí | Cần duyệt method? |
|---|---|---|---|---|
| E0 | Chạy pxattn / pmaxsim hiện tại trên dev-clean / dev-other (N2000, hard_N2000) → **mốc dev** | — | ~20 phút GPU | không |
| E1 | **Hậu xử lý, không train** (trên rank đầy đủ của E0): (a) trừ "điểm nền" của từng từ (điểm trung bình của từ đó khi là nhiễu); (b) gộp điểm scorer với hypmatch | ③ nền nhiễu; ① mượn ASR cho từ dễ, scorer cho từ khó | CPU, vài phút | không |
| E2 | **Thêm dữ liệu train:** + train-other-500 (+ clean-360) | ① bằng chứng yếu trên giọng khó | ~6–10 giờ / epoch | không (cùng mô hình) |
| E3 | **Loss theo list:** softmax trên 1 từ đúng + ~200 từ nhiễu ngẫu nhiên (+ ô "không có") thay cho BCE 1:4 | ③ hiệu chỉnh cho bài toán 1 / 2000 | như train hiện tại | nhẹ |
| E4 | **Khớp có thứ tự:** thay "mỗi âm vị lấy max" bằng căn chỉnh đơn điệu (soft-DTW, hoặc head CTC âm vị trên encoder Whisper + Viterbi như PTC) | ② | 1–2 ngày code + train | **có** — gần PTC, phải cân nhắc phần mới |
| E5 | Negative gần đồng âm theo curriculum (BR-ASR) | Recall_H, list `hard` | như train hiện tại | không |

**18/10 đêm:** E3 (dạng `--list_neg 200`) tệ hơn mọi metric → bỏ; E1 z-norm theo câu tệ hơn ở @99 → bỏ. Hou25 tái hiện: H#50 17.6 (tốt nhất), #50 95.4. Chi tiết `results.md` §8.

**Thứ tự đề xuất:** E0 → E1 (rẻ, cho biết nền nhiễu và hypmatch gỡ được bao nhiêu) → E2 → E3 → (E4 nếu sau E2–E3 vẫn thua PTC trên other) → E5.

## 5. Lộ trình so với PTC / BR-ASR trên 3 metric (test-other N2000)
| | Recall_B@99 ↓ | Recall_B#50 ↑ | Recall_H#50 ↓ |
|---|---|---|---|
| BR-ASR acoustic (2505.19179 Bảng 2) | 42.2 | 99.7 | 69.3 |
| PTC Retrieval (2609.28727 Bảng 3) | **16.9** | 99.3 | **22.0** |
| ours tốt nhất hiện có (pcascade / pmaxsim) | chưa đo (cần rank đầy đủ) | 98.6 | chưa đo đúng định nghĩa |

- **GĐ0 — thước đo khớp paper (1–2 ngày):** (a) sửa Recall_H theo BR-ASR (`../EVAL_PROTOCOL.md` §3); (b) đo lại 3 metric cho mọi hệ đã có (rank đầy đủ trên server, CPU); (c) chạy code PTC Retrieval: ra ≈ 16.9 / 99.3 / 22.0 thì thước đo của nhóm khớp. Rủi ro: checkpoint WavLM không có link → có thể phải tự train head CTC.
- **GĐ1 — E0:** mốc dev-other.
- **GĐ2 — E1 → E5 (2–3 tuần)**, mỗi thí nghiệm nhắm một metric: E1, E3 → @99; E2 → #50 và @99; E5, E4 → H#50 (Bảng 2 BR-ASR: curriculum đồng âm đưa H#50 từ 99.9 xuống 69.3). Giữ thay đổi nếu metric mục tiêu tốt lên trên dev-other mà 2 metric kia không xấu đi (#50 giảm ≤ 0.1, @99 và H#50 tăng ≤ 2).
- **GĐ3 — Hou25 tái hiện** (chạy khi GPU rảnh): TTS ~80 phút + encode + train.
- **GĐ4:** test-clean / test-other một lần → bảng "Chọn từ".

Bản tốt nhất → bảng "Chọn từ" (`../EVAL_PROTOCOL.md` §7), chạy test một lần. Đưa vào ASR: bàn sau.
