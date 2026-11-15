# Kế hoạch nghiên cứu scorer, bản 2 (lập 11/11/2026)

> **Thay bản 08/11** (bản cũ trong git, commit `a983255`). Đọc kèm `GOAL.md` (mục tiêu) và `PROPOSAL.md` mục 0 (tài liệu, gap).
>
> Người làm: **L** = Lâm, **H** = Hoàng, **C** = agent. Hạn ghi là ngày cuối.
>
> Trạng thái: ☐ chưa · ◐ đang · ☑ xong · ✗ bỏ (ghi lý do). Cập nhật mỗi thứ Hai.
>
> Nhãn nguồn: **[đo]** = nhóm đã chạy · **[TL]** = tài liệu · **[giả định]** = chưa kiểm.

---

## 1. Câu mục tiêu và thứ phải thắng

**Một scorer cắm vào mọi SpeechLLM, train một lần, đẩy từ đúng lúc mà không cần biết bên trong model.**

- **Bảng chính:** so với Hou25 cùng điều kiện:
  - IS21 LibriSpeech test-clean / test-other, N100–2000;
  - Whisper-large-v3-turbo đóng băng;
  - scorer train trên train-clean-100;
  - WER / U-WER / B-WER chấm bằng scorer chính thức, wnorm.
- **Bảng mở rộng:** cùng scorer, không train lại, chạy trên Qwen3-ASR và ít nhất 1 SpeechLLM mở khác.

---

## 2. Chuỗi lập luận phải lấp đầy

```
Quan sát (O) ──► Giả định (A) ──► Gap ──► Kiến trúc (khối nào ← giả định nào) ──► Thí nghiệm (G) ──► Đóng góp (C)
```

Paper chỉ đứng được khi **mỗi khối kiến trúc truy ngược được về một quan sát đã đo**, và **mỗi đóng góp có một bảng chứng minh**.

### 2.1 Quan sát hiện có

| # | Quan sát | Nguồn | Trạng thái |
|---|---|---|---|
| O1 | Lỗi còn lại khi dùng prompt top-50: 95% là từ **đã có trong prompt**; 64–70% là gần âm hoặc đồng âm | Hoàng F12 | [đo, Hoàng] |
| O2 | Đúng từ + gate theo vị trí MFA: B-WER **7.90** so với prompt Hou25 9.69; U-WER giữ 2.63. Bỏ gate: 22.26. Nhưng chèn sai 115 so với 39 | `bench/results.md` §6, test-other N2000 | [đo] |
| O3 | Vị trí đoán bằng S1 (thay MFA): 8.42. List thật top-50 + gate S1: WER 15.76, 37 câu lặp, vì từ nhiễu cũng có vị trí | như trên | [đo] |
| O4 | S1 **định vị tốt** (0.05 s, 92–96% trong span) nhưng **chọn kém** (R#10 65.6 so với 93.8 của cách lấy trung bình) | Hoàng F14; `scorer/analysis/RESULTS.md` A1 | [đo] |
| O5 | Câu dài làm R#10 tụt mạnh (test-other 92.6 → 77.4). Từ ngắn bị rớt nhiều hơn | `scorer/analysis/RESULTS.md` A4 | [đo], có biến nhiễu: câu dài có nhiều từ hiếm hơn |
| O6 | Trên list IS21, từ gần âm **không** phải lý do từ đúng rớt khỏi top-10 (bỏ hết chỉ +0.1 R#10). Nó là lý do của lỗi **viết** (O1) | như trên | [đo] |
| O7 | Đổi cách gộp lúc test (đọc ở vệt thay trung bình), cùng checkpoint: R#10 **không tăng** | `scorer/analysis/A1_REPORT.md` | [đo], biến nhiễu: CNN được train cùng average pool |
| O8 | Bias đúng thời điểm chỉ có cho CTC / transducer / AED có CTC. Với Whisper / SpeechLLM thì chỉ có prompt hoặc sửa sau; vị trí bị bỏ | `PROPOSAL.md` §0.1 | [TL], mới tra khoảng 10 lượt |
| O9 | Hou25 cần khoảng 256 từ để giữ 99% từ đúng; PTC 16.9 (số của paper, setup khác) | Hoàng; PTC Table 3 | [đo] / [TL] |
| O10 | **(R4)** Với prompt Hou25, lỗi còn lại 10.06 (plain): **26% bỏ sót khi chọn** (2.65), 74% đã có trong prompt mà vẫn sai. Gate oracle chạy **không prompt** (sửa 13/11): so với không context, nghe giống 7.23 → 2.37 và đồng âm 3.01 → 0.60, tức **ngang hoặc hơn prompt top-50** (2.88 / 0.80) mà không phụ thuộc định dạng prompt. Nhưng gate **thêm 2.02 lượt chèn lặp từ hiếm**, và O2 vẫn lẫn biến nhiễu "list đúng" | `related-work/novelty-gate.md` §4 | [đo] |
| O12 | **Prompt phụ thuộc vào từng model.** Whisper chỉ giữ 223 token cuối; đổi thứ tự từ là đổi B-WER (bảng tăng/giảm dần của Hou25 thực chất là do cắt); prompt kiểu "nói" khác kiểu liệt kê; prompt 1–2 từ làm Whisper lặp chữ (WER 14.29); Qwen lại tốt nhất với prompt ngắn. → Adapter dựa trên prompt **không** cắm chung được cho mọi SpeechLLM | Hoàng F1, F6, F13; Hou25 Table 2–3; `bench/results.md` §3 | [đo] + [TL] |
| O11 | **(R1–R2)** Ma trận cosine TTS × audio + CNN đọc vệt là của CB-Whisper (2309.09552), có trước Hou25. Boost lúc decode **không gate** trên Whisper đã có: Liu et al. 2508.17796, IS21 N1000, test-other B-WER 17.55 → 9.90. Gate theo thời gian cho decoder tự hồi quy đóng băng: **chưa thấy** | `related-work/novelty-gate.md` §1–2 | [TL] |

### 2.2 Giả định cần kiểm (mỗi giả định có đúng 1 thí nghiệm quyết định)

| # | Giả định | Dựa trên | Kiểm bằng |
|---|---|---|---|
| A1 | Có vị trí đúng thì gate lúc decode sửa được lỗi **viết** gần âm mà prompt và sửa-sau không sửa được | O1, O2 | G2 |
| A2 | Gate chịu được vị trí lệch tới một mức đo được | O3 (S1 lệch vẫn 8.42) | G1 |
| A3 | Biết decoder đang ở giây nào **chỉ từ chữ đã viết**, đủ tốt để gate | — | G6a |
| A4 | List thật hỏng vì thiếu quyết định *"đoạn này có từ nào trong list không"* | O3 | G3 |
| A5 | Dạy scorer **vị trí** của từ khi train (nhãn theo frame) giúp chọn tốt hơn ở câu dài và từ ngắn, đồng thời định vị tốt | O4, O5, O7 | G4 |
| A6 | Cho các từ tranh nhau tại cùng chỗ thì ít chọn nhầm, **trên list nghe giống** (không phải IS21 thường, xem O6) | O6 | G5 |
| A7 | Scorer độc lập với ASR thì chuyển được sang SpeechLLM khác mà không train lại | — | G6b |

### 2.3 Gap (bản viết vào paper; chỉ chốt sau phần R)

> Các scorer cho SpeechLLM (Hou25, BR-ASR, PTC) tối ưu khâu **chọn từ** rồi đưa từ vào dưới dạng **chữ**. Nhưng phần lớn lỗi còn lại là từ **đã chọn đúng mà vẫn viết sai** (O1).
>
> Bias đúng thời điểm, thứ sửa được loại lỗi này, mới chỉ có cho ASR đồng bộ theo frame (O8). Với decoder tự hồi quy đóng băng thì chưa có, vì không biết decoder đang nghe đoạn nào. Scorer hiện có cũng không cho ra đủ thông tin để làm việc đó: không có vị trí (Hou25), hoặc không có "không từ nào" ở từng đoạn (O3).

### 2.4 Kiến trúc: khối nào sinh ra từ giả định nào

```
                     ┌──────────────────────── SCORER (train 1 lần) ────────────────────────┐
audio ─► encoder riêng, đóng băng (A7) ─► bản đồ giống nhau từ × audio ─► đọc vệt (A5) ─► tranh nhau + "không từ nào" (A4, A6) ─► (từ, p, t_đầu, t_cuối)
list ──► biểu diễn từ (TTS hoặc âm vị: quyết ở D4) ───┘
                     └──────────────────────────────────────────────────────────────────────┘
                                                   │
                     ADAPTER KHÔNG DÙNG PROMPT (chốt 13/11, O12):  mức 1 gate (logits + tokenizer; con trỏ từ chữ đã viết, A3) · mức 2 sửa sau (chỉ text)
                     prompt chỉ là đường lui cho model đóng gọi qua API, không phải phương pháp chính
```

| Khối | Vì sao có (giả định) | Bản tối thiểu | Bản đầy đủ | Bỏ nếu |
|---|---|---|---|---|
| Encoder riêng | A7 | Whisper-large-v3 encoder đóng băng (như Hou25) | + WavLM để ablation | — |
| Biểu diễn từ | A5 (vệt chéo dốc 1 có sẵn khi hai phía đều là audio) | TTS như Hou25 | Âm vị + dự đoán độ dài, hoặc nhiều giọng TTS | D4 cho thấy không khác |
| Đọc vệt | A5, O4, O7 | Nhãn frame + đọc trong cửa sổ quanh vệt | Đường chéo đơn điệu, học end-to-end | G4 thua Hou25 khi train lại |
| Tranh nhau + "không từ nào" | A4, A6 | Ngưỡng p + giữ từ cao nhất trên mỗi đoạn | Phân phối theo từng đoạn thời gian trên {từ} ∪ {không}, train cùng | G3, G5 không giúp |
| Adapter (không prompt) | A1, A2, A3, O12 | Mức 2 (sửa sau theo span) + mức 1 trên Whisper bằng alignment heads | Mức 1 với con trỏ từ chữ đã viết, mọi model mở | G6a thua: mức 1 chỉ cho Whisper, model khác dùng mức 2 |

### 2.5 Đóng góp và bảng chứng minh

| # | Đóng góp | Bằng chứng trong paper |
|---|---|---|
| C1 | Scorer cho ra (từ, p, vị trí), có "không từ nào"; chọn ít từ mà vẫn giữ recall | **Bảng 2**: R_B#10, R_B@99, H#50, sai số vị trí, so Hou25 cùng checkpoint encoder và cùng dữ liệu |
| C2 | Adapter **không dùng prompt** (gate lúc decode / sửa sau theo span) cho SpeechLLM đóng băng, không cần biết bên trong model, không phụ thuộc định dạng prompt | **Bảng 1**: B-WER / U-WER / chèn sai, N100–2000 + `missing_N`, so prompt Hou25. **Bảng 4**: nhiều SpeechLLM |
| C3 | Phân tích: lỗi còn lại thuộc loại nào; vị trí cần chính xác đến đâu | **Hình** đường B-WER theo độ lệch vị trí (G1); bảng lỗi theo nhóm (G2) |

---

## 3. Mốc và điểm quyết định

| Mốc | Ngày | Câu hỏi | Nếu KHÔNG đạt |
|---|---|---|---|
| **M1** | 29/11 | G2: gate lúc decode (oracle) thắng **cả** prompt **và** sửa sau, ở nhóm lỗi gần âm; chèn sai ≤ prompt? | C2 đổi thành "sửa sau theo vị trí"; scorer vẫn giữ |
| **M2** | 29/11 | G6a: con trỏ từ chữ đã viết lệch ≤ mức cho phép của G1? | Mức 1 chỉ cho Whisper; model khác dùng mức 2 (sửa sau) |
| **M3** | 03/12 | Chốt giả định + kiến trúc (họp, **L duyệt**) | — |
| **M4** | 20/12 | G4: dạy vị trí khi train làm R#10 câu dài tăng, định vị ≤ mức G1? | Scorer = Hou25 (chọn) + S1 (định vị) + khối quyết định; claim C1 thu nhỏ |
| **M5** | 10/01 | Bảng 1 thắng Hou25 trên Whisper | Xem lại hướng với thầy |
| **M6** | 17/01 | Bảng 4: ≥ 2 SpeechLLM, adapter không prompt đều thắng prompt top-50 của chính model đó | Model nào hỏng thì ghi rõ lý do |
| **M7** | 05/02 | Nộp (hạn 09/02) | — |

---

## 4. Các phần việc

### R. Nghiên cứu nền: quan sát và độ mới (11–24/11, không cần GPU trừ R2)

| ID | Việc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|
| ☑ R1 (11/11 → `related-work/novelty-gate.md`) | Đọc **toàn văn**: CTC-WS 2406.07096, Spike-triggered 2310.04657, CB-Whisper 2309.09552, KG-Whisper 2406.02649, Simul-Whisper 2406.10052, TCPGen-Whisper (2509.09197 / 2508.17796), PAC 2509.12647, BR-ASR 2505.19179, CTC-aligned KWS 2406.07923 | `related-work/novelty-gate.md` | Mỗi paper: có vị trí? dùng vị trí lúc nào? loại ASR? phải train? Có Section/Table. Kết luận claim nào an toàn | 18/11 | C |
| ☑ R2 | Tra thêm: "attention-guided shallow fusion", "timestamp-conditioned biasing", "forced alignment during decoding", "query-by-example DTW keyword spotting" | bổ sung vào R1 | ≥ 6 truy vấn ghi lại; kết luận còn trống hay không | 18/11 | C |
| ◐ R3 (còn: top-50 chấm plain cho test-clean; kiểm thứ tự tăng dần trong prompt) | Đặt số tái hiện Hou25 cạnh số Hou25 công bố (đọc từ PDF, Table) | `bench/results.md` mục Hou25 | Bảng tái hiện / công bố; lệch bao nhiêu và lý do | 15/11 | C |
| ◐ R4 (test-other xong; còn test-clean và chép bảng vào `bench/results.md`) | Phân loại lỗi B-WER của prompt Hou25 trên **pipeline của mình**: chưa chọn / chọn rồi viết gần âm / đồng âm hoàn toàn / ranh giới từ / chèn. Dùng `bench/error_breakdown.py` | bảng trong `bench/results.md` | Khớp lại O1 bằng số của nhóm; tách riêng **đồng âm hoàn toàn** (phần audio không thể sửa) | 20/11 | C |
| ☑ R4b (13/11 → `scorer/analysis/TAIL.md`) | Đuôi khó của scorer: từ đúng sót top-50 theo đặc điểm. Kết quả: cắt 30 s; 2/3 số sót là riêng của Hou25; đuôi trùng nhóm ASR nghe sai | `TAIL.md` | — | 13/11 | C |
| R4c | Sửa cửa sổ 30 s (encoder chạy theo cửa sổ trượt), đo lại @99 / #50 cho Hou25 và pmaxsim. Cần GPU | `scorer/results.md` | @99 trước/sau; nếu giảm rõ thì mọi baseline cũng phải được sửa như vậy để so công bằng | 20/11 | C |
| R5 | Nghe 100 lỗi (50 chọn sai của scorer, 50 viết sai của ASR) | `scorer/analysis/listen100.md` | Mỗi lỗi 1 nhãn nguyên nhân; bảng tần suất | 22/11 | L, C |
| R6 | Tổng hợp quan sát | cập nhật mục 2.1 | Mỗi O có số + nguồn | 24/11 | C |

### B. Thí nghiệm trần, không train (12–29/11, 1 GPU, chỉ decode Whisper)

| ID | Giả định | Việc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|
| B0 | — | Xin GPU, dựng container, chạy lại 10 câu E1 cho khớp | — | Số khớp `bench/results.md` §6 | 13/11 | L, C |
| B1 (G1) | A2 | Oracle từ + span MFA dịch ngẫu nhiên ±0 / 50 / 100 / 200 / 500 ms, và cả câu | `bench/results.md` mục G1 + hình | Đường B-WER theo độ lệch, có CI; ghi **mức cho phép** | 20/11 | C |
| B2 (G2) | A1 | **Cùng một list** (sửa sau R4: không so oracle-gate với top-50-prompt nữa). List = top-50 ∪ từ đúng. (a) prompt list đó (mốc kiểu Hou25); **ba cách không prompt** (chốt 13/11, O12): (b) sửa sau kiểu PTC theo span, (c) boost không gate kiểu cây tiền tố (Liu 2508.17796), (d) gate. Thêm (e) prompt + gate chỉ để xem có cộng dồn không. Chia lỗi theo nhóm R4 | mục G2 | Bảng B-WER theo nhóm × 4 cách; CI ghép cặp. **(d) phải thắng (c) và (a) ở nhóm nghe giống / ranh giới** | 25/11 | C |
| B3 | A1 | **Bắt buộc:** chặn chèn lặp (R4: +2.02). Mỗi từ chỉ boost một lần trong span; dừng boost khi từ đã được viết; chỉnh λ. Chọn trên dev | mục G2 | Chèn lặp từ hiếm ≈ 0; chèn sai ≤ prompt | 27/11 | C |
| B4 (G6a) | A3 | Con trỏ từ chữ đã viết: mỗi khi xong 1 từ, căn phần text đã sinh với audio bằng CTC aligner (wav2vec2 như `align_words.py`) → giây hiện tại. So với alignment heads: độ lệch so với MFA, B-WER khi gate, thời gian chạy | `bench/gate_boost.py --pointer text` + mục G6a | Lệch ≤ mức cho phép G1; B-WER gần alignment heads; chậm thêm ≤ 2× | 29/11 (**M1, M2**) | C |
| B5 | — | Chạy gate trên test-clean (chưa có) | mục §6 | Đủ 2 tập | 29/11 | C |

### K. Chốt giả định và kiến trúc (30/11–03/12)

| ID | Việc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|
| K1 | Cập nhật mục 2 theo kết quả R, B: giả định nào đúng / sai, khối nào giữ / bỏ | `PLAN.md` mục 2, `PROPOSAL.md` | Mỗi khối có 1 số đo làm lý do | 02/12 | C |
| K2 | Họp duyệt thiết kế | biên bản `meetings/` | **L duyệt** (và thầy nếu cần) | 03/12 (**M3**) | L |

### D. Xây scorer, mỗi lần đổi 1 thứ (04/12–10/01)

Cố định cho mọi bước:
- encoder Whisper-large-v3 đóng băng, train-clean-100;
- chọn checkpoint trên dev-other N2000;
- đánh giá test-other N2000 (chính) và N100;
- metric:
  - R#10, R#50, R@99, H#50 trên `ctx_hard` và `near`;
  - sai số vị trí và % trong span;
  - chèn sai trên `missing_N2000`.

| ID | Giả định | Việc | DoD | Hạn | Ai |
|---|---|---|---|---|---|
| D0 (G3-i) | A4 | Bản rẻ, không train: điểm Hou25 → ngưỡng / top-k nhỏ → span S1 → gate | Gate + list thật: B-WER < 9.69, WER ≤ 3.41; nếu không, đo xem hỏng ở chọn hay ở vị trí | 08/12 | C |
| D1 (G4) | A5 | **Thí nghiệm then chốt:** cùng kiến trúc Hou25, train lại 2 bản: (a) nhãn cả câu (như Hou25), (b) thêm nhãn theo frame từ MFA, đọc trong cửa sổ quanh vệt | (b) R#10 câu ≥ 30 từ tăng có CI; vị trí ≤ mức G1 | 20/12 (**M4**) | C |
| D2 | A5 | Thay CNN + pool bằng đường chéo đơn điệu học được (chỉ làm nếu D1 thắng) | ≥ D1(b) trên R#10 và vị trí | 27/12 | C |
| D3 (G5) | A4, A6 | Tranh nhau + "không từ nào" theo đoạn thời gian; train với từ nhiễu gần âm đặt đúng chỗ. Đồng âm hoàn toàn trả về dạng nhóm | `near`/`ctx_hard` chọn nhầm giảm; `missing_N` gần 0 từ p ≥ θ | 03/01 | C |
| D4 | — | Biểu diễn từ: TTS so với âm vị (có dự đoán độ dài) | Chọn cái tốt hơn, hoặc ghi "ngang nhau, âm vị rẻ hơn" | 10/01 | C |

### E. Tích hợp và nhiều SpeechLLM (28/12–17/01)

| ID | Việc | DoD | Hạn | Ai |
|---|---|---|---|---|
| E1 | Bảng 1 trên Whisper-turbo: nguyên list / Hou25 top-50 prompt / scorer + prompt / scorer + gate / scorer + sửa sau. N100–2000, `missing_N`, 2 tập | Mọi ô có số + CI ghép cặp so với Hou25 | 10/01 (**M5**) | C |
| E2 (G6b) | Qwen3-ASR + 1 SpeechLLM mở (chọn ở K2): 3 mức, cùng scorer, không train lại | Bảng 4 | 17/01 (**M6**) | C |
| E3 | Thời gian chạy: scorer ms/câu theo N; gate chậm thêm bao nhiêu | Bảng phụ | 17/01 | C |

### F. Độ chắc và viết bài

| ID | Việc | DoD | Hạn | Ai |
|---|---|---|---|---|
| F1 | Ablation từng khối (tắt 1 khối mỗi dòng), 3 seed cho cấu hình chính; 960h ghi riêng | Bảng 3, mean ± std | 17/01 | C |
| F2 | Đóng băng số, hình | Mỗi số truy được về file output + lệnh chạy | 24/01 | C |
| F3 | Outline + Hình 1 (Hou25 so với của mình) | **L duyệt** | 14/12 | C, L |
| F4 | Bản nháp đủ 4 trang | Biên dịch được bằng template Interspeech | 24/01 | C, L |
| F5 | Thầy duyệt, sửa, nộp | Nộp | 05/02 (**M7**) | L |

---

## 5. Ước lượng GPU (1 card RTX 4090, chưa đo)

| Phần | Giờ |
|---|---|
| B (decode nhiều cấu hình, 2 tập) | khoảng 25 |
| D (D1 2 lần train, D2, D3, D4) | khoảng 50 |
| E (Whisper + 2 SpeechLLM) | khoảng 30 |
| F (ablation + 3 seed + 960h) | khoảng 60 |
| **Tổng** | **khoảng 165** |

## 6. Rủi ro và thứ tự cắt khi hết giờ

- **G2 thua sửa sau:** đóng góp C2 chuyển sang sửa sau theo vị trí; scorer vẫn giữ.
- **G6a thua:** mức 1 chỉ cho Whisper; claim "mọi SpeechLLM" giữ ở mức 0 + 2.
- **D1 không thắng:** dùng bản ghép (Hou25 chọn + S1 định vị + khối quyết định), C1 thu nhỏ.
- **G2 cho thấy gate không sửa được lỗi viết (R4 đã hé lộ dấu hiệu này):** C2 lùi xuống phụ. Trọng tâm chuyển sang scorer chọn ít từ mà không bỏ sót (bỏ sót = 26% lỗi còn lại), cộng sửa sau theo vị trí.
- **Novelty (R1–R2) thấy đã có người gate theo thời gian trên Whisper/SpeechLLM:** chuyển trọng tâm sang "không cần biết bên trong model" + list lớn nhiễu.
- **Cắt theo thứ tự:** 960h → D4 → D2 → SpeechLLM thứ 3.
