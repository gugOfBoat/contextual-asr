# Tiến độ scorer 13–15/11/2026: đã làm gì, ra gì, rút ra gì, đi tiếp đâu

> Cập nhật: 15/11/2026. Viết cho người **không theo dõi** các lượt chạy, đọc một mình là hiểu.
> - Số liệu gốc: `scorer/analysis/REPORT-1115.md` (bảng đầy đủ + khoảng tin cậy). File này chỉ giải thích lại, **không có số mới**. Mọi số ở đây đều tra được trong REPORT-1115, SMALL_R, TAIL.
> - Slide: `meetings/2026-11-20/hop-thay.pdf`. Kế hoạch gốc trước khi chạy: `scorer/PLAN-2011.md`.
> - Phạm vi: LibriSpeech test-other (2939 câu), test-clean (2620), bias list IS21; scorer train trên train-clean-100; ASR = Whisper-large-v3-turbo đóng băng, greedy. Chưa có Qwen3-ASR.
> - Quy ước: **[đo]** = chạy thật; **[diễn giải]** = suy luận của nhóm, chưa kiểm riêng.

**Cách đọc:** chỉ có 5 phút thì đọc mục 0 và mục 6. Muốn hiểu từng thí nghiệm thì đọc mục 4: mỗi thí nghiệm đều theo khung *câu hỏi → giả thuyết → làm gì → kết quả → rút ra → kết luận → điểm yếu*.

---

## 0. Tóm tắt trong một trang

**Câu hỏi lớn của 3 ngày:** Hou25 chấm từng từ trong list rồi đưa top-50 vào prompt, và làm mất thông tin "từ nằm ở giây nào". **Nếu scorer giữ lại vị trí thì có lợi gì không, ở cả phía scorer và phía ASR?**

**Câu trả lời ngắn: có, ở cả hai phía.**

| # | Câu hỏi | Trả lời (có số) | Độ chắc |
|---|---|---|---|
| 1 | Hou25 thật sự yếu cỡ nào nếu đo kỹ hơn top-50? | Ở top-50 mọi hệ đều 95–98%, nhìn như ngang nhau. Đo "từ đúng có đứng đầu không" (Sạch@1) thì Hou25 chỉ **72.6**, scorer âm vị 87.5 | Chắc |
| 2 | Hou25 sai vì đâu? | Vì **từ nhiễu bị đẩy lên**, không phải vì từ đúng bị kéo xuống. Báo nhầm tăng theo độ dài câu: 1.4 → 12.4 từ/câu. 96% từ vượt lên là từ khác hẳn, không phải từ nghe giống | Chắc về hướng |
| 3 | Giả thuyết của Hoàng (đọc theo vệt chéo) có đúng không? | **Đúng, khi train lại.** Sạch@1 73.3 → 82.9 (+9.5), định vị được 98.9% | Chắc |
| 4 | Nền scorer nên là gì? | **Âm vị**, không phải TTS. Hou25 sửa tốt nhất (82.9) vẫn thua âm vị (87.5), chậm hơn ~200 lần, và không hơn trên list từ nghe giống | Chắc |
| 5 | Khớp tự do và khớp đúng thứ tự có bù nhau không? | **Có.** Cộng điểm hai kiểu: +2.6, trong khi cộng hai model cùng kiểu chỉ +1.1 | Chắc về hướng |
| 6 | Dạy vị trí lúc train có giúp không? | **Có**, nhất là nhánh khớp đúng thứ tự: 85.8 → 88.3 | Chắc |
| 7 | Scorer đề xuất (một model, hai nhánh, dạy vị trí) đạt bao nhiêu? | **89.3** (Hou25 72.6). Cộng điểm hai model riêng: 90.8. Định vị sai 0.02 s | Chắc |
| 8 | Có vị trí thì lọc list được không? | Được. 2000 từ → ~2 từ/câu, precision 91%, giữ 91% từ đúng. Top-50 của Hou25: precision 3–4% | Chắc |
| 9 | Đưa vào Whisper, không prompt, có thắng Hou25 không? | **Thắng về B-WER** (lỗi ở từ hiếm): 7.53 so với 9.69 trên test-other, có ý nghĩa thống kê. **WER tổng chỉ ngang** (3.14 so với 3.39, chưa có ý nghĩa) | B-WER chắc; WER chưa |
| 10 | Whisper có cần vị trí không? | **Cần.** Cùng list, đẩy đúng lúc tốt hơn đẩy mọi lúc 0.42 WER, có ý nghĩa | Chắc |
| 11 | Scorer tốt hơn có giúp prompt không? | **Không.** Scorer tốt hơn 17 điểm mà prompt top-50 vẫn chỉ ngang Hou25 → kênh prompt đã bão hoà | Chắc (1 model ASR) |

**Ba điểm còn yếu, phải nói thật với thầy:**
1. WER tổng khi không prompt chưa thắng có ý nghĩa (−0.17 đến −0.25).
2. Khi audio **không có** từ nào trong list, mọi cách đều làm WER tệ đi ~0.25. Scorer chưa "dám nói không".
3. Hou25 ở đây là **bản nhóm tái hiện** (2 epoch), chưa phải checkpoint của Hoàng.

**Hướng đi tiếp (chi tiết mục 6):** scorer ra **(từ, xác suất, vị trí)**, rồi đẩy logit đúng lúc trong lúc decode. Ưu tiên: (1) dạy scorer nói "không có từ nào", (2) nâng nhóm từ Whisper nghe sai, (3) một model đuổi kịp bản cộng điểm, (4) đủ bảng N100–2000 + Hou25 của Hoàng + Qwen3-ASR.

---

## 1. Bối cảnh: bài toán và cách Hou25 làm

### 1.1 Bài toán

Một câu nói + một **bias list** (ví dụ 2000 từ hiếm, trong đó chỉ 1–3 từ thật sự có trong câu, còn lại là **từ nhiễu**). Mục tiêu: ASR viết đúng các từ hiếm có trong câu, mà **không** viết bừa các từ nhiễu.

Ví dụ thật (test-other, câu 4198-61336-0012): câu có từ `pileser`. List 2000 từ có `pileser` và 1999 từ khác như `flauve`, `emperour`, … Whisper không có list thường viết sai `pileser`.

### 1.2 Hou25 làm thế nào (mốc so sánh)

```
mỗi từ trong list ─TTS→ audio giả ─encoder→ đặc trưng từ
câu nói ─────────────────encoder→ đặc trưng câu
        → cross-attention → bản đồ S1 (thời gian câu × thời gian từ)
        → CNN → LẤY TRUNG BÌNH cả bản đồ → 1 con số p
        → chọn top-50 → nhét vào PROMPT của Whisper
```

Hou25 **làm mất vị trí ở hai chỗ**:
- Lấy trung bình cả bản đồ thành một số: không còn biết từ nằm ở giây nào.
- Đưa top-50 vào prompt: Whisper nhận 50 từ, không biết từ nào ở đâu, 48 trong số đó là nhiễu.

Prompt còn có giới hạn riêng: Whisper chỉ giữ 223 token cuối; đổi thứ tự từ là đổi kết quả; mỗi model ASR có kiểu prompt khác nhau.

### 1.3 Ý tưởng của nhóm

Bản đồ S1 của Hou25 thật ra **có** thông tin vị trí: Hoàng thấy có một **vệt chéo** đúng ở đoạn từ được nói (F10, F14). Vậy:
- Phía scorer: giữ vị trí lại → chấm điểm tốt hơn, và lọc được nhiễu (hai từ không thể nằm cùng một chỗ).
- Phía ASR: biết từ nằm ở giây 3.0–3.6 thì chỉ "nhắc" Whisper đúng lúc nó đang nghe đoạn đó, không cần prompt.

---

## 2. Từ điển tên gọi (đọc trước khi xem bảng)

### 2.1 Các scorer

| Tên trong bảng | Là gì | Ra vị trí? |
|---|---|---|
| **Hou25** | Bản nhóm tái hiện paper Hou25 (TTS + cross-attention + CNN + trung bình), train 2 epoch trên clean-100 | Không tin được (44%) |
| **Hou25 đọc vệt chéo** (`hou25_diag`) | Giữ nguyên Hou25, chỉ đổi bước "trung bình cả bản đồ" thành "lấy vệt chéo điểm cao nhất", rồi train lại | Có |
| **Hou25 top-k** (`hou25_topk`) | Như trên nhưng lấy trung bình k% ô cao nhất (không cần thành vệt) | Không |
| **pmaxsim** | Scorer âm vị của nhóm. Từ → chuỗi âm vị (g2p). Bản đồ = âm vị × frame audio. Mỗi âm vị tìm frame giống nhất **ở bất kỳ đâu** rồi lấy trung bình (khớp tự do) | Có |
| **pmono** | Cùng bản đồ âm vị × frame, nhưng các âm vị phải khớp **đúng thứ tự và liền nhau** (quy hoạch động). Trả về luôn đoạn [bắt đầu, kết thúc] | Có, biên rất ổn |
| **`_loc` / "dạy vị trí"** | Khi train, từ đúng chỉ được tính điểm trong đoạn nó thật sự được nói (từ căn chỉnh MFA, ± 0.1 s) | – |
| **Cộng điểm** (`ens`) | Cộng logit của pmaxsim_loc + pmono_loc (hai model train riêng). Biên lấy từ pmono | Có |
| **pdual / pdual2** | **Một model, hai nhánh** (tự do + đúng thứ tự) train chung. `pdual` dùng chung một bản đồ; `pdual2` mỗi nhánh có bản đồ riêng | Có |
| **pdual2_dl** | pdual2 + dạy vị trí. **Đây là scorer đơn đề xuất** | Có |

Mọi scorer dùng encoder Whisper-large-v3 **đóng băng**, cửa sổ trượt 30 s.

### 2.2 Các cách đưa list vào Whisper

| Tên | Cách làm |
|---|---|
| **prompt top-50** | Như Hou25: 50 từ điểm cao nhất vào prompt |
| **đẩy không vị trí** | Cộng λ vào logit token của các từ được chọn ở **mọi bước** decode |
| **đẩy có vị trí** (gate boost) | Ở mỗi bước, đọc alignment heads của Whisper để biết decoder đang "nghe" giây nào. Chỉ cộng λ cho từ có đoạn chứa giây đó (± 0.05 s). Mỗi từ chỉ đẩy đến khi nó đã được viết đủ số lần |
| **prompt + đẩy có vị trí** | Cả hai cùng lúc |
| **Trần** | Biết đúng từ và đúng vị trí (MFA), rồi đẩy có vị trí. Mức tốt nhất cách đẩy này đạt được |

Ví dụ đẩy có vị trí: `pileser` được scorer đặt ở 3.0–3.6 s. Khi Whisper đang viết và alignment heads cho biết nó đang nghe giây 3.2, ta cộng 5 điểm vào token đầu của `pileser`. Ở giây 1.0 hay 5.0 thì không cộng gì.

---

## 3. Thước đo: vì sao không dùng "top-50" như Hou25

Hou25 báo **"từ đúng có lọt top-50 không"**. Ở mốc đó mọi hệ đều đạt 95–98%, khác biệt bị che. Nhưng top-50 nghĩa là ~48 từ nhiễu mỗi câu. Khi không dùng prompt mà đẩy logit, mỗi từ nhiễu bị đẩy vào là một lỗi tiềm năng. Nên cần đo **ở R nhỏ**.

| Thước đo | Nghĩa | Ví dụ |
|---|---|---|
| **Sạch@K** | % từ đúng có **ít hơn K từ nhiễu** đứng trên nó. Chỉ đếm nhiễu, không đếm từ đúng khác, để câu nhiều từ hiếm không bị phạt oan | Câu có `pileser` và `assyria`. Xếp hạng: `assyria`, `pileser`, `flauve`… → `pileser` vẫn Sạch@1 vì `assyria` là từ đúng |
| **Đúng@chỗ** | Tại đoạn audio của từ đúng, nó có đứng trên mọi từ nhiễu được đặt **cùng chỗ** không | Ở 3.0–3.6 s có `pileser` (0.8), `pillar` (0.4) → đúng |
| **Định vị** | Sai số tâm (giây) so với MFA; % tâm nằm trong đoạn thật ± 0.1 s | |
| **Báo nhầm theo độ dài câu** | Số từ nhiễu p ≥ 0.5 mỗi câu, chia theo câu < 5 s … ≥ 20 s | |
| **Nhóm từ Whisper nghe sai** | Sạch@1 chỉ tính trên các từ mà Whisper không context viết sai (986 lượt, test-other). Đây là nhóm **cần** scorer nhất | |
| **WER / U-WER / B-WER** | Lỗi trên mọi từ / chỉ từ thường / chỉ từ hiếm của reference. Scorer chính thức IS21 (sub 4, ins/del 3), chuẩn hoá Whisper | B-WER 9.69 → 7.53 nghĩa là lỗi ở từ hiếm giảm ~22% |
| **Chèn sai** | Số từ trong list bị viết vào mà reference không có | |

Khoảng tin cậy 95% ghi trong ngoặc vuông [a; b] là bootstrap theo câu. CI không chứa 0 = có ý nghĩa thống kê.

---

## 4. Các thí nghiệm, theo thứ tự câu hỏi

Kế hoạch đặt trước khi chạy (`PLAN-2011.md`) có 3 quyết định cần ra:
- **D1. Nền scorer** là Hou25 sửa lại hay âm vị?
- **D2. Whisper có cần vị trí không?**
- **D3. Đẩy logit thế nào cho hợp lý?**

Luật đặt trước: Hou25 sửa tốt nhất phải đạt Sạch@1 ≥ 85.7 (mức của pmaxsim lúc đó) thì mới giữ khung Hou25.

### Dòng thời gian

| Khi nào | Máy | Việc | Thí nghiệm |
|---|---|---|---|
| 13–14/11 | CPU | Phân tích lại checkpoint cũ: đuôi khó, R nhỏ, lỗi "sai chỗ" ở ASR | T0, E0, N6 |
| 14/11 tối | CPU | Lập roadmap E0–E5, viết code (cửa sổ trượt, pool, MonoSim, gate boost `--once`) | – |
| 14/11 đêm | GPU 0 + 1 | Cửa sổ trượt; train lại 3 bản Hou25; list nghe giống; train pmaxsim / pmono; đưa vào Whisper | E1, E2, E3, E4, E5, B |
| 15/11 sáng | thêm GPU | Dạy vị trí; cộng điểm; một model hai nhánh; chạy N100–2000 | E6, E7, E8, E9 |
| 15/11 | CPU | Báo cáo, commit, slide; xoá container `qwen-3.5` | – |

---

### T0. Scorer bỏ sót từ nào? (CPU, 13/11) [đo]

**Câu hỏi.** Những từ đúng không lọt nổi top-50 là từ gì? Do audio khó hay do mô hình?

**Làm gì.** Lấy checkpoint cũ (18/10), xem từ đúng nào bị sót khỏi top-50 (N2000), chia theo đặc điểm từ.

**Kết quả (test-other).**
- Hou25 sót 4.61%, pmaxsim 1.56%. **174 / 242** từ Hou25 sót thì pmaxsim vẫn bắt được → audio có tín hiệu, lỗi do mô hình.
- Từ bắt đầu **sau giây 30**: sót 83–100% ở **mọi** hệ. Lý do: encoder Whisper chỉ nhận 30 s, phần sau bị cắt.
- Từ khó với scorer cũng là từ khó với ASR: 94% số từ "sót ở mọi hệ" là từ Whisper nghe sai.
- Từ ngắn (< 0.25 s), từ có `'` / `-` (`t'other`, `adn't`): sót gấp 2–3 lần trung bình.

**Kết luận.** **Có một lỗi kỹ thuật chung (cắt 30 s) cần sửa trước mọi thứ; và nhóm từ Whisper nghe sai phải thành một thước đo riêng.**

---

### E0. Đo lại bằng R nhỏ: khoảng cách thật giữa Hou25 và âm vị (CPU, 14/11) [đo]

**Câu hỏi.** Top-50 có che mất khác biệt không? Hou25 sai theo kiểu nào?

**Giả thuyết.** Top-50 quá dễ. Giả thuyết cũ của nhóm: "lấy trung bình làm loãng từ đúng".

**Kết quả** (test-other, checkpoint cũ):

| | Sạch@1 | Sạch@5 | Sạch@10 | Sạch@50 (≈ top-50) |
|---|---|---|---|---|
| Hou25 | **72.8** | 87.2 | 90.9 | 95.4 |
| pmaxsim | **85.7** | 95.1 | 96.7 | 98.4 |

**Rút ra được gì**
- Ở mốc 50 cách nhau 3 điểm; ở mốc 1 cách nhau **13 điểm**.
- Khi từ đúng bị vượt, từ vượt lên là **từ khác hẳn** (lệch > 2 âm vị) ở 96% số lượt; gần âm chỉ 3%. Ví dụ: `jeered` bị `appropriately` vượt.
- 76% báo nhầm của Hou25 **không giống đoạn âm nào** trong câu (pmaxsim: 40%).
- Báo nhầm của Hou25 tăng theo độ dài câu: 1.4 → 11.4 từ/câu (pmaxsim 0.5 → 1.5).

**Kết luận.** **Hou25 thua vì từ nhiễu bị đẩy lên (nhất là ở câu dài), ngược với giả thuyết cũ "từ đúng bị loãng".** [diễn giải: CNN phản ứng với mảng "giống vệt" bất kỳ; câu dài có nhiều mảng ngẫu nhiên hơn.]

**Caveat.** List IS21 ít từ gần âm, nên "96% khác hẳn" chưa nói được gì về từ nghe giống → cần E4.

---

### N6. Phía ASR: có list mà không có vị trí thì Whisper có viết từ vào **sai chỗ** không? (CPU, 14/11) [đo]

**Câu hỏi.** Câu có nhiều từ hiếm, prompt chỉ cho list mà không nói từ nào ở đâu. Whisper có nhầm chỗ không?

**Kết quả** (test-other, % lượt từ hiếm):
- Viết từ hiếm này vào chỗ của từ hiếm khác: **≤ 0.36%**. Gần như không.
- Nhưng **viết thêm** từ ở chỗ không có thì nhiều: prompt chỉ 1 từ → Whisper lặp từ đó ở 68.5% câu; đẩy không vị trí → 24.4% (câu 2–4 từ hiếm).
- Đẩy có vị trí (MFA): chèn gần như 0, viết đúng cao nhất (91.6–94.7%).

**Kết luận.** **Whisper tự đặt từ đúng chỗ vì nó nghe audio. Vị trí cần thiết không phải để "đặt đúng chỗ" mà để "không đẩy ở chỗ không có".**

---

### E1. Cửa sổ trượt 30 s [đo]

**Đây là sửa lỗi, không phải thí nghiệm.** Encoder chạy theo cửa sổ 30 s, bước 25 s, rồi ghép lại.

**Kết quả.** Từ bắt đầu sau giây 30 bị sót: trước 5–6 / 6, sau **0 / 6**. Mọi số từ đây trở đi đều chạy có cửa sổ trượt.

---

### E2. Kiểm giả thuyết của Hoàng: đọc theo vệt chéo thay cho trung bình (GPU, 14/11 đêm) [đo]

**Câu hỏi.** Nếu Hou25 không lấy trung bình cả bản đồ mà đọc theo vệt chéo, rồi **train lại**, thì có hết báo nhầm ở câu dài không?

**Giả thuyết** (Hoàng F14 + E0). Vệt chéo đúng chỗ là tín hiệu thật; trung bình cả bản đồ làm mảng ngẫu nhiên ở câu dài cũng được tính. Đọc vệt → ít báo nhầm, và ra luôn vị trí.
Cũng có thể **ngược lại**: lấy "chỗ cao nhất" trên nhiều vị trí dễ gặp một ô cao do may rủi. Vì vậy có thêm arm top-k để phân biệt "đọc cục bộ" với "đọc theo hình vệt".

**Setup.** 3 bản Hou25 train cùng lượt (cùng dữ liệu, seed, lr 5e-5, 2 epoch), chỉ khác bước gộp. Chấm đủ 2939 câu test-other, N2000.

**Kết quả**

| Cách gộp | Sạch@1 | Δ [CI 95%] | Báo nhầm, câu ≥ 20 s | Sạch@1 từ Whisper nghe sai | % tâm trúng |
|---|---|---|---|---|---|
| Trung bình (Hou25 gốc) | 73.3 | – | 6.1 | 46.3 | 79.3 |
| Top-k (đối chứng) | 73.8 | +0.5 [−0.9; +1.9] | 8.0 | 47.2 | 3.6 |
| **Vệt chéo** | **82.9** | **+9.5 [+7.9; +11.2]** | **1.9** | **60.9** | **98.9** |
| (tham chiếu) pmaxsim | 87.5 | +14.2 | 1.3 | 64.5 | 99.0 |

**Rút ra được gì**
- Vệt chéo: +9.5 điểm, báo nhầm ở câu dài giảm 3 lần, và cùng một model định vị được 98.9%. **Khớp giả thuyết.**
- Top-k không giúp, báo nhầm còn tăng. → Thứ có ích là **hình dạng vệt** (đúng thứ tự, liền nhau), không phải "đọc cục bộ" nói chung.
- Trước đây (A1, 09/11) chỉ đổi cách gộp lúc test mà không train lại thì không thấy lợi: CNN phải được train cho cách đọc mới.
- Vẫn thua pmaxsim 4.6 điểm → **không đạt luật D1** (≥ 85.7).

**Kết luận.** **Giả thuyết của Hoàng đúng. Nhưng Hou25 sửa tốt nhất vẫn kém âm vị, nên nó là baseline mạnh ("Hou25 + vị trí") trong paper, không phải nền.**

**Caveat.** Bản tái hiện 2 epoch, chưa phải checkpoint 3 epoch của Hoàng.

---

### E3. Train với nhiều nhiễu hơn (GPU, 14/11 đêm) [đo, chỉ dev]

**Câu hỏi.** Mỗi từ đúng lúc train chỉ đi kèm 4 từ nhiễu, trong khi test có 2000. Tăng lên thì có học cách nén điểm từ lạ tốt hơn không?

**Kết quả** (dev-other 500 câu, sau 3 epoch): 16 nhiễu + BCE cân bằng cho Sạch@1 **82.1**, kém bản 4 nhiễu (87.7).

**Kết luận.** **Không giúp ở dạng này → bỏ.** (Kết quả âm, ghi lại để không thử lại.)

**Caveat.** Chạy trên pmaxsim thay vì Hou25 như kế hoạch; chỉ có số dev, sai số ~±1.5.

---

### E4. Từ nghe giống: TTS có hơn âm vị không? (GPU) [đo]

**Câu hỏi.** Hou25 dùng audio TTS, nên có thể phân biệt từ nghe giống tốt hơn âm vị. Nếu đúng thì nên làm 2 tầng: âm vị lọc, TTS xếp lại.

**Setup.** List `hard_N2000`: thay nhiễu bằng nhiều từ gần âm với từ đúng.

**Kết quả** (test-other): Sạch@1 Hou25 **51.4**, pmaxsim **61.2** (+9.7 [+7.9; +11.4]). Trên list này 21% số lần từ đúng bị vượt là do từ gần âm (list thường: 3–7%).

**Kết luận.** **TTS không cho lợi thế với từ nghe giống. Bỏ phương án 2 tầng; nền là âm vị.** → Chốt **D1**.

---

### E5. Ép khớp đúng thứ tự (pmono) có tốt hơn khớp tự do (pmaxsim)? (GPU) [đo]

**Câu hỏi.** E2 cho thấy "hình vệt" quan trọng. Vậy với âm vị, ép các âm vị khớp đúng thứ tự có xếp hạng tốt hơn không?

**Setup.** Cả hai train 6 epoch trên clean-100, chọn ckpt theo Sạch@1 trên dev-other.

**Kết quả** (test-other)

| | Sạch@1 | List nghe giống | Báo nhầm câu ≥ 20 s | Precision ở p ≥ 0.5 | Sai số tâm |
|---|---|---|---|---|---|
| pmaxsim (tự do) | **87.5** | **64.7** | 1.3 | 75.8 | 0.05 s |
| pmono (đúng thứ tự) | 85.8 (−1.7) | 60.5 (−4.2) | **0.9** | **82.3** | **0.02 s** |

**Rút ra.** Ép thứ tự **không** làm xếp hạng tốt hơn. Đổi lại pmono ít báo nhầm hơn, precision cao hơn, và biên rất ổn (gần như không bao giờ mở sớm).

**Kết luận.** **Mỗi kiểu khớp mạnh một mặt: tự do để không sót, đúng thứ tự để loại nhiễu và cho biên.** → dẫn tới E6.

---

### E6. Hai kiểu khớp có bù nhau không? (CPU, cộng điểm, 15/11) [đo]

**Câu hỏi.** Cộng điểm pmaxsim + pmono có lợi hơn mức "gộp hai model nào cũng lợi" không?

**Setup.** Cộng logit, không train thêm. **Đối chứng:** cộng hai bản pmaxsim (cùng kiểu); cộng pmaxsim với pmaxsim train 960h (gấp 9.6 lần dữ liệu).

**Kết quả** (test-other, cùng script)

| | Sạch@1 |
|---|---|
| pmaxsim riêng | 87.2 |
| pmaxsim + pmaxsim (cùng kiểu) | 88.3 (+1.1) |
| pmaxsim + pmaxsim 960h | 89.7 |
| **pmaxsim + pmono** | **89.8** (+2.6) |

List nghe giống: +4.0. Sau tranh nhau tại chỗ ở p ≥ 0.5: precision **91.2%** (pmaxsim riêng 85.0%).

**Rút ra.** ~+1.1 là lợi chung của việc gộp; ~**+1.5 còn lại do hai cách khớp nhìn khác nhau**. Thêm nhánh đúng thứ tự lợi ngang thêm 9.6 lần dữ liệu train.

**Kết luận.** **Bù nhau thật → kiến trúc đề xuất: một model, hai nhánh.**

---

### E7. Dạy vị trí lúc train (GPU, 15/11 sáng) [đo]

**Câu hỏi.** Nếu lúc train chỉ cho phép từ đúng được tính điểm ở đúng đoạn nó được nói, scorer có học tốt hơn không?

**Kết quả** (test-other)

| | Sạch@1 | Từ Whisper nghe sai | Đúng@chỗ (câu ≥ 5 từ hiếm) |
|---|---|---|---|
| pmaxsim → + dạy vị trí | 87.5 → 88.3 (+0.9 [+0.2; +1.5]) | 64.5 → 65.9 | 95.8 → 95.7 |
| pmono → + dạy vị trí | 85.8 → **88.3** (+2.5) | 64.4 → **69.2** | 96.3 → **97.5** |
| Cộng điểm hai bản có dạy vị trí | **90.8** | – | – |

**Kết luận.** **Dạy vị trí giúp cả hai, giúp nhánh đúng thứ tự nhiều nhất, và giúp nhiều nhất ở nhóm từ Whisper nghe sai.**

---

### E8. Một model hai nhánh (pdual / pdual2) (GPU, 15/11) [đo]

**Câu hỏi.** Bản cộng điểm cần hai model. Gộp thành một model hai nhánh train chung thì được bao nhiêu?

**Kết quả** (test-other)

| | Sạch@1 | List nghe giống | Từ Whisper nghe sai | test-clean |
|---|---|---|---|---|
| pmaxsim (mốc) | 87.5 | 64.7 | 64.5 | 93.4 |
| pdual (chung bản đồ) + dạy vị trí | 88.3 | 66.1 | 68.8 | 94.2 |
| **pdual2 (bản đồ riêng) + dạy vị trí** | **89.3** (+1.8 [+0.9; +2.6]) | **67.7** (+3.0) | **70.9** | **94.7** |
| (cộng điểm 2 model riêng) | 90.8 | 69.6 | – | – |

**Rút ra.**
- Hai nhánh cần **bản đồ riêng** mới bù nhau; chung bản đồ thì chỉ ngang bản đơn.
- Một model lấy được hơn nửa phần lợi của bản cộng điểm (còn kém 1.5).
- Điểm dev nhảy mạnh giữa các epoch (84.8–90.2) → chọn ckpt còn nhiễu.

**Kết luận.** **`pdual2_dl` là scorer đơn tốt nhất hiện có: 89.3 Sạch@1, định vị 0.02 s, chấm 4–7 ms/câu (Hou25 ~1200 ms).**

---

### E9. Có vị trí thì lọc list được không? "Tranh nhau tại chỗ" (CPU, không train) [đo]

**Câu hỏi.** Hai từ không thể cùng nằm ở một chỗ. Cho chúng tranh nhau thì loại được bao nhiêu nhiễu?

**Cách làm.** Hai từ có đoạn chồng nhau (IoU > 0.3) thì chỉ giữ từ điểm cao hơn; sau đó lấy từ có p ≥ τ.

**Kết quả** (test-other, N2000)

| | Giữ từ đúng | Nhiễu / câu | Precision |
|---|---|---|---|
| Hou25 top-50 (cách của paper) | 95.4% | ~48 | ~3–4% |
| pmaxsim, τ 0.5, trước → sau tranh nhau | 91.2 → 89.9 | 0.71 → 0.40 | 69.5 → 80.0 |
| Cộng điểm có dạy vị trí, τ 0.5 | 91.1 | 0.16 | **91.3** |

Ví dụ (hình `tranh_nhau.pdf` trong slide):
- Câu 7018-75789-0016: mọi từ nhiễu trùng chỗ với từ đúng đều bị loại.
- Câu 4198-61336-0012 (chỗ còn yếu): `flauve` nghe như từ thường "fall", `emperour` nghe như "empire". Nhiễu khớp với **một từ thường** trong câu, không trùng chỗ với từ hiếm nào, nên luật tranh nhau không loại được.

**Kết luận.** **Vị trí biến bài toán "1 đúng trong 2000" thành nhiều bài nhỏ "1 đúng trong vài từ cùng chỗ". Từ 2000 từ còn ~2 từ/câu với precision ~91%.**

**Caveat.** Luật IoU đặt bằng tay; chưa có ô "không từ nào ở đây" → chưa xử lý được nhiễu khớp từ thường.

---

### B. Đưa vào Whisper: có thắng Hou25 không, và có cần vị trí không? (GPU, 14–15/11) [đo]

**Câu hỏi.**
1. Scorer có vị trí + đẩy logit, không prompt, có thắng Hou25 prompt top-50 không?
2. Cùng list, đẩy đúng lúc có hơn đẩy mọi lúc không? (**D2**)
3. Audio không có từ nào trong list thì có viết bừa không?

**Setup.** Tham số chọn trên dev-other 400 câu, **không chọn trên test**: giữ từ p ≥ 0.3 sau tranh nhau (~2.3 từ/câu), λ = 5 (không prompt) hoặc 3 (có prompt), biên 0.05 s. Greedy, có decode lại khi câu bị lặp (tỉ lệ nén > 2.4, như openai-whisper), áp cho mọi hệ.

**Kết quả** (test-other, N2000, scorer = cộng điểm có dạy vị trí)

| | Từ/câu | WER | U-WER | B-WER | Chèn sai |
|---|---|---|---|---|---|
| Không context | 0 | 4.25 | 2.63 | 17.92 | 13 |
| **Hou25: prompt top-50** | 50 | 3.39 | 2.65 | 9.69 | 39 |
| Scorer mới: prompt top-50 | 50 | 3.43 | 2.65 | 10.03 | 49 |
| Scorer mới: đẩy **không** vị trí | ~2.3 | 3.56 | 3.13 | 7.17 | 66 |
| **Scorer mới: đẩy có vị trí, không prompt** | ~2.3 | **3.14** | **2.62** | 7.53 | 49 |
| Scorer mới: prompt + đẩy có vị trí | 50 + ~2.3 | 3.22 | 2.80 | **6.76** | 113 |
| Trần (đúng từ + vị trí MFA) | – | 2.99 | 2.63 | 5.99 | 7 |

test-clean: Hou25 1.66 / 5.15 (WER / B-WER); đẩy có vị trí không prompt 1.59 / 3.93; prompt + đẩy 1.47 / 3.12; trần 1.53 / 2.77.

So với Hou25, CI 95%:

| | test-other | test-clean |
|---|---|---|
| Không prompt: ΔB-WER | **−2.16 [−2.89; −1.54]** | **−1.22 [−1.76; −0.68]** |
| Không prompt: ΔWER | −0.25 [−0.47; +0.03] chưa có ý nghĩa | −0.07 [−0.22; +0.10] |
| Không prompt: ΔU-WER | −0.03 (không đổi) | +0.07 (không đổi) |
| Prompt + đẩy: ΔWER | **−0.17 [−0.29; −0.06]** | **−0.19 [−0.27; −0.11]** |
| Prompt + đẩy: ΔU-WER | +0.15 (từ thường tệ hơn) | +0.04 |

**Rút ra được gì**
- **Không prompt vẫn thắng Hou25 về B-WER** ở cả hai tập, có ý nghĩa; từ thường không bị ảnh hưởng (ΔU-WER ≈ 0).
- WER tổng chỉ thấp hơn 0.25, **chưa có ý nghĩa**. Lý do [diễn giải]: từ hiếm chỉ chiếm phần nhỏ số từ.
- **Cần vị trí (D2):** cùng list, đẩy có vị trí so với không: ΔWER **−0.42 [−0.51; −0.34]**. Đẩy không vị trí làm từ thường hỏng (U-WER 3.13) vì nó chèn từ hiếm vào chỗ khác.
- **Kênh prompt đã bão hoà:** scorer tốt hơn 17 điểm Sạch@1 mà prompt top-50 vẫn chỉ ngang Hou25 (10.03 so với 9.69). Muốn lấy lợi của scorer tốt thì phải qua vị trí + đẩy logit.
- Prompt + đẩy có B-WER tốt nhất nhưng chèn sai gấp đôi (113) và U-WER tệ hơn.

**Audio không có từ nào** (list `missing_N2000`): WER 4.49 so với không context 4.25 (test-other); 2.31 so với 2.16 (test-clean). **Vẫn tệ đi 0.15–0.25.**

**Còn cách trần bao xa?** (dev-other, B-WER): trần 3.99 → đúng từ + vị trí do scorer đoán 5.61 → list do scorer chọn + vị trí scorer 7.22. Vậy **vị trí sai tốn ~1.6 điểm, list sai tốn ~1.6 điểm**, mỗi phần một nửa.

**Kết luận.** **Thắng Hou25 về B-WER mà không cần prompt, và Whisper cần vị trí. Chưa thắng WER tổng có ý nghĩa; chưa xử lý được audio không có từ nào.**

**Caveat.** Một model ASR (Whisper), greedy; tham số chọn trên 400 câu; mốc Hou25 trên `missing_N2000` chưa chạy lại với chống lặp (bản cũ: 4.53).

---

### E10. Theo cỡ list N100–2000 (một model `pdual2_dl`, 15/11) [đo]

| List | Sạch@1 | Precision | WER | B-WER |
|---|---|---|---|---|
| N100 | 98.2 | 97.8% | 2.96 | 7.14 |
| N500 | 94.1 | 89.2% | 3.04 | 7.37 |
| N1000 | 92.6 | 81.4% | 3.08 | 7.47 |
| N2000 | 89.0 | 69.9% | 3.22 | 7.65 |

(test-other, đẩy có vị trí, không prompt. Không context: 4.25 / 17.92.)

**Rút ra.** List càng dài thì precision giảm mạnh (98 → 70%), nhưng B-WER chỉ tăng nhẹ (7.14 → 7.65): **cách làm bền với list lớn**.

**Caveat.** **Chưa có Hou25 ở N100–1000** trong cùng pipeline, nên chưa so được ở các mức này.

---

## 5. Tổng hợp

### 5.1 Giả thuyết đặt trước so với thực tế

| Giả thuyết (PLAN-2011) | Thực tế | |
|---|---|---|
| Top-50 che khác biệt | 95.4 vs 98.4 ở top-50; 72.8 vs 85.7 ở Sạch@1 | ✅ |
| Hou25 sai vì "trung bình làm loãng từ đúng" | Sai vì **từ nhiễu bị đẩy lên** ở câu dài | ❌ (đổi cách nhìn) |
| S1: đọc theo vệt + train lại giảm báo nhầm ở câu dài | +9.5 Sạch@1, báo nhầm câu dài giảm 3 lần | ✅ |
| Hou25 sửa đuổi kịp âm vị (≥ 85.7) | 82.9 | ❌ → nền âm vị |
| S2: nhiều nhiễu khi train giúp | 82.1 vs 87.7 (dev) | ❌ |
| S3: TTS phân biệt từ nghe giống tốt hơn | âm vị hơn 9.7 điểm | ❌ → bỏ 2 tầng |
| S4: cửa sổ trượt sửa lỗi 30 s | 0 / 6 sót | ✅ |
| Ép thứ tự làm xếp hạng tốt hơn | −1.7, nhưng bù cho khớp tự do (+2.6 khi gộp) | ⚠️ một phần |
| D2: Whisper cần vị trí | ΔWER −0.42, có ý nghĩa | ✅ |
| Đẩy có vị trí thắng prompt Hou25 | B-WER thắng; WER tổng chưa | ⚠️ một phần |
| Chèn sai trên `missing_N` gần bằng không context | vẫn tệ hơn 0.15–0.25 WER | ❌ |

### 5.2 Ba quyết định đã ra

| Quyết định | Kết quả |
|---|---|
| **D1. Nền scorer** | **Âm vị, hai nhánh (tự do + đúng thứ tự), dạy vị trí.** Hou25 + vệt chéo làm baseline mạnh |
| **D2. Whisper cần vị trí?** | **Cần** |
| **D3. Đẩy thế nào?** | Lọc bằng tranh nhau tại chỗ + ngưỡng p ≥ 0.3; λ = 5, biên 0.05 s, mỗi từ đẩy một lần; có decode lại khi lặp. Đây là cấu hình chọn trên dev, chưa phải tối ưu |

### 5.3 Câu chuyện cho paper [diễn giải]

```
Hou25 bỏ vị trí ──► đo ở R nhỏ thấy thua 15 điểm, vì báo nhầm ở câu dài
       │
       ├─ phía scorer: giữ vị trí (vệt chéo / khớp đúng thứ tự / dạy vị trí)
       │      → 73 → 89 Sạch@1; lọc 2000 từ còn ~2 từ/câu, precision 91%
       │
       └─ phía ASR: prompt đã bão hoà; đẩy logit đúng lúc
              → B-WER 9.69 → 7.53, không cần prompt; cùng list, đúng lúc > mọi lúc
```

Đóng góp có thể đặt ở: (a) scorer hai nhánh có vị trí; (b) đẩy logit theo vị trí, không cần prompt; (c) cả hai thành một hệ. Đây là câu hỏi 1 xin ý kiến thầy.

### 5.4 Đối chiếu với các ý thầy/nhóm đã nêu

| Ý | Trạng thái |
|---|---|
| Whisper có cần thông tin vị trí? | ✅ Đã trả lời: cần (ΔWER −0.42) |
| Khảo sát cách điều chỉnh logit, cách nào general cho ASR khác | ⚠️ Mới tra ~10 lượt (slide D4). Cách hiện tại cần alignment heads, nên với SpeechLLM (Qwen3-ASR) phải suy ra giây decoder đang nghe theo cách khác. **Chưa làm** |
| Tránh viết bias word khi audio không có | ❌ Chưa đạt: `missing_N` vẫn +0.25 WER. Là việc ưu tiên 1 |
| Can thiệp đẩy xác suất lên hợp lý | ✅ Bản đầu: gate theo vị trí + đẩy một lần + chống lặp |
| Tìm gap của scorer + vị trí; chứng minh bằng ranking đúng top và vị trí đúng | ✅ Sạch@1, Đúng@chỗ, sai số định vị đều đã có |
| Adaptation cho một domain | Chưa làm, để 01/2027 |

---

## 6. Còn thiếu và bước tiếp theo

### 6.1 Giới hạn hiện tại

| Giới hạn | Ảnh hưởng | Cách gỡ |
|---|---|---|
| Hou25 là bản tái hiện 2 epoch | Có thể bị chê "baseline yếu" | Xin checkpoint 3 epoch của Hoàng, chạy lại cùng pipeline |
| WER tổng chưa thắng có ý nghĩa khi không prompt | Không được nói "thắng WER" | Giảm chèn sai (6.2 #1), nâng từ nghe sai (#2) |
| Audio không có từ thì tệ đi 0.25 | Reviewer sẽ hỏi ngay | 6.2 #1 |
| Một model kém bản cộng điểm 1.5 | Bản cộng điểm tốn 2 model | 6.2 #3 |
| Chỉ Whisper, chỉ N2000 cho Hou25 | Chưa nói được "general" | 6.2 #4, #5 |
| Tham số chọn trên 400 câu dev | Có thể chưa tối ưu, nhưng không rò test | Dev lớn hơn |

### 6.2 Việc tiếp theo, theo thứ tự ưu tiên

| # | Việc | Vì sao (số nào) | Đạt khi | Ước tài nguyên |
|---|---|---|---|---|
| 1 | **Ô "không từ nào" được train** + thay luật IoU bằng tranh nhau học được | `missing_N` +0.25 WER; câu B `flauve` ~ "fall" | WER trên `missing_N2000` ≤ không context + 0.05; precision tăng ở cùng recall | Train lại pdual2, ~1 đêm 1 GPU |
| 2 | **Nhóm từ Whisper nghe sai** (Sạch@1 70.9) và list nghe giống (67.7): nhiều cách phát âm, nhiễu gần âm đặt đúng chỗ khi train | Nhóm quyết định B-WER; một nửa khoảng cách tới trần do list | +3 điểm trên nhóm này | 1–2 đêm |
| 3 | **Một model ≥ 90.5** (đuổi bản cộng điểm): loss riêng từng nhánh, trung bình ckpt, dev lớn hơn | 89.3 vs 90.8; dev nhảy 84.8–90.2 | Sạch@1 ≥ 90.5 | 1 đêm |
| 4 | **Train 960h** | ckpt cũ: +1.8 Sạch@1 | Bản cuối cho paper | ~2 đêm, cache feature thêm |
| 5 | **Bảng chính:** đủ N100–2000 cả 2 tập; Hou25 của Hoàng; Hou25 + vệt chéo làm baseline mạnh | Hiện Hou25 chỉ có N2000 | Bảng 1 (scorer) + Bảng 2 (ASR) đủ ô | ~1 ngày GPU decode |
| 6 | **Qwen3-ASR:** suy ra giây decoder đang nghe từ chữ đã viết (không có alignment heads) | Câu hỏi "general cho ASR khác" | Có một dòng Qwen3-ASR trong Bảng 2 | Cần thiết kế, duyệt trước khi code |
| 7 | Khảo sát cách điều chỉnh logit (shallow fusion, trie boost, CTC-WS, PTC-Bias, Liu 2508.17796) | Ý của thầy; mốc so sánh | Bảng so sánh + chọn 1–2 baseline | CPU, đọc paper |

### 6.3 Mốc thời gian (đến hạn 09/02/2027)

| Mốc | Việc |
|---|---|
| 20/11 | Gặp thầy: chốt đóng góp chính, mốc so sánh, phạm vi model |
| đến 20/12 | Scorer cuối (#1–#4) |
| đến 17/01 | Đủ bảng (#5), Qwen3-ASR (#6), baseline logit (#7) |
| đến 05/02 | Bản nộp |

### 6.4 Ba câu cần thầy quyết (20/11)

1. **Đóng góp chính đặt ở đâu:** scorer hai nhánh có vị trí, hay đẩy logit theo vị trí không cần prompt, hay gộp cả hai thành một hệ?
2. **Mốc so sánh:** chỉ Hou25, hay thêm PTC-Bias và boost không gate kiểu Liu 2508.17796?
3. **Phạm vi:** Whisper + Qwen3-ASR có đủ cho Interspeech không?

---

## Phụ lục

### A. File và lệnh

| Thứ | Ở đâu |
|---|---|
| Bảng đầy đủ + CI | `scorer/analysis/REPORT-1115.md` |
| Phân tích CPU | `SMALL_R.md` (E0, N6, N7), `TAIL.md` (T0) |
| Code scorer | `scorer/models.py` (`Hou25 pool`, `MaxSim`, `MonoSim`, `DualSim`), `scorer/training/train.py` (`--loc`, `--cache`), `scorer/inference/rank.py` (`--win`, `--span`) |
| Đo | `scorer/analysis/eval_e.py`, `nms_at_place.py`, `ensemble_cpu.py`, `build_ens_lists.py` |
| Đẩy logit vào Whisper | `bench/gate_boost.py` (`--once`, `--fallback`, `--boost_ctx`) |
| So CI | `bench/compare.py` |
| Script chạy GPU | `scorer/runs/2026-11-14/run_e.sh`, `scripts/` |
| Output, ckpt (không có trong git, ~11 GB local) | `scorer/runs/2026-11-14/out/`, `ckpt/` |
| Trên server | `~/model_tmp/` 100 GB (feature cache 46 GB, ckpt); container đã xoá |
| Commit | `bc939b1` (code + kết quả), `7a2bfbd` (kế hoạch slide), `3469ad1` (slide) trên nhánh `scorer-vi-tri-s1`, chưa push |

### B. Bẫy kỹ thuật đã gặp (để lần sau tránh)

- Script `.sh` viết trên Windows có CRLF → phải `sed -i 's/\r$//'` trước khi chạy trên server.
- `train.py` nhiều arch bỏ qua `--name` → dùng `--tag`, nếu không sẽ ghi đè ckpt cũ.
- HF `openslr/librispeech_asr`: dev-clean tên là `validation.clean`.
- Decode chậm vì tranh luồng CPU → `OMP_NUM_THREADS=2`.
- Greedy Whisper lặp chữ khi bị đẩy → `--fallback`.
- Đệm số 0 ở cuối làm vệt chéo lệch về cuối câu → mặt nạ trong `diag_sum`.
