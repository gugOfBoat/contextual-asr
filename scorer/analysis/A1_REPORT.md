# A1: Đọc bản đồ điểm của Hou25 theo cách khác thì có chọn từ tốt hơn, và có định vị được từ không?

> Code: `scorer/analysis/a1_pool.py`, `a1_s1.py`, `a1_eval.py`. Số liệu gọn: `RESULTS.md` mục A1. Kế hoạch: `PLAN.md` P1.2.
> Cập nhật: 09/11/2026. Phạm vi: scorer Hou25 bản tái hiện của nhóm (2 epoch), LibriSpeech test-other, list IS21 N2000. Không train lại.

## Tóm tắt

| Câu hỏi | Câu trả lời (có số) | Độ chắc |
|---|---|---|
| Q2a. Giữ nguyên checkpoint, chỉ đổi cách gộp bản đồ điểm thành 1 điểm thì có chọn từ đúng tốt hơn không? | **Không.** Cách gốc (lấy trung bình) cao nhất: R#10 88,83%. Đọc theo vệt chéo: 85,27% (−3,56, CI [−4,51; −2,71]). Đọc trong cửa sổ: 84,72% (−4,12). Lấy ô cao nhất: 77,25% (−11,59). Thua ở cả xếp hạng trong câu lẫn ngưỡng chung (AUC gộp 0,983 so với 0,976 / 0,979 / 0,892). | Chắc, với checkpoint này: mọi khoảng tin cậy đều nằm hẳn dưới 0 |
| Q2b. Đọc ở đâu thì biết từ nằm ở đâu trong câu? | **Ở S1 (trước CNN), không ở bản đồ điểm sau CNN.** Vệt chéo trên S1: tâm đoán nằm trong span thật 92,1%, sai số trung vị 0,07 s. Vệt chéo trên bản đồ sau CNN: 45,0% (sai số 0,62 s). Cửa sổ: 5,3%. Đặt ngẫu nhiên: 8,0%. | Chắc về hướng và độ lớn |
| Q2c. S1 có dùng để chọn từ được không? | **Không.** R#10 chỉ 65,57%, thua cách gốc 23 điểm. | Chắc |

**Khuyến nghị.**
- Giữ cách lấy trung bình của Hou25 để **chọn** từ. Dùng vệt chéo trên S1 để **định vị** từ đã chọn.
- Giả thuyết "average pool làm loãng tín hiệu" **chưa được ủng hộ** khi chỉ đổi cách gộp lúc test. Muốn kiểm tiếp phải train lại với cách gộp mới.

---

## 1. Bối cảnh và câu hỏi

Scorer nhận 1 câu nói và list 2000 từ, rồi chấm điểm từng từ: từ này có trong câu không. Hou25 dùng scorer này để giữ top-50 rồi đưa vào prompt Whisper.

Ta muốn scorer cho thêm **vị trí** của từ, vì E1 cho thấy boost đúng chỗ thắng prompt (`bench/results.md` mục 6). Vậy cần biết hai điều:
1. Cách Hou25 gộp bản đồ điểm thành 1 con số có đang làm mất thông tin không? Đọc cách khác thì có chọn từ tốt hơn không?
2. Thông tin vị trí nằm ở tầng nào của mô hình?

| Câu hỏi | Nội dung | Giả thuyết ban đầu |
|---|---|---|
| Q2a | Đổi cách gộp lúc test → R#10 thay đổi ra sao? | Đọc ở vệt (diag / box) > trung bình, nhất là với câu dài |
| Q2b | Đọc vị trí từ tầng nào thì khớp span thật? | S1 định vị tốt (như F14 của Hoàng); bản đồ sau CNN cũng định vị được nếu đọc theo vệt |
| Q2c | S1 một mình chọn từ được không? | Kém hơn cách gốc, vì CNN có học thêm |

Chưa làm trong báo cáo này: test-clean cho mean / max / diag / box (đã dừng giữa chừng để nhường GPU cho E1), checkpoint của Hoàng, train lại với cách gộp mới.

### 1.1 Prior: vì sao tin trước điều này, và vì sao chọn các cách đọc này

**Paper Hou25** (Interspeech 2025, đọc toàn văn, `related-work/hou25-ranking-selection.md`):
- Pipeline [P, §2.2]: encoder Whisper → Linear 1280→368 → cross-attention 2 chiều → 4 ma trận cosine S1–S4 → CNN 4 lớp → **AdaptiveAvgPool** → Linear → điểm.
- Paper không thử cách gộp nào khác và không nói gì về vị trí.

**Ablation của Hoàng** (repo `itshoang2024/hou25_interspeech`, NOTES F10, F14; tóm tắt trong `GOAL.md` mục 3a; ta chưa chạy lại các số này):
- F10: bỏ S1 thì mAP rơi xuống 0,08; bỏ cross-attention chỉ mất ≤ 0,016. → Hou25 thực chất là "S1 + CNN".
- F14 (Table 9): dò vệt chéo trên S1 thì định vị được từ, sai số 0,05 s, 99% trúng span (±0,2 s), kể cả với từ Whisper chép sai.
- Bản S1-only: đọc điểm tại đỉnh bản đồ CAM cho AUC 0,992, so với 0,975 khi dùng output của chính mô hình. → Hoàng rút ra "pool làm loãng vệt ngắn".

**A4 của nhóm** (`RESULTS.md` mục đầu):
- R#10 của Hou25 tụt mạnh ở câu dài: 92,6% (câu < 15 từ) → 77,4% (câu ≥ 50 từ).
- Mức tụt xếp đúng theo mức độ lấy trung bình của cách gộp: Hou25 (trung bình) tụt nhiều nhất, pxattn ở giữa, pmaxsim (max) tụt ít nhất.

**Niềm tin ban đầu.** Trên bản đồ, từ đúng chỉ là một vệt ngắn. Lấy trung bình cả bản đồ thì vệt đó bị hoà vào nền. Câu càng dài thì nền càng lớn, nên càng loãng. Đọc điểm ngay tại vệt sẽ tránh được việc đó.

**Các cách đọc (arm) suy ra.** Tất cả dùng chung một checkpoint, chỉ khác bước cuối:

| Arm | Đọc gì | Kiểm điều gì |
|---|---|---|
| `mean` (**mốc**) | trung bình cả bản đồ | = đúng Hou25; dùng để kiểm phần tái hiện có khớp không |
| `max` | 1 ô cao nhất | dạng gộp "không loãng" cực đoan nhất |
| `diag` | vệt chéo dốc 1, chỗ tốt nhất | đọc đúng hình dạng mà một từ để lại trên bản đồ; cho luôn vị trí |
| `box` | cửa sổ rộng bằng từ, lấy hết các cột | đọc theo vùng thời gian nhưng không cần đúng hình chéo; cho vị trí |
| `s1` | vệt chéo trên S1 (trước cross-attention và CNN) | kiểm lại F14 trên checkpoint của nhóm; S1 chọn từ được không |
| `s1raw` | như `s1` nhưng trên feature Whisper thô, không qua lớp chiếu | **đối chứng âm**: nếu lớp chiếu học được gì thì arm này phải kém nhất |

**Tiêu chí.** Lưu ý trung thực: tiêu chí này được viết rõ ra trong báo cáo, sau khi đã chạy; PLAN.md chỉ định trước metric.
- Q2a được xác nhận nếu diag hoặc box hơn mean và khoảng tin cậy của chênh lệch nằm trên 0, đặc biệt ở nhóm câu ≥ 50 từ.
- Q2b: định vị được nghĩa là "tâm trong span" vượt xa mức ngẫu nhiên (8%).

---

## 2. Setup

### 2.1 Mô hình và cách tính bản đồ điểm

| Hạng mục | Giá trị |
|---|---|
| Checkpoint | `runs/2026-10-18/ckpt/hou25/best.pt`, nhóm tự train trên train-clean-100, **2 epoch** (epoch 3 bị dừng do hết giờ GPU) |
| Encoder | Whisper-large-v3 đóng băng; 1 frame = 0,02 s; cắt theo độ dài thật của audio |
| Biểu diễn từ | audio TTS (edge-tts) của từng từ → cùng encoder |
| CNN | 4 lớp conv 3×3, stride 1, kênh 4→32→64→128→256, ReLU. Vùng nhìn của mỗi ô ở lớp cuối = 9×9 frame (≈ 0,18 s × 9 frame TTS) |
| Phần cứng | RTX 4090 (card 3), bf16; ~0,83 s/câu cho cả 2000 từ |

**Bản đồ điểm từng ô (CAM).** Hou25 làm như sau:

```
h = CNN(S)                    # (256 kênh, Ts frame câu, Tb frame từ)
điểm = w · mean_ô(h) + b      # trung bình mọi ô, rồi qua 1 lớp linear
```

Lớp linear và phép trung bình đổi chỗ cho nhau được. Vì vậy:

```
cam[t, j] = w · h[:, t, j]    # điểm của từng ô
điểm Hou25 = mean(cam) + b    # đúng bằng output gốc
```

Có `cam` rồi thì đọc nó theo nhiều cách mà không phải train lại.

**Ví dụ.** Câu dài 300 frame (6 s), từ "harshly" dài 30 frame TTS → `cam` cỡ 300 × 30.

| Cách đọc | Công thức | Lấy bao nhiêu ô |
|---|---|---|
| mean | trung bình 300 × 30 ô | 9000 |
| max | ô lớn nhất | 1 |
| diag | `diag[t] = (1/30) · Σ_j cam[t+j, j]`, lấy t tốt nhất → vị trí = t | 30 ô trên một đường chéo |
| box | trung bình `cam[t : t+30, :]`, lấy t tốt nhất → vị trí = t | 30 × 30 = 900 |
| s1 | như diag nhưng trên `S1 = cos(proj(câu), proj(TTS))` thay cho `cam` | 30 |

Cùng một list thì `mean` chia cho cùng một Ts với mọi từ (ví dụ trên: 300). Điều này quan trọng ở mục 3.1.

### 2.2 Dữ liệu

| | Giá trị |
|---|---|
| Tập | LibriSpeech test-other, 2939 câu |
| List | IS21 N2000: từ hiếm đúng + distractor, trung bình 1995 từ/câu |
| Lượt từ đúng | 5248; trung bình 1,79 từ hiếm/câu (trung vị 1, tối đa 14) |
| Vị trí thật | căn chỉnh MFA của LibriSpeech (Lugosch et al. 2019, zenodo 2619474), chuyển bằng `bench/mfa_to_align.py`; 5191/5248 lượt từ (98,9%) có span |

### 2.3 Metric

- **R#K** = % lượt từ đúng có hạng < K **trong list của chính câu đó**. Câu có > K từ hiếm thì không thể đạt 100%: với K = 10, trần trung bình của nhóm câu ≥ 7 từ hiếm là 97,0%; các nhóm khác là 100%.
- **R#1, hạng trung vị**: hạng tính từ 0. File chỉ giữ top-200, nên hạng ≥ 200 được tính là 200.
- **AUC gộp** = xác suất một từ đúng có điểm cao hơn một từ nhiễu, gộp mọi câu. Dùng để biết điểm có so được **giữa các câu** không, tức là có dùng ngưỡng chung được không. Tính trên top-200 mỗi câu; từ nhiễu ngoài top-200 coi như thấp hơn mọi từ trong top-200, nên đây là xấp xỉ.
- **Định vị**: tâm dự đoán = (frame đầu + số frame TTS / 2) × 0,02 s. So với tâm span MFA (nếu từ xuất hiện nhiều lần thì lấy span gần nhất). Báo:
  - sai số trung vị;
  - % tâm nằm trong span;
  - % tâm trong [đầu − 0,2 s; cuối + 0,2 s] (cách tính của F14).
  - **Mức ngẫu nhiên**: đặt tâm đều trong câu thì trúng span 8,0%.
- **Khoảng tin cậy**: bootstrap 1000 lần theo câu. Chênh lệch giữa hai cách đọc là ghép cặp: cùng câu, cùng từ.

### 2.4 Nguyên tắc so sánh

- Chỉ đổi đúng 1 thứ là bước gộp. Cùng checkpoint, cùng feature, cùng list.
- Không có seed nào cả: suy luận tất định, nên không có nhiễu do chạy lại. Nhiễu duy nhất là chọn câu nào vào tập, và khoảng tin cậy bootstrap đo phần đó.

---

## 3. Kết quả

### 3.1 Q2a: Đổi cách gộp có chọn từ tốt hơn không?

**Câu hỏi.** Cùng checkpoint, đọc bản đồ theo max / diag / box thay cho trung bình thì từ đúng có lên hạng không?
**Giả thuyết** (prior: mục 1.1). diag / box > mean, nhất là với câu dài.
**Setup.** Như mục 2. 4 cách đọc trên CAM, cộng `s1` để tham chiếu.

**Kết quả chính** (test-other, N2000, 5248 lượt từ đúng)

| Cách đọc | R#1 | R#10 [CI 95%] | Δ R#10 so với mean [CI 95%] | R#50 | Hạng trung vị | Ngoài top-200 | AUC gộp |
|---|---|---|---|---|---|---|---|
| **mean (Hou25)** | **36,49** | **88,83** [87,74; 89,85] | – | **95,41** | 1 | **1,54%** | **0,983** |
| diag | 33,88 | 85,27 [84,14; 86,41] | −3,56 [−4,51; −2,71] | 94,42 | 1 | 2,12% | 0,976 |
| box | 33,48 | 84,72 [83,61; 85,89] | −4,12 [−4,73; −3,53] | 94,68 | 1 | 1,92% | 0,979 |
| max | 30,53 | 77,25 [76,09; 78,51] | −11,59 [−12,66; −10,57] | 85,67 | 2 | 9,98% | 0,892 |
| s1 | 28,11 | 65,57 [64,26; 67,04] | −23,27 [−24,72; −21,83] | 80,70 | 3 | 9,39% | 0,897 |
| s1raw | – | 0,55 | – | 2,69 | – | – | – |

- `mean` cho 88,83%, khớp rank cũ của checkpoint này (88,80%). → Cách tính CAM tái hiện đúng Hou25.
- Ngưỡng chung giữ trung bình 3 từ/câu bắt được bao nhiêu % từ đúng: mean 83,7%, box 74,9%, diag 66,4%, max 50,4%, s1 46,9%.

**So từng từ (ghép cặp, R#10)**

| So với mean | Chỉ mean trúng | Chỉ cách kia trúng | Trong 586 từ mean trượt, cách kia cứu được | Trần nếu mỗi từ chọn được cách tốt hơn |
|---|---|---|---|---|
| diag | 359 | 172 | 29,4% | 92,11 |
| box | 242 | 26 | 4,4% | 89,33 |
| max | 682 | 74 | 12,6% | 90,24 |
| s1 | 1421 | 200 | 34,1% | 92,64 |

**Theo độ dài câu** (số từ trong reference; R#10)

| Cách đọc | < 15 từ | 15–29 | 30–49 | ≥ 50 | Tụt từ ngắn → dài |
|---|---|---|---|---|---|
| mean | 92,7 | 90,4 | 86,4 | 77,6 | −15,1 |
| diag | 87,4 | 86,2 | 85,2 | 74,8 | −12,6 |
| box | 85,9 | 86,4 | 84,0 | 75,6 | −10,3 |
| max | 67,5 | 81,5 | 82,3 | 73,3 | +5,8 |
| s1 | 69,8 | 65,4 | 64,0 | 58,0 | −11,8 |

**Theo số từ hiếm trong câu** (R#10; trần = mức cao nhất có thể đạt)

| Số từ hiếm | n lượt | Trần | mean | diag | box | max | s1 |
|---|---|---|---|---|---|---|---|
| 1 | 837 | 100 | 94,1 | 88,6 | 88,1 | 67,1 | 70,8 |
| 2–3 | 2053 | 100 | 91,9 | 87,7 | 87,0 | 79,3 | 67,9 |
| 4–6 | 1685 | 100 | 88,7 | 86,2 | 85,6 | 82,3 | 64,5 |
| ≥ 7 | 673 | 97,0 | 73,3 | 71,3 | 71,2 | 71,0 | 54,7 |

**Theo độ dài của từ đúng** (số chữ cái; R#10)

| Độ dài | n | mean | diag | box | max | s1 |
|---|---|---|---|---|---|---|
| ≤ 4 | 526 | 81,4 | 72,2 | 77,0 | 59,7 | 45,2 |
| 5–6 | 1549 | 88,7 | 83,2 | 85,0 | 73,9 | 57,6 |
| 7–8 | 1724 | 89,7 | 86,9 | 85,6 | 79,4 | 68,9 |
| ≥ 9 | 1449 | 90,6 | 90,3 | 86,2 | 84,7 | 77,6 |

**Mỗi cách đọc thiên về từ dài hay từ ngắn?** Đo độ dài trung bình của top-10; từ đúng dài trung bình 7,29 chữ.

| Cách đọc | Độ dài TB của từ nhiễu trong top-10 | % từ nhiễu ≤ 4 chữ trong top-10 |
|---|---|---|
| mean | 5,82 | 34,9% |
| diag | 5,83 | 34,9% |
| box | 4,73 | 54,2% |
| max | **10,82** | 3,1% |
| s1 | 6,09 | 27,3% |

Ví dụ câu `1688-142285-0003`, không có từ hiếm nào:
- top-5 của `max`: transportability, intangibly, trisystem, vulgarisation, discouraging (toàn từ dài);
- top-5 của `mean`: yehs, dat, too', the', wun (toàn từ ngắn).

**Rút ra được gì**
1. **Mọi cách đọc khác đều thua mean, và thua rõ.** Khoảng tin cậy của chênh lệch nằm hẳn dưới 0 cho cả 4 arm. Kết quả lệch với giả thuyết.
2. **Mức thua không đều:**
   - diag và box thua nhẹ (−3,6 và −4,1).
   - max thua nặng (−11,6) vì nó **thiên hẳn về từ dài**: từ nhiễu trong top-10 dài trung bình 10,8 chữ.
   - *Ta diễn giải:* từ TTS dài cho bản đồ nhiều ô hơn, nên ô lớn nhất dễ cao hơn chỉ vì có nhiều ô để chọn. Chưa kiểm trực tiếp.
3. **box thiên về từ ngắn** (54% từ nhiễu trong top-10 có ≤ 4 chữ).
   - *Ta diễn giải:* cửa sổ của từ ngắn nhỏ, trung bình trên ít ô nên nhiễu hơn và dễ vọt lên.
4. **Câu dài vẫn tụt dù đọc ở vệt.** diag ở nhóm ≥ 50 từ đạt 74,8%, vẫn dưới mean (77,6%). Đọc ở vệt không sửa được hiện tượng tụt R#10 ở câu dài, vốn là bằng chứng chính cho giả thuyết "loãng" ở A4.
5. **Phần lớn mức tụt ở câu dài đến từ việc có nhiều từ hiếm.**
   - Nhóm ≥ 7 từ hiếm: mean / diag / box / max đều quanh 71–73%, khoảng cách giữa các arm gần như biến mất.
   - Nhóm chỉ có 1 từ hiếm: khoảng cách lớn nhất (mean 94,1 so với diag 88,6).
   - Câu nhiều từ hiếm khó vì nhiều từ phải chia nhau 10 chỗ, và cùng lúc phải thắng cả 1990 từ nhiễu. Trần lý thuyết vẫn là 97%, nên đây không chỉ là giới hạn trần.
6. **Có thông tin bổ sung nhưng chưa khai thác được.**
   - diag cứu được 29% số từ mà mean trượt.
   - Nếu mỗi từ chọn được cách tốt hơn thì R#10 lên 92,1 (+3,3).
   - Nhưng không có quy tắc đơn giản nào biết lúc nào nên tin diag; đổi hẳn sang diag thì mất nhiều hơn được (359 so với 172).
7. **Vì sao "loãng" không làm hại xếp hạng trong câu.** *Ta diễn giải:*
   - Trong cùng một câu, mọi từ đều chia cho cùng Ts frame. Câu dài làm điểm của *mọi* ứng viên cùng loãng như nhau, nên thứ tự gần như giữ nguyên.
   - Loãng chỉ có thể hại khi so điểm *giữa các câu* (dùng ngưỡng chung). Nhưng ở đây mean cũng thắng: AUC gộp 0,983, recall tại ngưỡng chung 83,7% so với 66–75%.
   - Vậy trên checkpoint này, cả hai cách dùng điểm đều không thấy dấu hiệu "loãng".

**Kết luận.** **Không. Với checkpoint Hou25 này, cách lấy trung bình là cách đọc tốt nhất cả khi xếp hạng trong câu (R#10) lẫn khi dùng ngưỡng chung (AUC). Đổi cách gộp lúc test làm R#10 giảm 3,6–11,6 điểm.**

**Caveat.**
- **CNN được train cùng average pool.** Gradient chỉ đi qua trung bình, nên CNN không có lý do gì để dồn tín hiệu vào vệt. Đọc ở vệt lúc test là dùng mô hình ngoài cách nó được train. Thí nghiệm này bác bỏ "đổi cách gộp lúc test thì tốt hơn", **không** bác bỏ "train với cách gộp khác thì tốt hơn".
- **Lệch với kết quả S1-only của Hoàng** (đọc tại đỉnh CAM: AUC 0,992 so với 0,975). Khác nhau ở 3 chỗ:
  - checkpoint: S1-only của Hoàng, so với bản đủ 4 kênh 2 epoch của nhóm;
  - định nghĩa "đỉnh": ta chưa đọc code của Hoàng, nên không biết nó gần `max` hay `diag` của ta;
  - metric.
  
  Chưa kết luận được ai đúng cho tới khi chạy cùng checkpoint.
- Checkpoint của nhóm chỉ 2 epoch. Mới đo trên test-other.
- AUC gộp là xấp xỉ (chỉ có top-200 mỗi câu).

### 3.2 Q2b: Đọc ở đâu thì định vị được từ?

**Câu hỏi.** Vệt chéo đọc từ tầng nào thì rơi đúng vào chỗ từ được nói?
**Giả thuyết.** S1 định vị tốt (F14). Vệt chéo trên CAM cũng định vị được, vì mỗi ô CAM chỉ nhìn vùng 9×9 frame xung quanh.
**Setup.** Chỉ xét từ đúng có span MFA (5191 lượt). Vị trí lấy tại chỗ cho điểm vệt cao nhất của chính từ đó; đây là định vị khi đã biết từ có trong câu.

**Kết quả**

| Đọc từ | Sai số trung vị | Tâm trong span | Trong [đầu − 0,2 s; cuối + 0,2 s] (kiểu F14) | Trong ±0,5 s quanh tâm |
|---|---|---|---|---|
| **s1** | **0,070 s** | **92,1%** | **92,8%** | 93,1% |
| diag trên CAM | 0,620 s | 45,0% | 46,2% | 47,3% |
| box trên CAM | 2,085 s | 5,3% | 9,9% | 11,2% |
| s1raw | 3,235 s | 1,5% | – | – |
| *đặt ngẫu nhiên* | – | *8,0%* | – | – |

Tham khảo Hoàng (F14, checkpoint khác, căn chỉnh wav2vec2): 0,05 s, 99% trúng span. Test-clean, s1: 0,070 s, 96,0% trong span.

**Theo nhóm từ**

| Nhóm | n | s1: trong span | diag CAM: trong span |
|---|---|---|---|
| Từ mean xếp trong top-10 | 4619 | 94,0% | 42,2% |
| Từ mean trượt top-10 | 572 | 76,6% | 67,8% |
| Từ Whisper (không context) chép sai | 948 | 83,4% | – |
| Câu < 5 s | 1315 | 90,4% | 54,1% |
| Câu 5–10 s | 2003 | 92,3% | 41,5% |
| Câu 10–20 s | 1608 | 93,2% | 42,1% |
| Câu ≥ 20 s | 265 | 91,7% | 44,2% |

**Rút ra được gì**
1. **S1 định vị gần như hoàn hảo**: 92% trong span, gấp 11 lần mức ngẫu nhiên, sai số trung vị 0,07 s (3,5 frame). Khớp hướng với F14 của Hoàng, dù thấp hơn (92% so với 99%). Có thể do khác checkpoint và khác mốc căn chỉnh; chưa tách được.
2. **S1 không phụ thuộc độ dài câu** (90–93% ở mọi nhóm). Vẫn được 83% với từ Whisper chép sai, tức là đúng nhóm từ mà biasing cần.
3. **Sau CNN, vị trí gần như mất.**
   - diag trên CAM chỉ 45%, sai số 0,62 s. Ở câu ≥ 5 s, nó rơi về khoảng 42%.
   - box chỉ 5%, **thấp hơn cả đặt ngẫu nhiên**: cửa sổ tốt nhất thường nằm hẳn ngoài chỗ từ.
   - Điều này trái với kỳ vọng: vùng nhìn 9×9 lẽ ra giữ được tính cục bộ.
   - *Ta diễn giải:* vì chỉ được train qua trung bình, CNN học "đặc trưng tổng thể" của cả bản đồ (nền, kết cấu), không học "đỉnh ở đúng vệt".
4. **Điều lạ, chưa giải thích được.** diag trên CAM định vị *tốt hơn* với từ mà mean xếp trượt (67,8%) so với từ mean xếp trúng (42,2%). Đã kiểm khả năng bị dính mép đầu câu: chỉ 0,9% lượt có frame đầu = 0/1, nên không phải do đó. Cần xem thêm bản đồ CAM của vài câu.
5. **Lớp chiếu đã train là bắt buộc.** Làm như S1 nhưng trên feature Whisper thô (s1raw) thì gần như không định vị được (1,5%, dưới cả ngẫu nhiên) và không xếp hạng được (R#10 0,55%). Vậy chính lớp Linear 1280→368 đã học được phép so khớp theo thời gian.

**Kết luận.** **Thông tin vị trí nằm ở S1 (sau lớp chiếu, trước CNN). Bản đồ điểm sau CNN không giữ được vị trí. Để có (từ, vị trí) thì chọn từ bằng điểm trung bình, rồi định vị bằng vệt chéo trên S1.**

**Caveat.**
- Đây là định vị **khi đã biết từ có trong câu**. Với từ nhiễu, S1 vẫn trả về một "vị trí". E1 cho thấy đó chính là lý do gate theo S1 hỏng với list top-50.
- Mốc MFA có sai số riêng của nó. 41 câu bị lệch số từ đã được đổi tên `<unk>` theo reference.

### 3.3 Q2c: S1 chọn từ được không?

Có trong bảng mục 3.1: **không**.
- R#10 65,57%, thua mean 23,3 điểm; AUC gộp 0,897.
- Dù vậy, S1 cứu được 34% số từ mà mean trượt. Có thông tin bổ sung, giống trường hợp diag.
- Hợp lý: S1 chỉ là cosine từng frame. Từ nghe na ná hoặc từ ngắn cũng cho vệt cao. CNN chính là phần học để phân biệt "vệt thật" với "vệt tình cờ".

---

## 4. Tổng hợp

### 4.1 Giả thuyết so với thực tế

| Giả thuyết | Thực tế |
|---|---|
| Đọc ở vệt (diag / box) chọn từ tốt hơn trung bình | ❌ thua 3,6–4,1 điểm R#10 |
| max ít loãng nên tốt ở câu dài | ⚠️ max tụt ít theo độ dài câu, nhưng thua nặng ở câu ngắn (67,5%) vì thiên về từ dài |
| Đọc ở vệt sửa được hiện tượng tụt ở câu dài | ❌ diag ≥ 50 từ vẫn 74,8% |
| S1 định vị được từ (F14) | ✅ 92,1% trong span, 0,07 s |
| Bản đồ sau CNN cũng định vị được | ❌ 45% (diag), 5% (box) |
| Lớp chiếu cần cho S1 (đối chứng âm s1raw) | ✅ s1raw không dùng được |
| S1 chọn từ kém hơn mean | ✅ −23 điểm |

### 4.2 Rút ra chung (xếp theo độ lớn ảnh hưởng)

1. **Chia việc rõ ràng**: điểm trung bình sau CNN để **chọn**, S1 để **định vị**. Khoảng cách rất lớn ở cả hai chiều: chọn thì 88,8 so với 65,6; định vị thì 92% so với 45%.
2. **Không có lợi gì khi đổi cách gộp lúc test.** Hướng "đọc ở vệt" chỉ còn mở nếu train lại.
3. **Có 3–4 điểm R#10 bổ sung** nằm ở diag và S1 (trần 92,1–92,6 so với 88,8). Đây là chỗ một bộ kết hợp có train (dùng cả mean, diag, S1 làm đặc trưng) có thể lấy lại. Đây là ý tưởng, chưa đo.

### 4.3 Hàm ý cho bước sau

- Cho DAS (`GOAL.md` mục 3):
  - Khối "định vị" có sẵn bằng chứng mạnh: S1 đủ tốt khi đã biết từ.
  - Phần thiếu là **quyết định "từ có trong câu không" ở cấp từng vị trí**. E1 cho thấy thiếu nó thì gate theo S1 hỏng (top-50 + S1: WER 15,8).
- Giả thuyết "average pool làm loãng" (H5): chuyển từ "nghiêng về ủng hộ" (A4) sang **"chưa được ủng hộ"**. Còn đúng một cách kiểm: train lại với diag hoặc box làm bước gộp, cùng dữ liệu, cùng số epoch.

---

## 5. Giới hạn và độ tin cậy

- **Một checkpoint, 2 epoch, của nhóm.** Mọi kết luận "cách đọc nào tốt hơn" gắn với checkpoint này. Phải chạy lại trên checkpoint của Hoàng (PLAN P0.2).
- **Một tập**: mean / max / diag / box chỉ có test-other. S1 có cả test-clean và cùng hướng.
- **Đổi cách gộp lúc test ≠ train với cách gộp đó** (xem caveat mục 3.1).
- **Chỉ có list ngẫu nhiên IS21.** Chưa đo với list nghe giống (`near_N2000`, `ctx_hard_N2000`). Ở đó tranh chấp giữa các từ na ná có thể đổi bức tranh.
- **Hạng bị cắt ở 200 và AUC là xấp xỉ.** Không ảnh hưởng R#10 / R#50.

## 6. Việc còn lại

| Việc | Trạng thái | Ghi chú / lệnh |
|---|---|---|
| mean / max / diag / box trên test-clean | chưa chạy (đã dừng để nhường GPU cho E1) | `python analysis/a1_pool.py --split test-clean --libri /work/libri/LibriSpeech` |
| Chạy lại toàn bộ trên checkpoint của Hoàng | chờ file (PLAN P0.2) | `--ckpt <best.pt của Hoàng>` |
| Train lại Hou25 với diag / box làm bước gộp | chưa làm, cần duyệt | kiểm thật sự giả thuyết H5 |
| Bộ kết hợp mean + diag + S1 | ý tưởng | trần +3,3 điểm R#10 |
| Xem bản đồ CAM của vài câu (vì sao diag định vị tốt hơn khi mean trượt) | chưa làm | |
| A1 trên list nghe giống | chưa làm | `--list hard_N2000` |

## Phụ lục

**Lệnh tái lập** (trong container trên 4090, `/work` = `~/model_tmp`):
```
cd /work/code/scorer
cat work/bias.f16 > /dev/null        # đưa 21 GB feature TTS vào cache trước, nếu không sẽ chậm gấp 4
python analysis/a1_pool.py --split test-other --libri /work/libri/LibriSpeech
python analysis/a1_s1.py   --split test-other --libri /work/libri/LibriSpeech
python analysis/a1_eval.py --split test-other
python analysis/a1_eval.py --split test-other --v mean s1 s1raw --loc loc_s1 --hyp ../bench/out/test-other/whisper_oracle_plain.tsv
```
Phân tích thêm của báo cáo này (CI, so ghép cặp, nhóm theo độ dài, AUC gộp, mức ngẫu nhiên) dùng script tạm trong scratchpad, chạy trên các file json dưới đây.

**File kết quả** (trên server, `~/model_tmp/code/scorer/work/out/a1/test-other/`):
- `rank_{mean,max,diag,box,s1,s1raw}_N2000.json`: top-200 mỗi câu kèm điểm.
- `loc_N2000.json`: {uid: {từ: [frame đầu diag, frame đầu box, số frame TTS]}}.
- `loc_s1_N2000.json`: {uid: {từ: [frame đầu s1, frame đầu s1raw, số frame TTS]}}.

**Log**: `bench/out/run1108_logs/a1_other.log`, `s1_other.log`, `s1_clean.log`.
