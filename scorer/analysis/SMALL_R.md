# Đo lại scorer với R nhỏ + chia nhóm lỗi (14/11/2026, CPU)

> Script: `small_r.py`, `small_r2.py`. Bảng thô: `small_r_other.md`, `small_r_clean.md`.
> Dữ liệu: top-50 trong `prob_N2000.json` của checkpoint 18/10, list IS21 N2000, train clean-100. Hou25 là bản tái hiện 2 epoch (bản 3 epoch của Hoàng chưa có số).
> **Sạch@K** = % từ đúng có ít hơn K từ **nhiễu** đứng trên nó. Không tính các từ đúng khác, để câu nhiều từ hiếm không bị phạt oan.

## N1. R nhỏ làm lộ khoảng cách mà #50 che mất

| test-other | Sạch@1 | Sạch@3 | Sạch@5 | Sạch@10 | Sạch@50 (≈ #50) | từ nhiễu p≥0.5 / câu |
|---|---|---|---|---|---|---|
| Hou25 | **72.8** | 83.8 | 87.2 | 90.9 | 95.4 | 2.6 |
| pmaxsim | **85.7** | 93.7 | 95.1 | 96.7 | 98.4 | 0.7 |
| pxattn | 77.4 | 89.4 | 92.0 | 94.6 | 97.7 | 1.7 |

test-clean, Sạch@1: Hou25 80.9, pmaxsim 93.0.

Ở mốc 50, hai hệ chỉ cách nhau 3 điểm; ở mốc 1, cách nhau **13 điểm**. Hou25 để một từ nhiễu đứng trên từ đúng ở 27% số lượt.

## N2. Từ đúng bị vượt bởi từ **khác hẳn**, không phải từ nghe giống (trên list IS21)

Xét các lượt có ít nhất một từ nhiễu đứng trên từ đúng, và nhìn từ nhiễu cao nhất (test-other):

| | khác hẳn (> 2 âm vị) | gần âm (≤ 2) | đồng âm | sau giây 30 |
|---|---|---|---|---|
| Hou25 (1429 lượt) | **96%** | 3% | 1% | 0.4% |
| pmaxsim (753) | 91% | 6% | 2% | 0.8% |

Ví dụ Hou25: `jeered` ← `appropriately`, `ivanovitch` ← `hatreds`, `simulates` ← `intuite`.

**Lưu ý:** list IS21 ít khi chứa từ gần âm với từ đúng (khớp O6). Phải đo lại trên list `hard_N2000` (T3) mới kết luận được về từ nghe giống.

## N3. Báo động giả của Hou25 phần lớn "không giống gì trong câu"

Xét các từ nhiễu có p ≥ 0.5 đứng trên một từ đúng. So mỗi từ nhiễu với **mọi đoạn liên tiếp** trong chuỗi âm vị của câu; khoảng cách tính bằng edit âm vị chia cho độ dài từ nhiễu.

| test-other | số lượt | giống gần như hệt một đoạn (≤ 0.2) | 0.2–0.4 | không giống đoạn nào (> 0.4) |
|---|---|---|---|---|
| Hou25 | 2700 | 7% | 17% | **76%** |
| pmaxsim | 597 | 25% | 35% | 40% |
| pxattn | 1517 | 14% | 28% | 58% |

test-clean cho kết quả cùng hướng (Hou25 76%, pmaxsim 34%).

- **Hou25:** phần lớn là báo động giả thật, tức từ nhiễu không khớp âm nào trong câu.
- **pmaxsim:** báo động giả ít hơn 4.5 lần. Trong số còn lại, 60% là "nghe đúng một đoạn âm nhưng gán nhầm từ".
- Không phải do vài "từ hub" lặp lại: 7587 lượt báo động giả của Hou25 rải trên 6323 từ khác nhau.

## N4. Báo động giả của Hou25 tăng mạnh theo độ dài câu

Số từ nhiễu p ≥ 0.5 trung bình mỗi câu (test-other):

| | < 5 s | 5–10 s | 10–20 s | ≥ 20 s |
|---|---|---|---|---|
| Hou25 | 1.4 | 2.5 | 6.1 | **11.4** |
| pmaxsim | 0.5 | 0.8 | 1.2 | 1.5 |
| pxattn | 1.6 | 1.7 | 2.0 | 2.3 |

Sạch@1 theo độ dài câu:

| | câu < 10 s | câu ≥ 10 s |
|---|---|---|
| Hou25 | 77.6 | **64.2** |
| pmaxsim | 86.2 | 84.8 |

**Diễn giải (chưa kiểm):** CNN của Hou25 phản ứng với mảng "giống vệt" bất kỳ trên bản đồ. Câu dài cho nhiều mảng ngẫu nhiên hơn, nên từ nhiễu được điểm cao hơn.

Điều này đi **ngược** giả thuyết cũ "lấy trung bình làm loãng từ đúng". Vấn đề có vẻ không phải là từ đúng bị kéo xuống, mà là **từ nhiễu bị đẩy lên**. Hệ quả cho T2: đổi sang lấy max hoặc top-k có thể làm báo động giả **tăng thêm**, vì càng nhiều vị trí thì càng dễ có một ô cao do may rủi.

## N5. Nhóm từ ASR nghe sai là nhóm scorer yếu nhất

| Sạch@1, test-other | Whisper không prompt nghe đúng | nghe sai |
|---|---|---|
| Hou25 | 79.0 | **45.7** |
| pmaxsim | 91.4 | 60.8 |
| pxattn | 83.6 | 50.9 |

Với từ Whisper nghe sai (chính là nhóm cần scorer), Hou25 chỉ xếp từ đúng đứng đầu được **chưa tới một nửa** số lượt.

## N6. Phía ASR: có list mà không có vị trí thì Whisper có viết từ vào **sai chỗ** không? → **Gần như không**

Script `bench/misplace.py`, bảng `misplace_other.md`. test-other, % trên số lượt từ hiếm của nhóm câu.

| | 1 từ hiếm / câu | 2–4 | ≥ 5 |
|---|---|---|---|
| "Đúng từ, sai chỗ" (viết từ hiếm khác của câu vào chỗ này), prompt oracle | 0.00 | 0.07 | 0.12 |
| như trên, boost không gate | 0.00 | 0.25 | 0.36 |
| Từ nhiễu của prompt viết vào chỗ từ đúng, prompt top-50 | 0.47 | 0.21 | 0.30 |
| **Chèn thêm từ hiếm của câu**, prompt oracle | **68.5** | 0.07 | 0 |
| như trên, boost không gate | 12.1 | **24.4** | 6.6 |
| như trên, gate MFA | 0 | 3.8 | 0.1 |
| Viết đúng: prompt oracle / gate MFA | 88.6 / 91.6 | 91.0 / 94.7 | 92.2 / 93.5 |

Rút ra:
- Whisper **nghe audio** nên tự đặt từ đúng chỗ. Chuyện "có nhiều từ mà không biết từ nào ở đâu" không phải lỗi đáng kể (≤ 0.36%).
- Thiếu vị trí gây lỗi theo kiểu khác: **viết thêm** từ ở chỗ không có.
  - Prompt chỉ 1 từ → Whisper lặp từ đó (68.5%, khớp F6).
  - Boost không gate → đẩy cả ở chỗ không có (24.4%).
- Gate theo vị trí: viết đúng cao nhất, và chèn gần như bằng 0 (phần 3.8% còn lại là vòng lặp, B3 xử lý).

## N7. Phía scorer: **tại đúng chỗ** từ được nói, từ đúng có đứng đầu không? → Có, và không phụ thuộc số từ hiếm

Hou25 (A1, top-50) cùng vệt S1 cho vị trí từng từ. Chỉ xét từ đúng đã lọt top-50.

| nhóm câu | đứng đầu so với **mọi** nhiễu của câu | đứng đầu so với nhiễu **có vệt cùng chỗ** | số nhiễu cùng chỗ (TB) |
|---|---|---|---|
| 1 từ hiếm | 83.9 | 90.2 | 14.1 |
| 2–4 | 79.0 | 90.6 | 9.2 |
| ≥ 5 | **68.1** | **91.0** | 5.1 |

- Xếp hạng cả câu tệ dần khi câu có nhiều từ hiếm (cũng là câu dài, nhiều báo động giả: N4).
- Xếp hạng **tại từng chỗ** giữ khoảng 90–91% ở mọi nhóm. Vị trí chia bài toán "1 đúng trong 2000" thành nhiều bài nhỏ "1 đúng trong khoảng 5–14 từ cùng chỗ".
- **Lưu ý:**
  - Chỉ tính từ đã lọt top-50.
  - Vị trí của từ nhiễu do vệt S1 đoán, vị trí từ đúng lấy từ MFA.
  - Muốn chắc phải đo trên scorer cho vị trí của mọi từ (E2 `diag`, pmaxsim).
