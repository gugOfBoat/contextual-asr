# Phần A: khám scorer Hou25 (PLAN.md P1)

## A4, bản chỉ dùng điểm (08/11/2026) [đo]

**Dữ liệu và cách làm**
- Nguồn: `rank_N2000.json` trên server (`code/scorer/work/out/<scorer>/<split>/`), xếp hạng đủ 2001 từ cho mỗi câu, IS21 N2000.
- Hou25 ở đây là **bản tái hiện của nhóm (2 epoch)**, không phải bản của Hoàng. Phải chạy lại khi có checkpoint của Hoàng.
- Script: `scorer/analysis/outrank.py`.
- Khoảng cách tính bằng edit distance trên chuỗi âm vị g2p.

**Từ đúng bị rớt khỏi top-10: kẻ vượt lên là ai?**

| Scorer | Tập | R#10 | Kẻ đứng đầu là từ na ná/gần (≤ 2 âm vị) | Cận trên nếu bỏ hết từ na ná/gần đứng trên | List có từ ≤ 2 âm vị (bất kể hạng) | Hạng trung vị của từ bị rớt |
|---|---|---|---|---|---|---|
| hou25 | other | 88.80 | 1.7% | 88.95 | 53.9% | 33 |
| hou25 | clean | 92.71 | 2.4% | 92.74 | 48.3% | 23 |
| pxattn | other | 92.97 | 2.7% | 93.08 | 53.9% | 27 |
| pxattn | clean | 96.17 | 4.1% | 96.26 | 48.3% | 17 |
| pmaxsim | other | 95.43 | 5.4% | 95.52 | 53.9% | 27 |
| pmaxsim | clean | 97.26 | 6.4% | 97.33 | 48.3% | 14 |

**Tỉ lệ rớt theo độ dài từ (số âm vị), test-other**

| Scorer | 0–3 âm vị | 4–5 | 6–8 | ≥ 9 |
|---|---|---|---|---|
| hou25 | 18.7% | 12.1% | 10.3% | 6.9% |
| pxattn | 12.6% | 7.2% | 6.4% | 4.9% |
| pmaxsim | 12.8% | 5.0% | 3.1% | 2.8% |

**R#10 theo độ dài câu (số từ trong reference)**

| Scorer | Tập | < 15 | 15–29 | 30–49 | ≥ 50 |
|---|---|---|---|---|---|
| hou25 | other | 92.6 | 90.4 | 86.4 | **77.4** |
| hou25 | clean | 96.5 | 95.8 | 91.7 | **81.2** |
| pxattn | other | 94.2 | 94.9 | 91.3 | 85.8 |
| pxattn | clean | 97.7 | 97.5 | 96.0 | 90.6 |
| pmaxsim | other | 96.8 | 95.8 | 94.3 | 92.9 |
| pmaxsim | clean | 98.6 | 98.4 | 97.3 | 92.0 |

## Rút ra

1. **Giả thuyết H7 sai trên IS21 (list ngẫu nhiên).**
   - Từ đúng gần như **không** bị từ na ná đẩy xuống. Kẻ vượt lên là hàng chục từ không liên quan; hạng trung vị của từ bị rớt là 14–33.
   - Một nửa số list có sẵn từ cách ≤ 2 âm vị, nhưng các từ đó hầu như không vượt lên được.
   - Nếu bỏ hết chúng, R#10 chỉ lên khoảng +0.1.
   - → Khối ② "tranh nhau giữa từ na ná" **không cải thiện được R#10 trên IS21**. Nó chỉ còn ý nghĩa với list nghe giống (H#50, `ctx_hard`, `near`). Cần xem lại vai trò của ② trong DAS.
2. **Lỗi chính là "nghe không ra từ đúng":** điểm của từ đúng thấp hẳn, không phải bị từ nào giành chỗ.
3. **Từ ngắn rớt nhiều nhất:** 0–3 âm vị rớt 9–19%, gấp 2–4 lần từ dài.
   - Một phần là lỗi của nhãn hoặc của g2p ("n't", "un", "harvey'swhich", "mo"). Cần lọc riêng để xem còn lại bao nhiêu.
4. **Câu dài làm R#10 tụt mạnh, nặng nhất ở Hou25** (other: 92.6 → 77.4).
   - Mức tụt xếp đúng theo **mức độ lấy trung bình** của cách gộp: hou25 (average pool) tụt nhiều nhất → pxattn (mean + max) → pmaxsim (max) tụt ít nhất.
   - Điều này **ủng hộ H5 (average pool làm loãng)**.
   - Biến nhiễu: câu dài có nhiều từ hiếm hơn, nên R#10 khó hơn (câu > 10 từ hiếm thì không thể đạt 100%). pmaxsim chịu cùng biến nhiễu đó mà tụt ít, nên vẫn nghiêng về "loãng".
   - Cần A1 (đổi cách gộp, cùng một checkpoint, GPU) để kết luận chắc.

## A1: cùng checkpoint Hou25, đổi cách gộp (08–09/11/2026, GPU 4090 card 3) [đo]

Báo cáo đầy đủ (CI, so ghép cặp, nhóm theo độ dài, AUC gộp, mức ngẫu nhiên): `A1_REPORT.md`.

**Câu hỏi (Q2):** Hou25 lấy trung bình cả bản đồ. Nếu đọc điểm ở đúng vệt thì R#10 có lên không? Đọc từ đâu thì định vị được từ?

**Setup**
- Checkpoint: hou25 của nhóm (2 epoch), không phải bản của Hoàng. Test-other, list IS21 N2000, 2939 câu, 5248 lượt từ đúng. Không train lại.
- Script: `a1_pool.py` (bản đồ điểm CAM sau CNN), `a1_s1.py` (S1 trước CNN), chấm bằng `a1_eval.py`.
- Vị trí thật: căn chỉnh MFA của LibriSpeech (Lugosch et al. 2019), `bench/mfa_to_align.py`; 98.9% lượt từ đúng có span.
- Cách đọc:
  - `mean`: trung bình CAM = đúng như Hou25.
  - `max`: ô CAM cao nhất.
  - `diag`: vệt chéo dốc 1 trên CAM, lấy chỗ tốt nhất.
  - `box`: cửa sổ rộng bằng từ trên CAM.
  - `s1`: vệt chéo trên S1 = cos(proj(speech), proj(TTS)).
  - `s1raw`: như s1 nhưng dùng feature Whisper thô.

**Kết quả: xếp hạng (test-other, N2000)**

| Cách đọc | R#10 | R#50 | câu < 15 từ | 15–29 | 30–49 | ≥ 50 |
|---|---|---|---|---|---|---|
| mean (= Hou25) | **88.83** | **95.41** | 92.7 | 90.4 | 86.4 | 77.6 |
| max | 77.25 | 85.67 | 67.5 | 81.5 | 82.3 | 73.3 |
| diag (CAM) | 85.27 | 94.42 | 87.4 | 86.2 | 85.2 | 74.8 |
| box (CAM) | 84.72 | 94.68 | 85.9 | 86.4 | 84.0 | 75.6 |
| s1 (vệt chéo S1) | 65.57 | 80.70 | 69.8 | 65.4 | 64.0 | 58.0 |
| s1raw | 0.55 | 2.69 | 0.4 | 0.4 | 0.7 | 1.1 |

- `mean` khớp với rank cũ (88.80), nên phần tái hiện đúng.
- Test-clean, s1: R#10 74.96, R#50 88.16. s1raw: 0.42.

**Kết quả: định vị từ đúng (tâm dự đoán so với MFA)**

| Đọc từ | Tập | Sai số trung vị | Tâm nằm trong span | Nhóm ASR chép sai: trong span |
|---|---|---|---|---|
| diag trên CAM | other | 0.620 s | 45.0% | |
| box trên CAM | other | 2.085 s | 5.3% | |
| s1 | other | 0.070 s | 92.1% | 83.4% (n=948) |
| s1 | clean | 0.070 s | 96.0% | 87.5% (n=511) |
| s1raw | other | 3.235 s | 1.5% | 1.9% |

- "Nhóm ASR chép sai" = từ đúng không xuất hiện trong bản chép Whisper không context (`whisper_oracle_plain`).

**Rút ra (A1)**
1. Trên checkpoint này, đọc điểm ở vệt (diag, box, max) **không** làm R#10 tăng; tất cả đều thua `mean` 3.5–11.6 điểm. Câu dài vẫn tụt dù đọc ở vệt (diag ≥50 từ: 74.8). → Giả thuyết "average pool làm loãng" **chưa được ủng hộ** khi giữ nguyên checkpoint (CNN được train cùng average pool, nên đổi cách gộp lúc test là lệch phân phối; muốn kết luận phải train lại với cách gộp mới).
2. **Vị trí nằm ở S1, không nằm ở CAM.** Vệt chéo trên S1 định vị từ đúng trong span 92–96% (trung vị 0.07 s), kể cả 83–88% với từ mà ASR chép sai. Sau CNN, vị trí gần như mất (diag 45%, box 5%).
3. S1 xếp hạng kém (R#10 65.6): S1 tốt để **định vị** từ đã biết, không tốt để **chọn** từ. Hợp lý nhất là ghép: chọn bằng `mean`, định vị bằng S1.
4. Lớp chiếu đã train là cần: S1 trên feature Whisper thô (s1raw) không dùng được (R#10 0.5, định vị 1.5%).

## Việc tiếp theo
- A1 đã chạy (đổi cách gộp lúc test). Còn thiếu: train lại với cách gộp mới, chuẩn hoá theo số từ hiếm trong câu.
- Tách nhóm từ ngắn: bao nhiêu là lỗi nhãn / g2p, bao nhiêu là nghe khó thật.
- Chạy lại tất cả trên checkpoint Hou25 của Hoàng.
