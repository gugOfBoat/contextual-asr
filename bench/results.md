# Kết quả baseline — LibriSpeech + IS21 (full test-clean 2620 / test-other 2939 câu)

Chạy 02/10/2026 trên 1× RTX 4090. Scorer IS21 chính thức (sub=4, ins/del=3; B-WER theo từ hiếm của reference).
Định dạng: test-clean / test-other. "plain" = chỉ hạ chữ thường + bỏ dấu câu; "wnorm" = EnglishTextNormalizer của Whisper áp cho cả ref, hyp và từ hiếm.

## 1. Sanity check pipeline
- Qwen3-ASR-1.7B, không context, wnorm: WER test-clean **1.63** = đúng số trên model card chính thức (1.63) → pipeline đúng.

## 2. Tái hiện Hou25 (Interspeech 2025), Bảng 1 — Whisper-large-v3-turbo, prompt "naive"

| Điều kiện | Hou25 WER | Mình WER (plain) | Hou25 B-WER | Mình B-WER (plain) |
|---|---|---|---|---|
| không prompt | 3.2 / 5.3 | 2.64 / 4.87 | 10.4 / 19.7 | 9.43 / 18.92 |
| + prompt N=100 | 2.7 / 4.8 | 2.55 / 4.20 | 7.8 / 15.2 | 8.04 / 13.98 |
| + prompt N=2000 | 2.9 / 5.2 | 2.79 / 4.97 | 9.8 / 19.5 | 9.15 / 18.69 |

→ Khớp **xu hướng** của Hou25: N=100 giúp, N=2000 gần như mất hết tác dụng (bias bị loãng). Số tuyệt đối của mình thấp hơn khoảng 0.3–0.6, có thể do khác chuẩn hoá text hoặc cấu hình decode (Hou25 không nêu rõ). Prompt dài được cắt còn 223 token cuối (giống openai-whisper).

wnorm: none 2.17/4.25, B 8.67/17.90 · N100 2.07/3.64, B 7.27/13.48 · N2000 2.32/4.39, B 8.34/17.83.

## 3. Qwen3-ASR-1.7B — context trong system prompt ("w1, w2, ...")

| Điều kiện | WER (wnorm) | U-WER (wnorm) | B-WER (wnorm) | B-WER (plain) |
|---|---|---|---|---|
| không context | 1.63 / 3.41 | 0.89 / 1.91 | 7.40 / 16.08 | 7.97 / 16.86 |
| oracle (chỉ từ đúng) | 1.22 / 2.43 | 0.87 / 1.80 | **4.01 / 7.73** | 4.03 / 7.94 |
| N=100 | 1.42 / 3.17 | 0.93 / 2.17 | 5.30 / 11.61 | 5.54 / 12.06 |
| N=500 | 1.60 / 3.33 | 0.94 / 1.96 | 6.73 / 14.88 | — / 15.46 |
| N=1000 | 1.67 / 3.45 | 0.95 / 1.98 | 7.30 / 15.92 | — / 16.52 |
| N=2000 | 1.69 / 3.55 | 0.96 / 2.02 | 7.40 / **16.45** | 7.98 / … |
| missing_N100 (list không có từ đúng) | 1.73 / 3.50 | 0.98 / 1.98 | 7.62 / **16.34** | — / 17.08 |

(Đủ 7 điều kiện × 2 tập, cập nhật 03/10. Test-other: B-WER tăng theo N, 11.61 → 14.88 → 15.92 → 16.45; N2000 và missing đều **tệ hơn** không context 16.08. Output + log server: `out/`, `logs_4090/`.)

Tham chiếu PTC-Bias (arXiv 2609.28727), N=2000: Qwen3-ASR + CTC-Filter B-WER 4.81/9.45; Qwen3-ASR + PTC-Bias 4.10/7.76, U-WER 1.17/2.68.

## 4. Nhận xét ban đầu
1. Qwen3-ASR **dùng context tốt hơn Whisper**: oracle giảm B-WER khoảng 46% (clean) / 52% (other), U-WER gần như giữ nguyên.
2. Khoảng cách oracle ↔ N=100 (5.30 so với 4.01) = cái giá của distractor → đây là chỗ "chọn/khám phá candidate" có thể thu hồi.
3. Chuẩn hoá text làm lệch B-WER khoảng 0.5–1 điểm → paper phải ghi rõ normalizer; đề xuất dùng wnorm (khớp model card).

## 5. Qwen3-ASR + scorer lọc list N=2000 (16–17/10, chi tiết `../scorer/results.md`)
wnorm, toàn bộ test-clean / test-other, list vào system prompt như mục 3.

| Cách chọn list từ N=2000 | WER | U-WER | B-WER |
|---|---|---|---|
| pxattn top-50 (kiểu Hou25) | 1.47 / 2.95 | 0.93 / 1.93 | 5.71 / 11.57 |
| **pxattn lọc p ≥ 0.5** | 1.32 / 2.68 | 0.88 / 1.85 | **4.78 / 9.71** |
| pxattn lọc p ≥ 0.7 | 1.34 / 2.74 | 0.88 / 1.86 | 4.96 / 10.25 |
| hxattn lọc p ≥ 0.5 | 1.57 / 3.23 | 0.94 / 1.95 | 6.58 / 14.07 |
| missing_N2000 → pxattn p ≥ 0.5 | 1.65 / 3.42 | 0.92 / 1.91 | 7.39 / 16.18 |

So với tham chiếu PTC-Bias cùng setting: ngang "Qwen3-ASR + CTC-Filter" (4.81 / 9.45), còn thua PTC-Bias (4.10 / 7.76).
