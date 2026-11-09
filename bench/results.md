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

Tham chiếu PTC-Bias (arXiv 2609.28727), N=2000 (Bảng 1, **Qwen3-ASR-0.6B**, posterior WavLM): Qwen3-ASR + CTC-Filter B-WER 4.81/9.45; Qwen3-ASR + PTC-Bias 4.10/7.76, U-WER 1.17/2.68.

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

So với tham chiếu PTC-Bias (khác model: 0.6B): ngang "Qwen3-ASR + CTC-Filter" (4.81 / 9.45), còn thua PTC-Bias (4.10 / 7.76).

## 6. E1: boost có gate theo vị trí trên Whisper (08–09/11, GPU 4090 card 3)
- Model: whisper-large-v3-turbo, greedy, `gate_boost.py`. wnorm, `e1_table.py`.
- Boost: cộng λ vào log-prob token của bias word. Gate: chỉ boost khi giây mà decoder đang nghe (đọc từ alignment heads) nằm trong span của từ ± margin.
- λ = 5, margin = 0.1 s, chọn trên dev-other sub400 (ctx oracle, span từ wav2vec2). Không chỉnh trên test. Bảng quét: `out/dev-other/`.
- Span: "MFA" = căn chỉnh MFA (Lugosch et al. 2019). "S1" = vị trí đoán bằng vệt chéo S1 của checkpoint Hou25 của nhóm (2 epoch), `scorer/analysis/a1_s1.py`.
- top-50 = 50 từ đầu theo Hou25 của nhóm (`ctx_a1mean_N2000`, gần như trùng `ctx_hou25_N2000`).
- "Chèn sai" = từ trong list N2000 xuất hiện trong hyp nhiều hơn trong ref. "Câu lặp" = hyp dài hơn 1.5 × ref + 3 từ.

**Test-other (2939 câu), list IS21 N2000**

| Hệ | WER | U-WER | B-WER | Chèn sai | Câu lặp |
|---|---|---|---|---|---|
| không context | 4.25 | 2.63 | 17.92 | 13 | 0 |
| oracle → prompt | 14.29 | 13.70 | 19.31 | 593 | – |
| top-50 → prompt (Hou25) | 3.41 | 2.65 | 9.84 | 40 | 0 |
| (ctx_hou25_N2000 → prompt) | 3.39 | 2.65 | 9.69 | 39 | 0 |
| oracle, boost không gate | 7.14 | 5.36 | 22.26 | 922 | 14 |
| **oracle, gate span MFA** | **3.19** | **2.63** | **7.90** | 115 | 1 |
| oracle, gate span S1 | 3.76 | 3.21 | 8.42 | 121 | 2 |
| top-50, boost không gate | 55.25 | 57.65 | 34.94 | 1638 | 146 |
| top-50, gate span S1 | 15.76 | 15.92 | 14.45 | 340 | 37 |

Test-clean, chỉ các hệ nền: không context 2.16 / 1.34 / 8.64; oracle → prompt 12.06 / 12.83 / 6.05; Hou25 top-50 → prompt 1.87 / 1.45 / 5.13 (WER / U-WER / B-WER).

**Rút ra**
1. Ở mức trần (đúng từ, đúng vị trí), gate thắng prompt Hou25: B-WER 7.90 so với 9.69, U-WER giữ nguyên 2.63. Bỏ gate (boost luôn bật) thì hỏng: B-WER 22.26, 14 câu lặp → **vị trí là thứ làm boost dùng được**.
2. Thay vị trí thật bằng vị trí đoán từ S1 vẫn hơn prompt (8.42), nhưng U-WER tăng 0.6.
3. Với list thật (top-50) thì gate theo S1 **thua xa prompt** (WER 15.76, 37 câu lặp). S1 đoán vị trí cho mọi ứng viên, nên từ nhiễu cũng được boost ở đâu đó. Muốn đi tiếp cần một bước quyết định "từ này có thật trong câu không" trước khi boost (ngưỡng điểm, hoặc chỉ boost top-k nhỏ). Đây là giả thuyết, chưa đo.
4. Đưa oracle vào prompt làm Whisper hỏng (WER 14.29), khớp với `whisper_oracle.tsv` cũ chạy bằng `run_whisper.py` (WER 20.18). Do đó không lấy "oracle → prompt" làm trần của prompt.
5. Chưa chạy các điều kiện gate trên test-clean.
