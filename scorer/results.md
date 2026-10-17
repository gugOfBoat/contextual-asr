# Kết quả scorer (đã chạy thật, 1× RTX 4090, 16/10/2026)

Train: train-clean-100 (28.539 câu), 5 epoch, lr 2e-4, encoder Whisper-large-v3 đóng băng, bias = âm vị (g2p_en).
Checkpoint = best theo Recall@10 trên 500 câu dev-clean (list = từ đúng + 100 ngẫu nhiên): pmaxsim 99,80 (ep 3), pxattn 99,70 (ep 2).
Dev này quá dễ (bão hoà từ epoch 0) → không dùng được để so kiến trúc.

## 1. Bộ lọc "có trong câu" — `inference/eval_filter.py`
Precision/Recall tính trên từ hiếm của reference; "Rỗng" = % câu bị lọc hết.
Mức rỗng lý tưởng = % câu không có từ hiếm: test-clean 24,4%, test-other 27,2%.

**test-clean** (P / R / số từ mỗi câu / rỗng)

| list | model | p≥0.5 | p≥0.7 |
|---|---|---|---|
| N100 | pxattn | 97.6 / 94.5 / 2.10 / 24.6% | 98.6 / 91.9 / 2.02 / 25.6% |
| N100 | pmaxsim | 98.7 / 96.4 / 2.12 / 25.2% | 99.8 / 89.1 / 1.94 / 27.6% |
| N2000 | pxattn | 67.4 / 94.5 / 3.04 / 15.0% | 79.8 / 91.9 / 2.50 / 19.5% |
| N2000 | pmaxsim | 80.1 / 96.4 / 2.61 / 20.8% | 95.7 / 89.1 / 2.02 / 26.8% |
| N2000 | cascade pmaxsim→pxattn | 71.4 / 94.5 / 2.87 / 16.0% | 81.7 / 91.8 / 2.44 / 19.9% |
| kho 209k (không list) | pmaxsim | 5.8 / 92.5 / 35.0 / 2.6% | 16.3 / 87.7 / 11.7 / 8.9% |
| missing_N100 | pxattn | — / — / 0.05 / 95.0% | — / — / 0.03 / 97.3% |
| missing_N2000 | pxattn | — / — / 0.99 / 46.1% | — / — / 0.50 / 63.6% |
| missing_N2000 | pmaxsim | — / — / 0.52 / 64.9% | — / — / 0.09 / 92.0% |

**test-other**

| list | model | p≥0.5 | p≥0.7 |
|---|---|---|---|
| N100 | pxattn | 94.6 / 88.8 / 1.68 / 29.1% | 97.4 / 83.7 / 1.53 / 31.7% |
| N2000 | pxattn | 48.1 / 88.8 / 3.30 / 13.3% | 67.0 / 83.7 / 2.23 / 21.7% |
| N2000 | pmaxsim | 69.5 / 91.1 / 2.34 / 21.5% | 92.5 / 79.3 / 1.53 / 32.5% |
| kho 209k | pmaxsim | 3.9 / 85.0 / 38.6 / 1.9% | 10.2 / 77.3 / 13.6 / 7.6% |
| missing_N2000 | pmaxsim | — / — / 0.71 / 56.5% | — / — / 0.12 / 89.6% |

Thời gian chấm / câu (test-clean): pxattn N100–N2000 + missing: 560 ms; pmaxsim gồm cả kho 209k: 75 ms; cascade 77 + 61 ms.

Đọc nhanh:
- List IS21: từ 100–2000 từ còn ~2–3 từ/câu, giữ ~90–96% từ đúng.
- pmaxsim (rẻ) lại tốt hơn pxattn ở N2000 → xác suất của pmaxsim tách tốt hơn khi list dài. Chưa rõ vì sao.
- List không có từ đúng: phần lớn câu trả về rỗng → bộ lọc biết "không chọn gì".
- Không cần list (kho 209k): giữ 88% từ đúng (clean) với ~12 từ/câu, nhưng precision chỉ 16%.

## 2. Nhãn "từ khó" — `data/qwen_pass1.py` (Qwen3-ASR không context, chuẩn hoá Whisper)
| split | câu | lượt từ hiếm | Qwen chép sai ("khó") |
|---|---|---|---|
| dev-clean | 2.703 | 5.755 | 7,33% |
| dev-other | 2.864 | 4.893 | 13,71% |
| train-clean-100 | 28.539 | 97.853 | 7,03% |

→ Qwen KHÔNG "thuộc" train-clean-100 (7,03% ≈ dev) → học nhãn khó trên train được.

## 3. "Đáng bias" = có trong câu × chưa có trong hyp lần 1 — `inference/worth.py`
So với từ khó thật (từ hiếm của ref mà hyp lần 1 bỏ sót): test-clean 408 lượt, test-other 831 lượt.

| list / model | split | p≥0.5: P / R / từ/câu | p≥0.7 |
|---|---|---|---|
| N2000 pxattn | clean | 10.2 / 71.6 / 1.10 | 16.7 / 64.2 / 0.60 |
| N2000 pmaxsim | clean | 17.5 / 70.6 / 0.63 | 47.1 / 49.0 / 0.16 |
| N2000 pxattn | other | 10.2 / 68.8 / 1.90 | 18.6 / 59.2 / 0.90 |
| kho 209k pmaxsim | clean | 0.2 / 48.3 / 32.7 | 0.6 / 39.0 / 9.7 |

Đọc nhanh: từ khó với Qwen cũng khó với scorer (recall 64–72% so với 92–95% trên mọi từ hiếm); sau khi bỏ từ Qwen đã chép đúng, phần còn lại phần lớn là nhiễu (precision 10–20%) nhưng chỉ ~1 từ/câu. Có giúp Qwen hay không → xem B-WER (mục 4).

## 4. B-WER khi đưa list vào Qwen3-ASR (đã chạy thật, 16–17/10)
Qwen3-ASR-1.7B, list vào system prompt `"w1, w2, ..."` (như baseline), list N2000 của IS21, chấm `bench/score.py --whisper-norm`.
Toàn bộ test-clean (2.620 câu) / test-other (2.939 câu). Thứ tự từ trong list: tăng dần theo điểm (Hou25 Bảng 3).

| Cách chọn list (từ N2000) | WER | B-WER clean | B-WER other |
|---|---|---|---|
| không context (baseline) | 1.63 / 3.41 | 7.40 | 16.08 |
| nguyên list N2000 (baseline) | 1.69 / 3.55 | 7.40 | 16.45 |
| pxattn top-50 (kiểu Hou25) | 1.47 / 2.95 | 5.71 | 11.57 |
| **pxattn p ≥ 0.5** | **1.32 / 2.68** | **4.78** | **9.71** |
| pxattn p ≥ 0.7 | 1.34 / 2.74 | 4.96 | 10.25 |
| pmaxsim p ≥ 0.7 | 1.41 / 2.89 | 5.46 | 11.39 |
| "đáng bias": pxattn p ≥ 0.5 + chưa có trong hyp lần 1 | 1.33 / 2.76 | 5.01 | 10.39 |
| "đáng bias": pmaxsim p ≥ 0.5 + chưa có trong hyp lần 1 | 1.35 / 2.72 | 5.08 | 10.32 |
| hxattn p(có) ≥ 0.5 | 1.57 / 3.23 | 6.58 | 14.07 |
| hxattn p(có) ≥ 0.3 | 1.57 / 3.22 | 6.58 | 14.16 |
| hxattn p(khó) ≥ 0.1 | 1.54 / 3.29 | 6.48 | 14.56 |
| hxattn p(khó) ≥ 0.3 | 1.60 / 3.34 | 7.13 | 15.40 |
| không list: pmaxsim kho 209k, p ≥ 0.7 | 1.81 / 3.70 | 8.12 | 16.49 |
| không list: + chưa có trong hyp lần 1 | 1.90 / 3.80 | 9.06 | 17.61 |
| list toàn từ sai (missing_N2000) → pxattn p ≥ 0.5 | 1.65 / 3.42 | 7.39 | 16.18 |
| list toàn từ sai → hxattn p ≥ 0.5 | 1.70 / 3.60 | 7.65 | 16.52 |
| oracle (baseline) | 1.22 / 2.43 | 4.01 | 7.73 |
| PTC-Bias (theo paper, N2000, chưa tái hiện) | — | 4.10 | 7.76 |

Đọc nhanh:
- Lọc theo ngưỡng (≈3 từ/câu) tốt hơn top-50 kiểu Hou25: 5.71 → 4.78 (clean), 11.57 → 9.71 (other).
- Bỏ các từ Qwen đã chép đúng ("đáng bias", không train) kém hơn chút: lần 2 Qwen lại chép sai một số từ đó khi không được nhắc.
- hxattn (thêm hyp lần 1 + loss theo list) kém hẳn: lọc "có mặt" chỉ còn recall 47%, giữ ~26 từ/câu ở N2000 (bảng lọc trong `runs/2026-10-16/log_hx.txt`).
  Dev: Recall@10 77% (pxattn 99,7%), Hard-R@5 54%. Giả thuyết (chưa kiểm): model học tắt theo hyp, bỏ qua audio.
- Không có list, chỉ dùng âm thanh trên kho 209k → hại; list sai sau khi lọc bằng pxattn → không hại.

## 5. File lưu (`scorer/runs/2026-10-16/`, 171 MB)
- `ckpt/{pxattn,pmaxsim,hxattn}/best.pt` — checkpoint dùng cho mọi số trên
- `log_*.txt` — log train, lọc, Qwen; `pass1_*.jsonl` — hyp lần 1 + nhãn khó; `phones.tsv` — âm vị 217k từ
- `out/<model>/<split>/prob_*.json` (top-50 kèm xác suất), `time.json`
- Output Qwen: `bench/out/<split>/qwen3asr_<ctx>.tsv`; list đã lọc: `bench/protocol/<split>/ctx_<tên>.json`
- KHÔNG kéo về: `rank_*.json` (691 MB, tính lại được từ ckpt), `bias.f16` (đã xoá 16/10), audio LibriSpeech
