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
| PTC-Bias (theo paper, N2000, Qwen3-ASR-0.6B, chưa tái hiện) | — | 4.10 | 7.76 |

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

## 6. Đánh giá lại theo `EVAL_PROTOCOL.md` (17/10, CPU + rank đầy đủ trên máy GPU)
Log: `runs/2026-10-17/log_eval_retrieval.txt` (scorer), `log_hypmatch.txt` (baseline), `log_build_lists.txt` (dựng list).
Checkpoint và ngưỡng vẫn là bản cũ (chọn khi đã nhìn test) → số **sơ bộ**; dev theo IS21 đã dựng (`bench/protocol/dev-*`), chưa chạy scorer trên dev.

**Tầng (a), list IS21 N2000** — Recall_B#K (%), K99 = K cố định nhỏ nhất đạt 99% (từ `rank_*.json` đầy đủ):

| Hệ | clean #10 / #50 / K99 | other #10 / #50 / K99 | từ khó #50 (clean / other) |
|---|---|---|---|
| hypmatch (không train, khớp âm vị với hyp Qwen lần 1) | 95.61 / 98.23 / — | 92.17 / 95.41 / — | 75.98 / 71.84 |
| pxattn | 96.19 / 99.17 / 37 | 92.99 / 97.71 / 132 | 92.65 / 90.49 |
| pmaxsim | 97.26 / 99.51 / 21 | 95.43 / 98.44 / 147 | 94.61 / 92.30 |
| pcascade | 96.82 / 99.56 / 21 | 93.90 / 98.57 / 111 | 95.83 / 93.14 |
| PTC Retrieval (paper, Bảng 3, chỉ other) | — | — / **99.3** / Recall_B@99 16.9 | — |
| BR-ASR acoustic (paper) | — | — / **99.7** / Recall_B@99 42.2 | — |

**Lọc ngưỡng, N2000** (P / R / từ mỗi câu / R từ khó):

| | clean | other |
|---|---|---|
| hypmatch, chỉ giữ từ có nguyên văn trong hyp (@1.0) | 81.7 / 94.1 / 2.50 / 24.3 | 77.3 / 86.6 / 2.00 / 18.7 |
| pxattn p ≥ 0.5 | 67.4 / 94.5 / 3.04 / 71.6 | 48.1 / 88.8 / 3.30 / 68.8 |
| pmaxsim p ≥ 0.5 | 80.1 / 96.3 / 2.61 / 70.6 | 69.5 / 91.1 / 2.34 / 67.0 |

**Tầng (b) — phân rã lỗi từ hiếm** (`bench/score.py --ctx`, wnorm), Qwen + pxattn p ≥ 0.5 / oracle:
- clean: bỏ sót khi chọn 2.12% · có trong list mà ASR vẫn sai 2.66% (oracle: 4.01%)
- other: bỏ sót khi chọn 4.97% · có trong list mà ASR vẫn sai 4.69% (oracle: 7.67%)
- `bench/compare.py`: pxattn p ≥ 0.5 so với top-50: ΔB-WER −0.94 [−1.32, −0.55] (clean), −1.86 [−2.40, −1.32] (other) → có ý nghĩa; G = 77.3% cả 2 tập (top-50: 49.8% / 55.9%).

Đọc nhanh:
- **Trên test-other, scorer của nhóm còn thua PTC / BR-ASR ở bước chọn từ**: Recall_B#50 97.7–98.6 so với 99.3–99.7; muốn đạt 99% phải giữ 111–147 từ.
- Baseline không train ngang scorer khi tính trên **mọi** từ hiếm (Qwen đã chép đúng ~93% → chỉ cần tìm lại trong hyp). Khác biệt nằm ở **từ khó**: scorer giữ 90–96% từ khó trong top-50, baseline chỉ 72–76% → giá trị của scorer = nghe được từ mà ASR chép sai.
- Lỗi còn lại chia đôi giữa "bỏ sót khi chọn" và "ASR có từ trong list mà vẫn sai" → cải thiện scorer chỉ gỡ được khoảng một nửa.
- (Bản cũ, đã thay bằng §7) Recall_H#K chỉ so được giữa các hệ trên cùng list; giá trị phụ thuộc số từ gần đồng âm trong list (L1: ~800 từ cả tập; L2 `hard_N2000`: ~85k). Chưa chạy scorer trên L2 (cần GPU).

## 7. So với PTC / BR-ASR trên đúng 3 metric của họ (18/10, test-other N2000)
Recall_H theo định nghĩa BR-ASR (2505.19179 Eq. 4 + chú thích 2): H = từ **trong list** có Levenshtein âm vị ≤ 2 với 1 từ đúng → test-other 20.955 từ (~7 / câu). Log: `runs/2026-10-18/log_brasr_metrics.txt`.
Checkpoint cũ (chọn trên dev dễ) → số **sơ bộ**. Hàng PTC / BR-ASR chép từ paper (PTC Bảng 3; BR-ASR Bảng 2), **không chạy lại**; PTC không ghi cách dựng H.

| Hệ | Recall_B@99 ↓ | Recall_B#50 ↑ | Recall_H#50 ↓ |
|---|---|---|---|
| ngẫu nhiên (tính tay: 50 / 2001) | ~1980 | 2.5 | 2.5 |
| hypmatch (không train, khớp với hyp Qwen) | n/a | 95.41 | 35.70 |
| pxattn | chưa đo (rank không lưu điểm) | 97.71 | **21.58** |
| pmaxsim | chưa đo | 98.44 | 25.52 |
| pcascade | chưa đo | 98.57 | 25.90 |
| BR-ASR acoustic (paper) | 42.2 | **99.7** | 69.3 |
| PTC Retrieval (paper) | **16.9** | 99.3 | 22.0 |

test-clean (không có số paper): pxattn 99.17 / 22.29 · pmaxsim 99.51 / 26.35 · pcascade 99.56 / 26.62 · hypmatch 98.23 / 37.68 (#50 / H#50).

Đọc:
- **Recall_H#50:** của nhóm ngang PTC (21.6–25.9 vs 22.0), tốt hơn hẳn BR-ASR (69.3) → scorer **phân biệt được từ nghe giống** ngang SOTA.
- **Recall_B#50:** thua 0.7–1.6 điểm (97.7–98.6 vs 99.3) → **bỏ sót từ đúng** nhiều hơn PTC: cứ 100 từ đúng thì mất 1.4–2.3 từ, PTC mất 0.7.
- Hai metric phải đọc cùng nhau: scorer bỏ qua audio có H#50 = 2.5% (rất "tốt") nhưng B#50 cũng 2.5%.
- Recall_B@99 cần điểm đầy đủ 2000 từ → chạy lại `rank.py` có lưu điểm (GPU).

## 8. Hou25 tái hiện + E3 (loss theo list) + Recall_B@99 (18/10 đêm, test-other N2000)
Log: `runs/2026-10-18/log_night_metrics.txt`; ckpt `runs/2026-10-18/ckpt/{hou25,pxattn_list}/best.pt`. H theo BR-ASR như §7. Mọi hệ train clean-100, encoder Whisper-large-v3 đóng băng.

| Hệ | Bias | Recall_B@99 ↓ | Recall_B#50 ↑ | Recall_H#50 ↓ | Recall_B#10 | ms/câu |
|---|---|---|---|---|---|---|
| **Hou25 tái hiện** (cross-attn + CNN) | audio TTS | 180.8 | 95.39 | **17.59** | 88.80 | 922 |
| pxattn | âm vị | 106.7 | 97.71 | 21.58 | 92.97 | 332 |
| pmaxsim | âm vị | **99.8** | **98.44** | 25.52 | **95.43** | 4 (+ encoder) |
| pxattn_list (E3) | âm vị | 518.8 | 94.95 | 22.28 | 87.88 | 362 |
| BR-ASR acoustic (paper) | | 42.2 | 99.7 | 69.3 | | |
| PTC Retrieval (paper) | | 16.9 | 99.3 | 22.0 | | |

test-clean: Hou25 81.4 / 97.77 / 16.09 · pmaxsim 10.6 / 99.51 / 26.35 · pxattn_list 77.1 / 98.05 / 23.45 (@99 / #50 / H#50).

Đọc:
- **Âm vị thắng Hou25 ở tìm từ đúng** (#50 98.4 vs 95.4, @99 100 vs 181) và rẻ hơn nhiều (không TTS, không 21 GB feature). **Hou25 thắng ở không bị lừa bởi từ nghe giống** (H#50 17.6, tốt hơn cả PTC 22.0). Giả thuyết (chưa kiểm): CNN trên ma trận frame × frame thấy được thứ tự âm. Lưu ý: Hou25 chỉ được 2 epoch (epoch 3 bị dừng ở giới hạn giờ GPU).
- **E3 (thêm 200 từ nhiễu + loss theo list) làm tệ đi mọi metric** → bỏ dạng này. Nghi do BCE bị pha loãng (chia cho ~220 ứng viên/câu) nên từ đúng ít được đẩy lên.
- **Recall_B@99 thua PTC rất xa** (100 vs 16.9). Chuẩn hoá điểm trong từng câu làm tệ hơn (pmaxsim 157) → không phải lỗi hiệu chỉnh; nguyên nhân là ~1% từ đúng khó nhất gần như không được nghe ra (K99 trong từng câu cũng 130–150). Khớp lỗi ① ở `DEEPDIVE.md`.

## 9. Train trên 960h so với clean-100 (19/10, list N2000)
Cùng kiến trúc, loss, negative; chỉ đổi dữ liệu train: clean-100 (28.5k câu, giọng đọc rõ) → clean-100 + clean-360 + other-500 (281k câu, có giọng khó). 1 epoch 960h, chọn ckpt theo dev-other N2000 (best = step 22295). "cùng bước" = ckpt step 17836 ≈ số bước của bản clean-100 (5 epoch). Log: `runs/2026-10-18/log_night_metrics.txt`.

| test-other | Recall_B@99 ↓ | Recall_B#50 ↑ | Recall_H#50 ↓ | Recall_B#10 |
|---|---|---|---|---|
| pmaxsim 100h | 99.8 | 98.44 | 25.52 | 95.43 |
| pmaxsim 960h cùng bước | **53.8** | **98.82** | 24.86 | 96.06 |
| pmaxsim 960h best | 64.9 | 98.57 | 24.29 | **96.47** |
| pxattn 100h | 106.7 | 97.71 | 21.58 | 92.97 |
| pxattn 960h cùng bước | 93.5 | 98.17 | **20.59** | 93.94 |
| pxattn 960h best | 63.9 | 98.59 | 21.47 | 94.70 |
| PTC (paper) | 16.9 | 99.3 | 22.0 | |

test-clean (@99 / #50 / H#50): pmaxsim 10.6 / 99.51 / 26.35 → 960h 7.7–7.9 / 99.5 / 26.4–26.7; pxattn 18.1 / 99.17 / 22.29 → 960h 11.8–14.6 / 99.5 / 19.7–21.9.

Đọc: thêm dữ liệu giọng khó giảm @99 trên test-other ~35–46% và tăng #50 0.4–0.9 điểm, không đánh đổi H#50; test-clean gần như bão hoà. Vẫn còn cách PTC (#50 98.6–98.8 vs 99.3; @99 54–65 vs 16.9) → dữ liệu giải thích một phần, không phải toàn bộ.

## 10. Bảng 1 + Bảng 2 kiểu Hou25: scorer của nhóm → Whisper-large-v3-turbo (28/10, GPU 0)
Cùng ASR, cùng list IS21, cùng prompt naive và thứ tự tăng dần như Hou25; chỉ đổi scorer. Scorer bản **100h** (clean-100, cùng dữ liệu với Hou25). Số Hou25 chép từ paper (Bảng 1, 2), không tự chạy.
Chấm `bench/score.py` **plain** (Hou25 không nêu normalizer); HF greedy, không fallback nhiệt độ. Mọi số: `runs/2026-10-28/score_plain.txt`, pipeline `pipe_hou25tab.sh`, log `logs/log_hou25tab.txt`.

**Bảng 1** — B-WER clean / other (WER trong ngoặc):

| | N=100 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|
| không prompt (mình) | 9.43 / 18.92 (2.64 / 4.87) | ← | ← | ← |
| + prompt nguyên list (mình) | 8.04 / 13.98 | 8.87 / 18.22 | 9.20 / 18.32 | 9.15 / 18.69 |
| Hou25 paper: không prompt / + prompt | 10.4 / 19.7 · 7.8 / 15.2 | 9.5 / 19.1 | 10.0 / 19.3 | 9.8 / 19.5 |
| **Hou25 paper: + scorer top-50** | 5.4 / 10.0 (2.3 / 4.2) | 5.4 / 10.7 | 5.6 / 11.3 | 5.8 / 11.7 (2.5 / 4.5) |
| **pxattn top-50** | **4.57 / 8.92** (2.18 / 3.66) | **4.89 / 9.21** | **5.16 / 9.79** | **5.29 / 10.00** (2.35 / 4.23) |
| **pmaxsim top-50** | 4.60 / 10.30 (1.92 / 3.81) | 4.95 / 9.53 | 5.17 / 9.83 | 5.47 / 10.37 (2.22 / 4.31) |
| pxattn p≥0.5 (naive) | 8.42 / 29.35 (19.6 / 21.0) | 8.14 / 19.10 | 8.16 / 23.29 | 8.33 / 12.45 (11.2 / 9.2) |
| pmaxsim p≥0.5 (naive) | 6.30 / 28.34 (21.5 / 23.5) | 6.23 / 19.42 | 6.20 / 25.89 | 6.32 / 17.70 (16.8 / 15.4) |

**Bảng 2** — N=100, so kiểu prompt (WER · B-WER, clean / other):

| | Hou25 paper | Mình |
|---|---|---|
| no distractor (oracle) | 4.93 / 7.64 · 5.59 / 10.02 | 20.08 / 20.89 · 5.12 / 26.24 |
| naive (nguyên list) | 2.67 / 4.78 · 7.78 / 15.21 | 2.55 / 4.20 · 8.04 / 13.98 |
| spoken (nguyên list) | 2.91 / 4.95 · 8.30 / 15.50 | 2.32 / 4.30 · 7.12 / 14.56 |
| pxattn top-50, naive | — | 2.18 / 3.66 · 4.57 / 8.92 |
| pxattn p≥0.5, naive | — | 19.58 / 21.03 · 8.42 / 29.35 |
| pxattn p≥0.5, spoken | — | 2.37 / 3.78 · 5.09 / 9.61 |
| pmaxsim p≥0.5, spoken | — | 2.29 / 3.82 · 4.96 / 9.96 |

**Ablation câu có >1 từ hiếm** (1289 / 1304 câu), B-WER clean / other: nguyên list N100 6.60 / 13.43 · pxattn top-50 N100 4.37 / 8.47, N2000 5.10 / 9.58 · pmaxsim top-50 N100 4.45 / 10.18 · pxattn p≥0.5 N100 naive 7.96 / 11.89 · oracle 4.33 / 8.63 · pxattn p≥0.5 spoken 5.02 / 9.36. Chi tiết mọi hàng trong `score_plain.txt`.

Đọc:
- **Cùng cách chọn top-50, pxattn thắng Hou25 ở cả 8 ô** (B-WER thấp hơn 0.5–0.8 clean, 1.1–1.7 other; WER cũng thấp hơn). pmaxsim thắng 7/8 (thua ở N100 test-other: 10.30 vs 10.0). Baseline mình tự chạy lệch số paper khoảng ±1 theo cả 2 chiều. Chưa có khoảng tin cậy so với Hou25 (không có output từng câu của họ).
- **Prompt rất ngắn làm Whisper lặp chữ** (oracle và p≥0.5 naive: WER 10–23%; ví dụ "…of god or god or god…"; 32/40 câu hỏng nặng nhất có prompt đúng 1 từ). Hou25 Bảng 2 cũng thấy WER tăng ở no distractor nhưng nhẹ hơn nhiều → có thể họ decode có fallback (chưa kiểm). Qwen3-ASR không bị (p≥0.5 là cấu hình tốt nhất ở §4).
- **Bọc prompt spoken chữa được lặp**: p≥0.5 + spoken về WER bình thường (2.3 / 3.8), B-WER 5.0 / 9.6–10.0, vẫn kém top-50 naive.
- Lát >1 từ hiếm: top-50 cải thiện đều như cả tập; thiệt hại của p≥0.5 naive tập trung ở câu ≤1 từ hiếm (prompt 0–1 từ).
