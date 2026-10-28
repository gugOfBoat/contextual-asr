# STATE — hiện trạng nghiên cứu

> Đọc file này ĐẦU mỗi phiên, cập nhật CUỐI mỗi phiên. Cập nhật lần cuối: 28/10/2026.

## 1. Mục tiêu
- **Venue:** Interspeech 2027, hạn nộp **09/02/2027** (còn ~18 tuần tính từ 02/10/2026).
- **Bài toán (sửa 18/10 theo user):** contextual ASR cho **SpeechLLM nói chung** khi bias list **lớn và nhiều nhiễu** — điểm chung của Hou25, PTC-Bias, BR-ASR, Huber & Waibel, Kong et al. / RLBR, Novitasari (ta diễn giải): list dài → (1) từ đúng bị loãng, (2) mô hình tự chèn từ trong list → cần chọn / định vị bias word dựa vào audio, scale được. (Hướng cũ 17/09 "không có list sẵn → tự dựng list" tạm gác.)
- **Đóng góp đang xây:** scorer độc lập với mô hình ASR (audio + list → điểm từng từ; hướng mở rộng: biết vị trí từ).
- **Backbone thí nghiệm:** `Qwen/Qwen3-ASR-1.7B` (đang dùng), sau sẽ thêm Whisper (như Hou25) / SpeechLLM khác; đóng băng hay gating / RL là câu hỏi thực nghiệm ở bước sau, không phải định nghĩa bài toán.
- **Định hướng từ họp 17/09:** 2 nguồn candidate = (1) retrieve từ corpus, (2) generate từ phân bố xác suất T×V của decoder → ranking; LLM xác định từ có khả năng sai. Chi tiết: `meetings/2026-09-17.md`.

## 2. Đã chốt
- Benchmark chính: LibriSpeech test-clean/other + IS21 (Le et al. 2021), N ∈ {100, 500, 1000, 2000}, thêm `missing_N` và kho 209k từ (`bench/README.md`).
- Metric: WER / B-WER / U-WER bằng scorer IS21 chính thức; normalizer = Whisper EnglishTextNormalizer (wnorm).
- Paper mốc cần vượt: **Hou25** (Interspeech 2025, ranking & selection bias words) và **PTC-Bias** (arXiv 2609.28727, training-free). Xem `related-work/README.md`.

## 3. Chưa chốt
- **Method** (ứng viên trong `method/`; bản mới nhất `method/PROPOSE-huong-nghien-cuu-mai.md` 01/10). Khung đề xuất: list-free cho từng câu (H1: tự sinh list) + can thiệp có chọn lọc theo độ bất định (H2). Bản đồ gap: `related-work/gap-map-2026-10-03.md`. Lưu ý: RAG context discovery (2509.19567) đã làm "tự tìm context từ segment trước", nên không claim phần đó.
- Bộ domain thật thứ hai: ConEC (Earnings-21/22) chưa tải. Ứng viên "context thật" cho LibriSpeech: UserLibri (văn bản phần còn lại của sách).

## 4. Số liệu hiện có (`bench/results.md`, wnorm, clean / other) — đủ 7 điều kiện
| Qwen3-ASR-1.7B | WER | B-WER |
|---|---|---|
| không context | 1.63 / 3.41 | 7.40 / 16.08 |
| oracle | 1.22 / 2.43 | 4.01 / 7.73 |
| N=100 | 1.42 / 3.17 | 5.30 / 11.61 |
| N=500 | 1.60 / 3.33 | 6.73 / 14.88 |
| N=1000 | 1.67 / 3.45 | 7.30 / 15.92 |
| N=2000 | 1.69 / 3.55 | 7.40 / 16.45 |
| missing_N100 | 1.73 / 3.50 | 7.62 / 16.34 |
| PTC-Bias (N=2000, theo paper, **Qwen3-ASR-0.6B** + WavLM) | — | 4.10 / 7.76 |

Output + log của các lượt chạy: `bench/out/`, `bench/logs_4090/`.

**16–17/10 — scorer (việc thầy giao: audio + bias list → điểm liên quan):** `scorer/`, kết quả `scorer/results.md`, file `scorer/runs/2026-10-16/`.
- pxattn (âm vị thay TTS, cross-attention) + lọc p ≥ 0.5 trên N2000: Qwen B-WER **4.78 / 9.71** (top-50 kiểu Hou25: 5.71 / 11.57; oracle 4.01 / 7.73). Số PTC-Bias (CTC-Filter 4.81 / 9.45, PTC 4.10 / 7.76) là trên Qwen3-ASR-**0.6B** → không so trực tiếp. Protocol đánh giá thống nhất: `EVAL_PROTOCOL.md`.
- hxattn (thêm hyp lần 1, nhãn "khó"): kém (6.58 / 14.07), nghi học tắt theo hyp. Nhãn khó: Qwen sai 7,0% từ hiếm trên train-clean-100 (≈ dev 7,3%).
- Bản tái hiện Hou25 (TTS) chưa train: feature đã xoá. Slide họp 17/10: `meetings/2026-10-17/`.
- Máy GPU vẫn giữ LibriSpeech + checkpoint + output để chạy tiếp.
- 17/10: repo đã đưa lên GitHub (`.gitignore`, `README.md`, `NEXT_STEPS.md`, `bench/data/README.md`, `scorer/runs/README.md`). Tên thành viên, đường dẫn máy, chi tiết server chỉ để ở `AGENTS.local.md` (không lên git).
- 17/10 tối: protocol đánh giá `EVAL_PROTOCOL.md` + công cụ (`bench/build_lists.py`, `eval_retrieval.py`, `compare.py`, `score.py --ctx`, `baseline_hypmatch.py`). Kết quả `scorer/results.md` §6: chọn từ trên test-other còn thua PTC (Recall_B#50 97.7–98.6 vs 99.3); giá trị của scorer nằm ở từ khó.
- 18/10: **chốt thực nghiệm phần scorer** (`EVAL_PROTOCOL.md` §7: bảng chọn từ + ablation, chọn trên dev-other; phần ASR chưa bàn) và **phân tích bản chất scorer** (`scorer/DEEPDIVE.md`). Lỗi chính trên test-other là scorer nghe kém từ đúng (77–85% từ bị rớt có điểm < 0.3), từ gần đồng âm chỉ ~5%. Kế hoạch E0–E5.
- 18/10 đêm (GPU 0 + 2): **Hou25 tái hiện** (TTS + CNN) test-other N2000: @99 180.8 / #50 95.39 / H#50 **17.59**; pmaxsim 99.8 / **98.44** / 25.52; PTC 16.9 / 99.3 / 22.0. Âm vị tìm từ đúng tốt hơn Hou25, Hou25 ít bị lừa bởi từ nghe giống hơn. E3 (loss theo list) tệ hơn → bỏ. @99 thua PTC xa do ~1% từ đúng không nghe ra (`scorer/results.md` §8). Container đã xoá, artifact còn trong `~/model_tmp`.
- **Bước tiếp:** chạy scorer trên dev (chọn ckpt + θ) và list `hard_N2000` (cần GPU); tầng (c) với list oracle; đọc UGPCB / COALA / Contextual Earnings-22 → `NEXT_STEPS.md`.
- 28/10 (GPU 0): **Bảng 1/2 kiểu Hou25 trên Whisper-turbo**: top-50 pxattn thắng Hou25 cả 8 ô B-WER (N100 4.57 / 8.92 vs 5.4 / 10.0; N2000 5.29 / 10.00 vs 5.8 / 11.7), pmaxsim 7/8. Prompt 0–1 từ làm Whisper lặp chữ → p≥0.5 hỏng trên Whisper (Qwen không bị), prompt spoken chữa được. Ablation câu >1 từ hiếm trong `scorer/results.md` §10.

## 5. Ba việc tiếp theo
1. Ghi biên bản họp 03/10 (`meetings/2026-10-03.md`) theo quyết định của thầy.
2. Feasibility 2 tuần (tới ~17/10): H1 (LLM sinh list → Recall@K, B-WER), H2 (log-prob lần 1 định vị từ hiếm sai), chạy code PTC-Bias, tải UserLibri và đo độ phủ.
3. Đọc toàn văn: Huber & Waibel (2608.05759, có Qwen3-ASR), LOGIC (2601.15397), Novitasari (2604.12398).
