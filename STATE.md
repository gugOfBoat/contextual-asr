# STATE — hiện trạng nghiên cứu

> Đọc file này ĐẦU mỗi phiên, cập nhật CUỐI mỗi phiên. Cập nhật lần cuối: 03/10/2026.

## 1. Mục tiêu
- **Venue:** Interspeech 2027, hạn nộp **09/02/2027** (còn ~18 tuần tính từ 02/10/2026).
- **Bài toán:** contextual biasing cho từ hiếm / tên riêng trên ASR **đóng băng**, khi **không có bias list cho sẵn** (hoặc list lớn, nhiều nhiễu) → hệ thống phải tự tạo/chọn candidate.
- **Model nền:** `Qwen/Qwen3-ASR-1.7B` (frozen), tiếng Anh.
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
| PTC-Bias (N=2000, theo paper) | — | 4.10 / 7.76 |

Output + log của các lượt chạy: `bench/out/`, `bench/logs_4090/`.

**16–17/10 — scorer (việc thầy giao: audio + bias list → điểm liên quan):** `scorer/`, kết quả `scorer/results.md`, file `scorer/runs/2026-10-16/`.
- pxattn (âm vị thay TTS, cross-attention) + lọc p ≥ 0.5 trên N2000: Qwen B-WER **4.78 / 9.71** (top-50 kiểu Hou25: 5.71 / 11.57; oracle 4.01 / 7.73). Ngang CTC-Filter trong PTC-Bias (4.81 / 9.45), thua PTC-Bias (4.10 / 7.76).
- hxattn (thêm hyp lần 1, nhãn "khó"): kém (6.58 / 14.07), nghi học tắt theo hyp. Nhãn khó: Qwen sai 7,0% từ hiếm trên train-clean-100 (≈ dev 7,3%).
- Bản tái hiện Hou25 (TTS) chưa train: feature đã xoá. Slide họp 17/10: `meetings/2026-10-17/`.
- Máy GPU vẫn giữ LibriSpeech + checkpoint + output để chạy tiếp.
- 17/10: repo đã đưa lên GitHub (`.gitignore`, `README.md`, `NEXT_STEPS.md`, `bench/data/README.md`, `scorer/runs/README.md`). Tên thành viên, đường dẫn máy, chi tiết server chỉ để ở `AGENTS.local.md` (không lên git).
- **Bước tiếp:** setup chuẩn so với Hou25 → `NEXT_STEPS.md`.

## 5. Ba việc tiếp theo
1. Ghi biên bản họp 03/10 (`meetings/2026-10-03.md`) theo quyết định của thầy.
2. Feasibility 2 tuần (tới ~17/10): H1 (LLM sinh list → Recall@K, B-WER), H2 (log-prob lần 1 định vị từ hiếm sai), chạy code PTC-Bias, tải UserLibri và đo độ phủ.
3. Đọc toàn văn: Huber & Waibel (2608.05759, có Qwen3-ASR), LOGIC (2601.15397), Novitasari (2604.12398).
