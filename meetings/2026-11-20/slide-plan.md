# Kế hoạch slide gặp thầy 20/11/2026

> Nguồn số: `scorer/analysis/REPORT-1115.md` (mục ghi trong ngoặc).
> Template: Beamer Madrid như `meetings/2026-11-04/hop-thay.tex` (Primary `111B88`, Accent `F2A900`, T5, `pdflatex`).
> Thời lượng: **12–15 phút nói**, 11 slide chính + 4 slide dự phòng.

## Thông điệp (một câu, nói ở slide 1 và nhắc ở slide cuối)

> **Scorer biết từ nằm ở đâu thì vừa chọn từ tốt hơn, vừa cho phép đẩy từ vào ASR đúng lúc mà không cần prompt. Trên IS21 LibriSpeech, cách này thắng Hou25 về B-WER, có ý nghĩa thống kê.**

## Mạch kể

```
Vấn đề (Hou25 bỏ vị trí)  →  Đo lại cho công bằng (R nhỏ)  →  Vị trí có ích ở scorer (3 kết quả)
→  Vị trí có ích ở ASR (bảng Whisper)  →  Còn thiếu gì  →  Kế hoạch đến 09/02  →  Xin ý kiến thầy
```

## Danh sách slide

| # | Tiêu đề | Nội dung chính (1 ý / slide) | Hình / bảng | Nguồn |
|---|---|---|---|---|
| 1 | Scorer có vị trí cho contextual ASR | Thông điệp một câu; 3 con số: Sạch@1 73 → 89, B-WER 9.69 → 7.53, không cần prompt | 3 ô số lớn | mục 1 |
| 2 | Bài toán & mốc Hou25 | Audio + list 2000 từ → chọn từ → ASR. Hou25: scorer TTS + CNN, **lấy trung bình cả bản đồ**, top-50 vào prompt. Mất vị trí | Sơ đồ pipeline Hou25, khoanh đỏ chỗ "average pool" và "prompt" | GOAL §2 |
| 3 | Đo lại cho công bằng | Top-50 che khác biệt: 95.4 so với 98.4. **Sạch@1** (từ đúng có đứng đầu không): 72.8 so với 85.7. Top-50 của Hou25 có precision 3–4% | Biểu đồ cột Sạch@1 / 5 / 10 / 50 cho Hou25 và pmaxsim | mục 2, 3; SMALL_R N1 |
| 4 | Hou25 sai vì đâu | Từ đúng bị vượt bởi từ **khác hẳn** (96%), không phải từ nghe giống. Báo nhầm tăng theo độ dài câu: 1.4 → 12.4 từ/câu | Đường "báo nhầm theo độ dài câu": Hou25 so với âm vị | mục 3; SMALL_R N2–N4 |
| 5 | Kết quả 1: giả thuyết của Hoàng đúng | Chỉ đổi "trung bình" → "đọc theo vệt chéo" rồi train lại: Sạch@1 73.3 → 82.9 (+9.5 [+7.9; +11.2]); định vị 98.9%. Top-k không giúp → **hình dạng vệt** mới là thứ có ích | Bảng 3 cách gộp; hình bản đồ S1 có vệt chéo (tái dùng hình từ slide 04/11) | mục 5 |
| 6 | Kết quả 2: scorer hai nhánh | Nhánh khớp tự do (không sót) + nhánh khớp đúng thứ tự (loại nhiễu, cho biên) + dạy vị trí lúc train. Một model 89.3; cộng điểm 90.8; từ Whisper nghe sai 64.5 → 70.9; định vị 0.02 s | Sơ đồ kiến trúc hai nhánh (vẽ mới) + bảng 5 dòng | mục 6, 6b, 6c, 7c |
| 7 | Kết quả 3: có vị trí thì lọc được list | Hai từ không thể ở cùng một chỗ → cho tranh nhau. 2000 từ → ~2 từ/câu, precision 91%, giữ 91% từ đúng. Theo cỡ list: N100 97.8% → N2000 70% | Ví dụ minh hoạ 1 câu (trục thời gian, các từ tranh nhau) + bảng N100–2000 | mục 4, 7c |
| 8 | Kết quả 4: đưa vào Whisper | Đẩy logit đúng lúc, không prompt: test-other 3.14 / 7.53 so với Hou25 3.39 / 9.69; test-clean 1.59 / 3.93 so với 1.66 / 5.15. ΔB-WER −2.16 [−2.89; −1.54] | **Bảng chính** (6 dòng × 2 tập) | mục 7b |
| 9 | Hai phát hiện về ASR | (a) Kênh prompt bão hoà: scorer tốt hơn 17 điểm mà prompt top-50 vẫn chỉ ngang Hou25. (b) Whisper cần vị trí: đẩy đúng lúc tốt hơn đẩy mọi lúc 0.42 WER. (c) Biết đúng từ + đúng chỗ thì B-WER 5.99: còn 1.5 điểm để lấy | 2 cột nhỏ + vạch "trần" | mục 7, 7b |
| 10 | Còn thiếu | WER tổng chưa có ý nghĩa (−0.17 đến −0.25); audio không có từ nào thì WER vẫn tệ đi 0.25; một model kém bản cộng điểm 1.5; chưa có ckpt Hou25 của Hoàng, chưa thử Qwen3-ASR | Danh sách 4 dòng, mỗi dòng 1 con số | mục 8 |
| 11 | Kế hoạch đến 09/02/2027 & xin ý kiến | Mốc 12/2026: scorer cuối (một model ≥ 90.5, khối "không từ nào" được train, 960h). 01/2027: đủ bảng N100–2000, ckpt Hoàng, Qwen3-ASR. 01–02/2027: viết. **3 câu hỏi cho thầy** (bên dưới) | Timeline ngang 4 mốc | mục 1b |

**Slide dự phòng** (chỉ mở khi thầy hỏi):

| # | Nội dung | Nguồn |
|---|---|---|
| D1 | Định nghĩa các thước đo: Sạch@K, Đúng@chỗ, B-WER (IS21, sub 4 / ins-del 3) | mục 2 |
| D2 | List nghe giống (`hard_N2000`): TTS không hơn âm vị; hai nhánh +3.0 | mục 3, 7c |
| D3 | Chọn tham số trên dev-other (τ, λ, biên), chống lặp khi decode | mục 7 |
| D4 | Tài liệu liên quan & điểm mới: CB-Whisper, CTC-WS, PTC-Bias, Liu 2508.17796 | `related-work/novelty-gate.md` |

## Ba câu hỏi xin ý kiến thầy (slide 11)

1. **Đóng góp chính đặt ở đâu:** scorer hai nhánh có vị trí (phía retrieval), hay "đẩy logit theo vị trí, không cần prompt" (phía ASR), hay gộp cả hai thành một hệ?
2. **Mốc so sánh:** chỉ Hou25, hay thêm PTC-Bias và cách boost không gate kiểu Liu 2508.17796?
3. **Phạm vi:** Whisper + Qwen3-ASR là đủ cho Interspeech, hay cần thêm một SpeechLLM nữa?

## Hình cần làm

| Hình | Cách làm | Dữ liệu |
|---|---|---|
| Pipeline Hou25 (slide 2) | TikZ, sửa từ slide 04/11 | – |
| Cột Sạch@K (slide 3) | matplotlib → PDF | `scorer/analysis/e1_other.md` |
| Báo nhầm theo độ dài câu (slide 4) | matplotlib, 2–3 đường | `e2_other1000.md` mục 2 |
| Kiến trúc hai nhánh (slide 6) | TikZ hoặc /excalidraw-diagram → PNG | `models.py: DualSim` |
| Ví dụ tranh nhau tại chỗ (slide 7) | matplotlib: trục thời gian, mỗi từ là một đoạn màu, từ thắng tô đậm | 1 câu test-other lấy từ `span_N2000.json` của `pdual2_dl` |
| Timeline (slide 11) | TikZ | – |

## Việc cần làm để ra slide

| Việc | Ước giờ |
|---|---|
| Vẽ 6 hình (script trong `meetings/2026-11-20/imgs/`) | 2 |
| Viết `hop-thay.tex` theo bảng trên | 2 |
| Kịch bản nói `kich-ban.md` (mỗi slide 3–5 câu) | 1 |
| Biên dịch, soát số với REPORT-1115 | 0.5 |

## Lưu ý khi trình bày

- Mọi số trên slide phải tra được trong REPORT-1115.md. Ghi rõ "Hou25 = bản tái hiện của nhóm (2 epoch)".
- Tách **đã đo** và **diễn giải**: "kênh prompt bão hoà" là diễn giải từ một phép so; không nói thành định luật.
- Khi nói "thắng Hou25", luôn nói kèm **B-WER**. WER tổng chưa có ý nghĩa thống kê, không được làm tròn thành "thắng".
- Công lao: giả thuyết "đọc theo vệt" là của **Hoàng** (F14); slide 5 ghi rõ.
