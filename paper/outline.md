# Paper outline — Interspeech 2027 (hạn 09/02/2027)

> Interspeech: 4 trang nội dung + trang tham khảo. Template chính thức tải khi hội nghị công bố.

## Title (tạm)
(chờ chốt method)

## 1. Introduction
- Vấn đề: contextual biasing giả định có bias list sạch; thực tế list không có sẵn hoặc lớn, nhiều nhiễu.
- Bằng chứng: Qwen3-ASR oracle vs N=100 (B-WER 4.01 vs 5.30, test-clean).
- Đóng góp: (chờ chốt method)

## 2. Related work
Xem `related-work/README.md`.

## 3. Method
(chờ chốt)

## 4. Experimental setup
- Data: LibriSpeech + IS21; (domain thật: ConEC?)
- Model: Qwen3-ASR-1.7B frozen.
- Metric: WER / B-WER / U-WER (scorer IS21), Recall@K cho candidate.
- Baseline: no context, prompt list, PTC-Bias, Hou25 (tự implement), CTC-WS.

## 5. Results
- Bảng chính: N ∈ {100, 500, 1000, 2000} + missing + kho 209k.
- Ablation: (chờ)

## 6. Conclusion
