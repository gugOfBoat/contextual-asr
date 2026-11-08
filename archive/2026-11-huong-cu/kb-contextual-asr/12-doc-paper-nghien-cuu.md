# Chương 12 — Đọc paper như một người làm nghiên cứu

*Mục tiêu của chương này không phải chọn đề tài. Mục tiêu là đọc đủ sâu để biết tác giả bắt đầu từ đâu, họ đã chứng minh được gì và câu hỏi nào vẫn chưa có bằng chứng.*

## 12.1 Phân biệt bốn tầng công việc

1. **Nền tảng:** hiểu CTC/RNN-T/Speech-LLM, biasing, retrieval và metric.
2. **Survey:** biết những nhóm phương pháp nào đang tồn tại.
3. **Deep dive:** tái dựng logic của từng paper từ problem đến evidence.
4. **Synthesis/proposal:** chỉ làm sau khi nhiều deep dive cho thấy một pattern lặp lại.

Knowledge base chương 1–11 phục vụ tầng 1. Các báo cáo landscape trong `4-papers` phục vụ tầng 2. Chương này tổ chức tầng 3. Hiện tại **chưa thực hiện tầng 4**.

## 12.2 Một paper phải được đọc theo chuỗi nào

Không đọc tuần tự từ Abstract tới References. Đọc theo chuỗi kiểm chứng:

```text
Hiện tượng / failure
        ↓
Câu hỏi nghiên cứu của tác giả
        ↓
Giả thuyết cơ chế
        ↓
Thay đổi kỹ thuật tối thiểu
        ↓
Thiết kế thí nghiệm có cô lập thay đổi đó không?
        ↓
Kết quả nào thực sự hỗ trợ giả thuyết?
        ↓
Kết quả nào chưa giải thích được?
```

Sau khi đọc, phải trả lời được bảy câu:

1. Baseline hỏng ở tình huống cụ thể nào?
2. Tác giả cho rằng nguyên nhân là gì?
3. Method thay đúng mắt xích nào?
4. Supervision đến từ đâu?
5. Bảng/ablation nào là bằng chứng quan trọng nhất?
6. Có cách giải thích khác cho cùng kết quả không?
7. Paper chỉ chứng minh trên phạm vi nào?

## 12.3 Reading map method-first 2025–2026

### Tuyến A — Dynamic vocabulary và contextualization hiệu quả

1. `deep-BR-ASR-Gong2025.md` — retrieve candidate khi list rất lớn.
2. [`deep-dynac-interspeech2025.md`](../4-papers/frontier-2026/deep-dynac-interspeech2025.md) — dynamic phrase token và intermediate self-conditioned CTC.
3. [`deep-coala-interspeech2026.md`](../4-papers/frontier-2026/deep-coala-interspeech2026.md) — absolute scoring cho nhiều positive entity.

Câu hỏi mang theo: retrieval, representation và scoring đang giải ba failure khác nhau nào; paper có báo RTF/latency hay chỉ B-WER?

### Tuyến B — Personalized contextual ASR trong speech chồng lấn

1. CMT-LLM (Interspeech 2025) — unified contextual multi-talker recognition.
2. [`deep-calm-icassp2026.md`](../4-papers/frontier-2026/deep-calm-icassp2026.md) — joint target-speaker acoustic conditioning và linguistic biasing.

Câu hỏi mang theo: speaker identity có thực sự quyết định phrase context nào được dùng, hay hai nguồn context mới chỉ được ghép vào cùng model?

### Tuyến C — Chọn, nén và làm sạch conversational history

1. `deep-paperA-flynn-ragni-2023.md` và `deep-paperB-lee-2024.md` — nền về history và noisy transcript.
2. `deep-mars-aaai2026.md` — retrieve/select acoustic và textual history.
3. [`deep-conversation-compression-2026.md`](../4-papers/frontier-2026/deep-conversation-compression-2026.md) — nén mỗi prior audio turn thành K latent tokens.
4. [`deep-noisy-context-speechllm-2026.md`](../4-papers/frontier-2026/deep-noisy-context-speechllm-2026.md) — train với history lỗi, dropout và DPO.

Câu hỏi mang theo: selection giảm context không liên quan, compression giảm cost và robust training giảm error propagation—paper có cô lập được ba tác dụng này không?

### Tuyến D — Broad context và reasoning

1. `deep-common-word-cues-icassp2026.md` — cue đơn giản và vị trí bias word.
2. [`deep-metadata-reasoning-interspeech2026.md`](../4-papers/frontier-2026/deep-metadata-reasoning-interspeech2026.md) — reasoning supervision từ mô tả video rộng.
3. [`deep-indiccontexteval-interspeech2026.md`](../4-papers/frontier-2026/deep-indiccontexteval-interspeech2026.md) — benchmark phụ trợ để kiểm tra grounding bằng context đúng/sai.

Câu hỏi mang theo: context giúp vì acoustic grounding hay chỉ vì language prior đủ mạnh để rewrite transcript?

### Tuyến E — Inference-time memory và post-ASR correction

1. [`deep-voice-memory-2026.md`](../4-papers/frontier-2026/deep-voice-memory-2026.md) — text memory, act/abstain và over-correction.
2. Đọc tiếp ontology memory để so memory dạng rule với graph/relation memory.

Câu hỏi mang theo: memory lưu gì, cập nhật bằng supervision nào và còn dựa vào acoustic evidence không?

## 12.4 Sáu method papers nên đọc trước

| Thứ tự | Paper | Trạng thái | Method/bài toán cần hiểu |
|---:|---|---|---|
| 1 | DYNAC | Interspeech 2025 | Dynamic vocabulary NAR nhanh, giữ rare/unseen phrase mà không phá U-WER |
| 2 | CALM | ICASSP 2026 | Joint acoustic–linguistic personalization cho multi-speaker overlap |
| 3 | MARS | AAAI 2026 | Retrieval/selection của acoustic và textual conversation history |
| 4 | Distilling Conversations | Preprint 2026 | Fixed-latent compression cho prior audio history |
| 5 | Metadata-Driven Reasoning Chains | Interspeech 2026 | Reasoning supervision để dùng broad metadata |
| 6 | Voice Memory | Preprint 2026 | Inference-only memory và quyết định act/abstain |

Benchmark như IndicContextEval được đọc **sau** vòng method để thiết kế evaluation/negative control. Thứ tự mới đi từ kiến trúc contextual ASR chuyên biệt → bài toán multi-speaker → long context → reasoning → memory mới nổi.

## 12.5 Mẫu ghi chú deep dive

Mỗi hồ sơ trong `4-papers/frontier-2026` dùng cùng cấu trúc:

1. Thẻ paper và trạng thái peer review.
2. **Phiếu đọc nhanh bốn mục bắt buộc:** bài toán, thực nghiệm, dataset/model/metric và kết quả định lượng.
3. Vì sao paper xuất hiện vào thời điểm này.
4. Baseline/failure mà tác giả bắt đầu.
5. Method, loss và dữ liệu supervision.
6. Experimental contract: dataset, model, metric, baseline.
7. Ba bằng chứng quan trọng nhất.
8. Paper chứng minh được gì.
9. Paper chưa chứng minh được gì.
10. Câu hỏi cần mang sang paper kế tiếp.

Phiếu đọc nhanh phải đủ để tạo một slide paper độc lập:

```text
Bài toán: baseline hỏng ở đâu?
Thực nghiệm: tác giả thay biến nào và giữ biến nào?
Dataset/model/metric: kết quả có hiệu lực trong phạm vi nào?
Kết quả: số nào trả lời trực tiếp câu hỏi nghiên cứu?
```

Phần cuối không ghi “ý tưởng đề tài”. Chỉ ghi **câu hỏi chưa được paper trả lời**, tránh biến limitation của tác giả thành novelty claim quá sớm.

## 12.6 Khi nào mới đủ dữ liệu để làm slide synthesis

Chỉ bắt đầu slide khi có thể lập một bảng ít nhất ba paper với các cột:

| Paper | Failure đầu vào | Context type | Intervention | Supervision | Evidence chính | Boundary |
|---|---|---|---|---|---|---|

Nếu chưa điền được cột `Evidence chính` và `Boundary` bằng số/bảng cụ thể, tức là mới đọc abstract chứ chưa deep dive.

## 12.7 Quy tắc dừng trước proposal

Sau mỗi paper, chỉ kết luận một trong ba dạng:

- “Paper đã chứng minh X trong điều kiện Y.”
- “Kết quả gợi ý X nhưng thí nghiệm chưa cô lập được Y.”
- “Paper chưa kiểm tra Z.”

Không viết “ta sẽ làm Z” cho tới khi paper kế tiếp xác nhận Z chưa bị giải quyết ở nơi khác.
