# Họp thầy 23/10 — Vấn đề → Thí nghiệm → Insight → Đề xuất

## 1. Vấn đề
**Câu hỏi nghiên cứu:** SpeechLLM được đưa bias list lớn, nhiều nhiễu thì vì sao không dùng được, và làm sao để dùng được?

**Bằng chứng vấn đề có thật** (Qwen3-ASR-1.7B, LibriSpeech + list IS21, B-WER clean / other):

| List | 0 từ | oracle (chỉ từ đúng) | N100 | N500 | N1000 | N2000 |
|---|---|---|---|---|---|---|
| B-WER | 7.40 / 16.08 | **4.01 / 7.73** | 5.30 / 11.61 | 6.73 / 14.88 | 7.30 / 15.92 | 7.40 / 16.45 |

List càng dài, lợi ích càng mất. Ở N2000, đưa list cũng như không. Khoảng trống cần lấp: 7.40 → 4.01 (clean), 16.45 → 7.73 (other).

**Giả thuyết ban đầu:** nút thắt là **chọn từ**. Nếu dùng audio lọc list còn vài từ đúng thì SpeechLLM dùng được như oracle. Đây cũng là giả định chung của Hou25, PTC-Bias, BR-ASR.

## 2. Thí nghiệm và insight

### TN1 — Lọc list bằng audio có cứu được không?
- **Thiết kế:** scorer (audio + list → điểm từng từ) giữ từ có p ≥ 0.5, so với top-50 kiểu Hou25. Đưa list đã lọc vào Qwen.
- **Kết quả (B-WER clean / other):**

| | B-WER |
|---|---|
| N2000 | 7.40 / 16.45 |
| top-50 kiểu Hou25 | 5.71 / 11.57 |
| **lọc theo ngưỡng** | **4.78 / 9.71** |
| oracle | 4.01 / 7.73 |

- **Insight:** lọc bằng audio lấp được ~75% khoảng trống. **Ít mà đúng tốt hơn nhiều mà đủ**: top-50 kém hơn hẳn lọc theo ngưỡng (~3 từ/câu).

### TN2 — Biểu diễn từ thế nào: audio TTS (Hou25) hay âm vị?
- **Thiết kế:** cùng dữ liệu train (100 h), đo khâu chọn từ bằng metric của PTC / BR-ASR. Test-other, list 2000.

| | Recall_B#50 ↑ | Recall_H#50 ↓ | Tốc độ |
|---|---|---|---|
| Hou25 tái hiện (TTS + CNN) | 95.4 | **17.6** | 922 ms/câu |
| âm vị (của nhóm) | **98.4** | 25.5 | 4 ms + encoder |
| PTC (paper) | 99.3 | 22.0 | |

#50 = % từ đúng nằm trong top-50. H#50 = % từ nghe giống từ đúng lọt top-50.

- **Insight:** âm vị **tìm từ đúng tốt hơn**, rẻ hơn ~200 lần, chấm ngay được từ mới. Hou25 **ít bị lừa bởi từ nghe giống** hơn. Giả thuyết (chưa kiểm): CNN của Hou25 nhìn được thứ tự âm.

### TN3 — Vì sao khâu chọn từ còn thua PTC?
Kiểm lần lượt từng giả thuyết:

| Giả thuyết | Kiểm bằng | Kết quả |
|---|---|---|
| Điểm lệch giữa các câu (hiệu chỉnh) | chuẩn hoá điểm theo câu | tệ hơn → **loại** |
| Loss chưa hợp bài toán 1 / 2000 | loss theo list + 200 từ nhiễu | tệ hơn → **loại** |
| Scorer **nghe kém** giọng khó | phân tích từ bị rớt | 77–85% từ đúng bị rớt có điểm < 0.3 |
| → do **thiếu dữ liệu** | train 100 h → 960 h, cùng số bước | @99: 100 → **54** từ/câu; #50: 98.4 → 98.8 |
| Phần đuôi còn lại | xem 52 từ tệ nhất | phần lớn là **lỗi dữ liệu**: *could n't* bị tách, g2p đọc sai; chính tả giọng địa phương; câu > 30 s bị encoder cắt |

@99 = số từ/câu phải giữ để không sót quá 1% từ đúng. PTC: 16.9.

- **Insight:** khoảng cách với PTC chủ yếu do **dữ liệu train + tiền xử lý**, không do kiến trúc. Loại các từ lỗi dữ liệu thì @99 ≈ 20 (chỉ mô phỏng, không báo cáo).
- **Khâu chọn từ đã gần trần.**

### TN4 — Chọn tốt rồi thì khoảng trống còn lại nằm ở đâu?
- **Thiết kế:** phân rã từng lỗi từ hiếm sau khi đưa list đã lọc vào Qwen.
- **Kết quả:** lỗi còn lại chia khoảng đôi:
  - (a) scorer bỏ sót từ;
  - (b) **từ đã có trong list mà Qwen vẫn chép sai** hoặc bị từ nhiễu kéo đi.
- **Insight:** cải thiện khâu chọn chỉ gỡ được (a). Nửa (b) là chuyện **SpeechLLM dùng list thế nào**, và chưa paper nào trong nhóm mốc giải quyết trực tiếp.

## 3. Insight tổng hợp
> Bài toán list lớn có **2 nút thắt nối tiếp**: (1) chọn đúng từ, (2) SpeechLLM dùng đúng từ đã chọn.
> Các paper hiện tại (Hou25, PTC, BR-ASR) tập trung vào (1). Nhóm đã đưa (1) tới gần trần với chi phí rất thấp.
> **Nút thắt đang dịch sang (2).**

## 4. Đề xuất: biasing có **vị trí** (position-aware)
- **Ý tưởng:** scorer trả ra (từ, điểm, **khoảng thời gian**). SpeechLLM chỉ được nhắc từ đó ở đúng đoạn audio đó, hoặc decode lại cục bộ đoạn đó.
- **Vì sao đánh đúng vào (2):**
  - Không bias ở chỗ không có từ → giảm chèn từ sai.
  - Chỉ rõ chỗ cần sửa → giảm chép sai từ đã có trong list.
- **Vì sao mới** (kiểm sơ bộ, chưa quét kỹ):
  - PTC có khoảng thời gian nhưng chỉ dùng để sửa sau decode, không đo độ chính xác định vị.
  - Novitasari (2604.12398) có tagger vị trí nhưng bỏ đi khi suy luận.
- **Kiểm trước khi làm — thí nghiệm đo trần** (~1 ngày GPU, hạ tầng có sẵn):

| Đưa vào Qwen | Trả lời |
|---|---|
| oracle | trần lý tưởng |
| oracle + 5 / 20 / 50 từ nhiễu | nhiễu ít đã hỏng bao nhiêu |
| oracle + 5 từ nghe giống | hỏng do đồng âm |
| **oracle + vị trí thật** (forced aligner) | vị trí lấp được bao nhiêu |

- **Tiêu chí đi tiếp:**
  - Nếu vị trí lấp được phần đáng kể khoảng trống (b): làm hướng này.
  - Nếu không: viết paper theo hướng chọn từ + phân tích, và tìm đóng góp khác.

## 5. Cần thầy quyết
| # | Câu hỏi | Em đề xuất |
|---|---|---|
| 1 | Dừng tối ưu khâu chọn từ, chuyển trọng tâm sang (2)? | Có. Chỉ sửa nốt 2 lỗi tiền xử lý (rẻ) |
| 2 | Hướng position-aware có đáng làm? | Chạy thí nghiệm đo trần trước rồi quyết |
| 3 | SpeechLLM giữ đóng băng hay được fine-tune / RL? | Đóng băng cho bản chính (cắm được nhiều backbone); RL / gating là phần mở rộng |
| 4 | Hướng "tự dựng list khi không có list" (họp 17/09) | Gác, để tập trung |
| 5 | Tập thứ hai ngoài LibriSpeech | Thêm Earnings / ConEC trước khi nộp |

## 6. Giới hạn của số liệu
- Số PTC chép từ paper (2609.28727 Bảng 3), chưa chạy lại code của họ. PTC dùng Qwen3-ASR-0.6B, nhóm dùng 1.7B → B-WER không so trực tiếp.
- Scorer 960 h: 1 epoch, 1 seed. Hou25 tái hiện: 2 epoch.
- Số B-WER ở TN1 là scorer bản 100 h, chưa đưa bản 960 h vào ASR.
