# Kịch bản thuyết trình — họp thầy 23/10 (~12 phút + thảo luận)

Ký hiệu: **🔴 NHẤN MẠNH** = câu bắt buộc nói rõ, chậm · ⏱ = thời gian gợi ý · 💬 = câu hỏi thầy có thể hỏi + cách trả lời.

---

## Trang 1 — Tiêu đề (⏱ 10 s)
> "Hôm nay em báo cáo theo mạch: vấn đề, 4 thí nghiệm, insight rút ra, rồi đề xuất hướng tiếp và xin thầy quyết mấy việc."

## Trang 2 — Tóm tắt 1 slide (⏱ 1 phút)
> "Em tóm tắt trước cả buổi trong 1 slide.
> Câu hỏi của em: đưa cho SpeechLLM một list từ lớn, nhiều nhiễu thì vì sao nó không dùng được, và làm sao để dùng được.
> 🔴 **Kết luận chính: có 2 nút thắt nối tiếp. Nút thứ nhất là chọn đúng từ, em đã đưa tới gần trần. Nút thắt đang dịch sang chỗ thứ hai: SpeechLLM dùng từ đã chọn thế nào.**
> Cuối buổi em xin thầy quyết 4–5 việc ở ô vàng."

## Trang 3 — Vấn đề (⏱ 1 phút)
> "Đây là Qwen3-ASR với list IS21 trên LibriSpeech.
> Chỉ đưa đúng từ cần (oracle) thì B-WER từ 7.40 xuống 4.01, nên biasing rất có ích.
> Nhưng list thật thì dài. 🔴 **Ở 2000 từ, B-WER là 7.40, y như không đưa list.** List càng dài, lợi ích càng mất.
> Có 2 kiểu hỏng: từ đúng bị loãng, và mô hình tự chèn từ trong list.
> Giả thuyết ban đầu của em, cũng là giả định của Hou25, PTC, BR-ASR: nút thắt là **chọn từ**."

💬 *"Sao chọn LibriSpeech + IS21?"* → Đây là benchmark chuẩn mà Hou25, PTC, BR-ASR đều dùng, nên so được trực tiếp. Tập thứ hai em xin thầy ý kiến ở cuối.

## Trang 4 — Hình hệ thống + 2 nút thắt (⏱ 1.5 phút) ⭐ QUAN TRỌNG
> "Hệ thống: audio cùng list 2000 từ đi vào **scorer**. Scorer chấm điểm từng từ, chỉ giữ từ có xác suất ≥ 0.5, trung bình khoảng 3 từ/câu, rồi đưa vào prompt của SpeechLLM.
> Thầy nhìn trục bên dưới: test-other từ 16.45 xuống 9.71, oracle là 7.73.
> 🔴 **Phần xanh: em đã lấp được khoảng 75% khoảng trống nhờ chọn từ. Phần đỏ còn lại 25% chính là chỗ em muốn bàn hôm nay.**
> Đường nét đứt màu tím là đề xuất: scorer trả thêm **vị trí** của từ."

## Trang 5 — Kiến trúc scorer (⏱ 1.5 phút)
> "Scorer nhận 1 câu nói và 1 từ, trả ra xác suất từ đó được nói.
> - Phía audio: encoder Whisper-large-v3 **đóng băng**, mỗi 20 ms ra 1 vector.
> - Phía từ: em **không dùng TTS như Hou25** mà đổi từ sang âm vị. Ví dụ *cabinet* là K AE B AH N AH T.
> - Phần train chỉ 2–5 triệu tham số.
> 3 biến thể chỉ khác nhau ở bước so:
> - **pmaxsim:** mỗi âm vị tìm khoảnh khắc giống nó nhất. Rất nhanh nhưng không xét thứ tự âm.
> - **pxattn:** âm vị nhìn vào audio qua cross-attention.
> - **Hou25 tái hiện:** dùng audio TTS và CNN trên ma trận giống nhau.
> 🔴 **Điểm quan trọng: scorer độc lập với SpeechLLM. Nó chỉ trả ra list, nên cắm được vào mô hình nào cũng được.**"

💬 *"Sao encoder đóng băng?"* → Rẻ, train trên 1 GPU, và giữ scorer độc lập. Kết quả 960 h cho thấy chưa cần mở encoder.

## Trang 6 — Train (⏱ 30 s, nói nhanh)
> "Mỗi câu train: từ hiếm của câu là nhãn 1, từ sai lấy gần chính tả hoặc ngẫu nhiên là nhãn 0, loss BCE. Có 2 bản: 100 h giọng sạch, và 960 h có thêm giọng khó."

## Trang 7 — TN1: lọc có cứu được không (⏱ 1 phút)
> "Có. Lọc theo ngưỡng cho B-WER 4.78 / 9.71, lấp khoảng 77% khoảng trống.
> 🔴 **Insight thứ nhất: ít mà đúng tốt hơn nhiều mà đủ.** Giữ 50 từ như Hou25 cho 5.71, còn giữ khoảng 3 từ cho 4.78."

## Trang 8 — TN2: TTS hay âm vị (⏱ 1 phút)
> "So với Hou25 em tái hiện, cùng dữ liệu train:
> 🔴 **Âm vị tìm được từ đúng nhiều hơn (98.4 so với 95.4) và nhanh hơn khoảng 200 lần, không cần 21 GB feature TTS.**
> Ngược lại, Hou25 ít bị lừa bởi từ nghe giống hơn. Em đoán do CNN nhìn được thứ tự âm, nhưng chưa kiểm."

💬 *"Hou25 tái hiện có chuẩn không?"* → Paper không có code, nên em tự cài theo mô tả. Mới train 2 epoch, em ghi rõ trong giới hạn. Hou25 không báo recall, nên đây là số recall đầu tiên của Hou25.

## Trang 9 — TN3: vì sao thua PTC (⏱ 1.5 phút) ⭐ QUAN TRỌNG
> "Em kiểm lần lượt từng giả thuyết:
> - Hiệu chỉnh điểm và loss: thử rồi đều tệ hơn, **loại**.
> - Scorer nghe kém giọng khó: 77–85% từ đúng bị rớt có điểm gần 0.
> - Vậy có phải do thiếu dữ liệu? Em train lại trên 960 h, **giữ nguyên số bước** để so công bằng.
> 🔴 **Kết quả: @99 giảm gần một nửa, từ 100 xuống 54 từ/câu, và không mất gì ở khả năng phân biệt từ nghe giống. Tức thiếu dữ liệu là một nguyên nhân thật.**"

## Trang 10 — Vì sao @99 còn 54 (⏱ 1 phút)
> "@99 rất nhạy: chỉ khoảng 52 từ đúng tệ nhất quyết định con số này. Em liệt kê ra thì phần lớn là **lỗi dữ liệu**: *could n't* bị tách nên g2p đọc thành chữ cái, chính tả giọng địa phương, câu dài hơn 30 giây bị encoder cắt.
> 🔴 **Mô phỏng bỏ các từ lỗi này thì @99 còn khoảng 20, gần PTC (16.9). Số này em không báo cáo. Nhưng nó cho thấy khoảng cách với PTC chủ yếu là lỗi tiền xử lý, sửa rẻ.**"

💬 *"Bỏ từ khỏi benchmark là gian?"* → Đúng, nên em chỉ dùng để chẩn đoán. Cách sửa hợp lệ là âm vị đúng cho mảnh từ rút gọn và cắt audio dài. PTC cũng gặp đúng các từ này.

## Trang 11 — TN4 + insight tổng hợp (⏱ 1.5 phút) ⭐ QUAN TRỌNG NHẤT
> "Sau khi chọn tốt, em phân rã từng lỗi còn lại: khoảng một nửa là scorer bỏ sót, nửa kia là **từ đã có trong list mà SpeechLLM vẫn chép sai**.
> 🔴 **Đây là insight chính của em: bài toán có 2 nút thắt nối tiếp. Các paper hiện tại chủ yếu nhắm nút thứ nhất. Em đã đưa nút thứ nhất tới gần trần với chi phí rất thấp. Nút thắt đang dịch sang chỗ thứ hai.**"
(Dừng 2 giây, để thầy phản ứng.)

## Trang 12 — Đề xuất (⏱ 1.5 phút)
> "Đề xuất: scorer trả thêm **khoảng thời gian** của từ. Khi đó SpeechLLM chỉ bias đúng đoạn đó: không bias chỗ không có từ thì giảm chèn sai, chỉ rõ chỗ cần sửa thì giảm chép sai.
> Theo em kiểm sơ bộ, PTC có khoảng thời gian nhưng chỉ dùng để sửa sau decode, còn Novitasari có tagger nhưng bỏ đi khi suy luận.
> 🔴 **Nhưng em chưa muốn làm ngay. Em đề xuất 1 thí nghiệm đo trần, khoảng 1 ngày GPU: đưa oracle kèm vị trí thật vào Qwen, xem vị trí lấp được bao nhiêu. Lấp được nhiều thì đi tiếp, không thì em chuyển hướng.**"

💬 *"Novelty có chắc không?"* → Mới kiểm sơ bộ. Nếu thầy đồng ý hướng này, em sẽ quét kỹ trước khi code.

## Trang 13 — Cần thầy quyết (⏱ còn lại: thảo luận)
> "Em xin thầy quyết 5 việc, mỗi việc em đã ghi đề xuất của mình."
> 🔴 **Hỏi kỹ nhất câu 1 và câu 3:**
> 1. Chuyển trọng tâm sang nút thắt thứ hai được không.
> 3. SpeechLLM giữ đóng băng hay được fine-tune / RL. Câu này quyết định cả hướng method.

Trước khi thầy hỏi, chủ động nói giới hạn:
> "Số PTC em chép từ paper. PTC dùng Qwen 0.6B nên B-WER không so trực tiếp. Bản 960 h mới 1 epoch."

---

## 3 câu phải thầy nhớ được sau buổi họp
1. **Chọn từ bằng audio đã gần trần**, rẻ, tốt hơn Hou25.
2. **Nút thắt đang dịch sang "SpeechLLM dùng từ đã chọn"** (khoảng nửa lỗi còn lại).
3. **Đề xuất vị trí của từ + 1 ngày GPU để kiểm** trước khi làm.

## Ghi lại trong buổi họp
- Thầy quyết gì cho 5 câu ở trang 13.
- Thầy phản bác insight nào.
- Việc giao tiếp theo + hạn.

→ Sau họp ghi vào `meetings/2026-10-23.md`.
