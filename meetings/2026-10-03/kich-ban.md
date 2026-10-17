# Kịch bản báo cáo 03/10 (khoảng 12 phút + hỏi đáp)

**Mạch chung:** mỗi slide trả lời đúng 1 câu thầy sẽ tự hỏi, theo thứ tự này:

> Giao gì, xong chưa? → Đo cái gì? → Số có tin được không? → Số nói gì? → Người khác làm tới đâu? → Em định làm gì, có dựa trên cơ sở nào không? → Khi nào biết là khả thi? → Cần thầy quyết gì?

Nguyên tắc khi nói:
- **Nói kết luận trước** (slide 2), chi tiết sau.
- Mỗi con số đi kèm "nghĩa là gì".
- Chưa chắc thì nói "em chưa kiểm".

| # | Thầy muốn biết | Em nói (1–2 câu) |
|---|---|---|
| 2 | Hôm nay có gì, kết luận là gì? | "Em báo 3 điều: đã làm gì, số nói gì, làm gì tiếp. Kết luận: baseline xong và khớp số chuẩn; Qwen dùng context rất tốt khi list ngắn và đúng, list dài hoặc sai thì vô dụng; em đề xuất bài toán tự dựng list và xin 2 tuần làm feasibility." |
| 3 | Việc thầy giao đã xong chưa? | Đọc nhanh từng dòng. Chỗ đánh dấu ● là chỗ chờ thầy quyết (ConEC) hoặc đang làm (Hou25). |
| 4 | Benchmark đo cái gì? | "Mỗi câu đi kèm một list gồm từ đúng và N từ nhiễu, đưa list vào prompt của Qwen frozen. B-WER là lỗi trên từ hiếm. Có 4 kiểu list: không có, chỉ từ đúng (mốc trên), đúng + nhiễu (giống các paper), chỉ nhiễu (em tự thêm để đo tác hại)." |
| 5 | Số có tin được không? | "Có 3 kiểm tra: WER 1.63 bằng model card; chạy Whisper ra cùng xu hướng Hou25; dùng nguyên scorer chính thức. Chuẩn hoá text làm lệch 0.5–1 điểm nên em chốt chuẩn hoá Whisper." |
| 6 | Kết quả chính? | Chỉ vào đường cong: "Có list đúng thì B-WER xuống 4.01. N=100 còn 5.30. Lên 2000 thì về 7.40, bằng không đưa gì. Chấm đỏ là list sai, còn tệ hơn." |
| 7 | Vậy số nói lên gì? | "Ba điều: context có ích lớn, list dài vô dụng, list sai có hại. Muốn hưởng lợi thì list phải ngắn và có từ đúng. Còn 3.4 điểm B-WER để lấy lại." |
| 8 | Người khác đã làm tới đâu? Mình còn chỗ không? | "Hai paper mạnh nhất đều giả định list cho sẵn và luôn có từ đúng. PTC-Bias trên Qwen đạt 4.10, gần oracle của em (4.01), nên bài toán 'chọn từ trong list' gần như kịch trần. Chỗ trống là khi không có list." **Nói rõ:** số PTC-Bias trích từ paper, khác pipeline. |
| 9 | Em định làm gì? Dựa trên giả định nào? | "Ý tưởng: từ topic hoặc bản decode đầu, cho LLM sinh list, tìm chỗ ASR không chắc rồi chỉ can thiệp ở đó. Ý tưởng này dựa trên 2 giả định: H1 là LLM đoán ra được từ hiếm, H2 là lỗi nằm ở chỗ model không chắc. Method chưa chốt." |
| 10 | Khi nào biết là khả thi? | "Hai tuần, 4 thí nghiệm rẻ. Mỗi cái đều có phương án dự phòng nếu kết quả xấu. Khoảng 17/10 có số thì em chốt method với thầy." |
| 11 | Có kịp hạn nộp không? | Đọc lộ trình. Nhấn 2 mốc kiểm soát: 17/10 và cuối tháng 11. |
| 12 | Em cần gì ở thầy? | Hỏi từng câu, câu 1 quan trọng nhất (chọn bài toán "không có list"). |

## Câu thầy có thể hỏi vặn
- **"Sao biết LLM đoán được tên riêng trong sách?"** → Chưa biết, đó chính là thí nghiệm H1. Nếu không đoán được thì chuyển sang retrieve từ kho 209k.
- **"Kho 209k có rò rỉ test không?"** → Không. Kho lấy từ dữ liệu train của LibriSpeech (README của IS21). Nó phủ khoảng 97% từ hiếm của test.
- **"Khác PTC-Bias ở đâu?"** → PTC-Bias cần list cho sẵn, em thì không. Phần định vị theo thời gian họ đã có, nên em không claim phần đó là mới.
- **"Sao không train?"** → Model frozen, đúng như thầy định hướng ở buổi 17/09. Can thiệp chỉ qua context hoặc logit.
- **"Số của Hou25 lệch với của em?"** → Lệch khoảng 0.3–1 điểm do họ không nêu normalizer và cấu hình decode. Xu hướng thì khớp.
