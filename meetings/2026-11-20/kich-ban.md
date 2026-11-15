# Kịch bản nói, gặp thầy 20/11/2026 (12–15 phút)

> Slide: `hop-thay.pdf`. Số liệu: `scorer/analysis/REPORT-1115.md`. Mỗi slide ~1 phút, riêng slide 8 và 9 mỗi slide ~2 phút.

**Slide 2 – Một ý chính, ba con số**
- Em báo cáo một ý: scorer biết từ nằm ở giây nào thì có lợi ở cả hai đầu: chọn từ tốt hơn, và cho phép đẩy từ vào ASR đúng lúc, không cần prompt.
- Ba con số:
  - Sạch@1 từ 73 lên 89.
  - B-WER khi đưa vào Whisper từ 9.69 xuống 7.53.
  - Không dùng prompt.
- Lưu ý: Hou25 ở đây là bản nhóm tái hiện, chưa phải checkpoint của anh Hoàng.

**Slide 3 – Bài toán và Hou25**
- Hou25 vứt mất vị trí ở hai chỗ: lấy trung bình cả bản đồ thành một số, và đưa top-50 vào prompt.
- Prompt thì phụ thuộc nhiều thứ: Whisper chỉ giữ 223 token cuối, đổi thứ tự là đổi kết quả, mỗi model lại có kiểu prompt riêng.
- Câu hỏi: giữ lại vị trí thì được gì?

**Slide 4 – Đo lại cho công bằng**
- Hou25 báo "từ đúng có lọt top-50 không". Ở mốc đó mọi hệ đều 95–98%, nên khác biệt bị che.
- Em đo Sạch@1: từ đúng có đứng đầu so với mọi từ nhiễu không. Ở mốc này Hou25 cách scorer âm vị 15 điểm.
- Top-50 nghĩa là mỗi câu có khoảng 48 từ nhiễu. Nếu không dùng prompt thì mỗi từ nhiễu bị đẩy vào là một lỗi.

**Slide 5 – Hou25 sai vì đâu**
- Khi từ đúng bị vượt, 96% trường hợp từ vượt lên là từ khác hẳn, không phải từ nghe giống.
- Báo nhầm tăng theo độ dài câu: từ 1.4 lên 12.4 từ mỗi câu. Scorer âm vị gần như phẳng.
- Em diễn giải: lấy trung bình cả bản đồ thì câu dài có nhiều mảng ngẫu nhiên giống vệt.

**Slide 6 – Giả thuyết của anh Hoàng đúng**
- Anh Hoàng chỉ ra bản đồ S1 có vệt chéo đúng chỗ từ được nói.
- Em giữ nguyên Hou25, chỉ đổi bước gộp sang "đọc theo vệt chéo" rồi train lại. Sạch@1 tăng 9.5 điểm, báo nhầm ở câu dài giảm 3 lần, và model định vị được 99%.
- Lấy top-k thì không giúp. Nghĩa là thứ có ích là hình dạng vệt: đúng thứ tự, liền nhau.

**Slide 7 – Scorer hai nhánh**
- Từ đó em làm scorer âm vị hai nhánh:
  - nhánh khớp tự do để không sót từ;
  - nhánh khớp đúng thứ tự để loại nhiễu và cho biên vị trí.
- Khi train, từ đúng chỉ được tính điểm ở đúng đoạn nó được nói.
- Một model đạt 89.3; cộng điểm hai model đạt 90.8. Nhóm từ Whisper nghe sai tăng từ 64.5 lên 70.9.
- Sai số vị trí 0.02 giây, chấm nhanh hơn Hou25 khoảng 200 lần.

**Slide 8 – Lọc list nhờ vị trí**
- Hai từ không thể nằm cùng một chỗ, nên cho chúng tranh nhau. Từ 2000 từ còn khoảng 2 từ mỗi câu, precision 91%.
- Câu A: các từ nhiễu trùng chỗ đều bị loại.
- Câu B là chỗ còn yếu: "flauve" nghe như "fall", "emperour" như "empire". Nhiễu khớp một từ thường trong câu thì luật này chưa loại được.

**Slide 9 – Bảng Whisper**
- Cách đẩy: khi decoder đang nghe trong đoạn của từ thì cộng điểm vào token của từ đó, mỗi từ chỉ đẩy một lần.
- Không prompt: B-WER thấp hơn Hou25 2.16 điểm trên test-other và 1.22 điểm trên test-clean, có ý nghĩa; lỗi ở từ thường không đổi.
- **Nói rõ:** WER tổng chỉ thấp hơn 0.25 và chưa có ý nghĩa thống kê. Thêm prompt thì WER thắng có ý nghĩa.
- Trần (biết đúng từ, đúng chỗ): 5.99.

**Slide 10 – Ba phát hiện**
- Kênh prompt đã bão hoà: scorer tốt hơn 17 điểm mà đưa top-50 vào prompt vẫn chỉ ngang Hou25.
- Whisper cần vị trí: cùng một list, đẩy đúng lúc tốt hơn đẩy mọi lúc 0.42 WER.
- Còn 1.5 điểm tới trần, một nửa do vị trí, một nửa do list.
- Vậy hướng đúng là scorer ra (từ, p, vị trí) và dùng lúc decode.

**Slide 11 – Còn thiếu**
- Đọc nhanh 5 dòng. Nhấn hai điểm:
  - audio không có từ nào trong list thì WER vẫn tệ đi;
  - bản một model còn kém bản cộng điểm 1.5.
- Kể thêm những gì đã thử mà không được.

**Slide 12 – Kế hoạch và xin ý kiến**
- Ba mốc: 20/12 scorer cuối; 17/01 đủ bảng; 05/02 bản nộp.
- Xin thầy ý kiến 3 câu (đọc trên slide). Câu quan trọng nhất là câu 1: đóng góp chính đặt ở đâu.

## Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn | Slide |
|---|---|---|
| Sạch@1 khác Recall thế nào? | Chỉ đếm từ nhiễu đứng trên, nên câu nhiều từ hiếm không bị phạt oan | D1 |
| Từ nghe giống thì sao? | Trên list nghe giống, hai nhánh +3.0; TTS không hơn âm vị | D2 |
| List nhỏ hơn thì sao? | N100: WER 2.96, precision 98% | D2 |
| Tham số chọn ở đâu? | dev-other 400 câu, không chọn trên test | D3 |
| Đã có ai làm chưa? | Vị trí theo frame có ở CTC; PTC dùng vị trí để sửa sau. Chưa thấy ai đẩy đúng lúc cho decoder đóng băng, nhưng mới tra khoảng 10 lượt | D4 |
| So với số Hou25 công bố? | Bản tái hiện khớp xu hướng của paper; chưa có checkpoint của anh Hoàng để so cùng pipeline | – |
