# Kịch bản báo cáo 17/10 — scorer chấm bias word (~10 phút)

Slide: `scorer-bao-cao.pdf` (10 trang). Mỗi slide: **nói gì** + **nếu thầy hỏi**.

---

## Slide 1 — Tiêu đề (10 giây)
> "Hôm nay em báo cáo việc thầy giao buổi trước: scorer chấm bias word theo audio."

---

## Slide 2 — Thầy giao gì, em làm tới đâu (40 giây)
> "Thầy giao em cài một scorer: đầu vào là audio và bias list, đầu ra là điểm của từng từ, cho biết từ nào có khả năng liên quan đến audio.
> Em đã làm xong và train xong trên LibriSpeech train-clean-100, 28.539 câu. Đầu ra là xác suất từ 0 đến 1: từ này có được nói trong audio không.
> Em đo 2 tầng: một là scorer tự nó chọn đúng bao nhiêu từ; hai là đưa list đã lọc vào Qwen3-ASR thì lỗi từ hiếm giảm bao nhiêu.
> Kết quả chính: list 2000 từ còn khoảng 3 từ mỗi câu mà vẫn giữ 95% từ đúng, và B-WER của Qwen giảm từ 7,40 xuống 4,78, gần mức oracle 4,01."

---

## Slide 3 — Bài toán và dataset (1 phút)
> "Em nhắc lại bài toán một chút. ASR hay sai ở từ hiếm như tên riêng: người nói 'neverbend', Qwen chép 'never been'. Cách chữa là đưa trước một bias list các từ có thể xuất hiện.
> Ngoài đời list thường dài, vì mình không biết trước người dùng sẽ nói gì — ví dụ gọi điện thì phải đưa cả danh bạ. Mỗi câu chỉ có 1–2 từ đúng, còn lại là từ nhiễu.
> Benchmark IS21 mô phỏng đúng chuyện đó: mỗi câu LibriSpeech có list gồm từ hiếm thật của câu cộng N từ hiếm ngẫu nhiên, N từ 100 đến 2000.
> Bảng bên phải: chỉ đưa từ đúng thì B-WER 4,01; nhưng list 2000 từ thì 7,40 — bằng không đưa gì. Nên cần scorer để chọn lọc."

---

## Slide 4 — Kiến trúc (1,5 phút) — chỉ vào hình từ trái sang phải
> "Hàng trên là audio. Em cho qua encoder Whisper-large-v3, giữ đóng băng, ra 50 frame mỗi giây, rồi qua 1 lớp Linear và 1 lớp Transformer nhỏ để chuyển sang không gian của scorer.
> Hàng dưới là một từ bias, ví dụ 'neverbend'. Em không đọc từ đó thành âm thanh bằng TTS như Hou25, mà đổi sang chuỗi âm vị bằng g2p — ô đen là âm vị thật: N EH V ER B EH N D. Âm vị qua embedding và 2 lớp Transformer.
> Ở giữa là 2 lớp cross-attention: mỗi âm vị của từ 'đi tìm' đoạn audio giống nó. Nếu từ thật sự được nói, các âm vị sẽ tìm thấy một đoạn khớp liên tiếp.
> Sau đó lấy trung bình và max, qua MLP, ra một xác suất. Với câu ví dụ, scorer cho 'neverbend' là 0,70.
> Train: từ đúng là từ hiếm của câu; từ sai lấy 40% là từ gần chính tả, 60% ngẫu nhiên — giống cách Hou25 làm. Loss BCE, 5 epoch, khoảng 5 triệu tham số; encoder và g2p không train."

**Nếu thầy hỏi:**
- *Vì sao âm vị thay TTS?* → "TTS phải gọi dịch vụ đọc cho từng từ, 209 nghìn từ mất hơn 1 tiếng, và từ mới (ví dụ LLM sinh ra) phải đọc lại. Âm vị đổi 209 nghìn từ trong 14 giây."
- *Khác Hou25 chỗ nào?* → "Hou25 dùng TTS + ma trận tương đồng + CNN, lấy cứng top-50. Em dùng âm vị + cross-attention, ra xác suất để lọc theo ngưỡng."
- *g2p đoán sai từ lạ thì sao?* → "Có thể. Từ không có trong từ điển CMUdict thì g2p đoán. Em chưa đo ảnh hưởng."

---

## Slide 5 — Từng khối làm gì (1 phút) — đọc lướt bảng, nhấn dòng cross-attention
> "Encoder Whisper biến audio thành 50 vector mỗi giây, mỗi vector mô tả 20 mili-giây âm thanh; em giữ nguyên, không train.
> g2p biến chữ thành âm vị, cũng không train.
> Phần được train là: 1 lớp nhỏ phía audio, encoder âm vị 2 lớp, 2 lớp cross-attention và MLP cuối — tổng khoảng 5 triệu tham số.
> Quan trọng nhất là cross-attention: mỗi âm vị của từ 'hỏi' toàn bộ audio xem đoạn nào giống mình. Nếu từ thật sự được nói thì các âm vị đều tìm thấy chỗ khớp, điểm cao."

---

## Slide 6 — Train thế nào (1 phút) — chỉ theo hình từ trái sang phải
> "Lấy một câu train thật: 'as avonlea housekeepers were wont to tell in awed voices…'. Câu này có 3 từ hiếm: avonlea, awed, housekeepers — đó là 3 từ đúng, nhãn 1.
> Từ sai nhãn 0, lấy gấp 4 lần: 40% là từ gần chính tả như avonley, owed, housekeeping — để model học phân biệt từ nghe gần giống; 60% là từ ngẫu nhiên như brigand, bather.
> Cả 15 từ cùng audio đưa vào scorer, ra 15 xác suất. Loss BCE kéo xác suất lên 1 với từ đúng, xuống 0 với từ sai, rồi cập nhật trọng số.
> Mỗi bước 8 câu, 1 epoch là cả 28.539 câu, khoảng 1 giờ; train 5 epoch trên 1 card 4090, sau mỗi epoch đo trên dev-clean và giữ bản tốt nhất."

**Nếu thầy hỏi:**
- *Vì sao 1:4, vì sao 40/60?* → "40/60 theo Hou25. Tỉ lệ 1:4 paper không nói, em chọn theo thông lệ."
- *Overfit không?* → "Dev-clean gần như bão hoà từ epoch đầu (Recall@10 ~99,7%), vì list dev chỉ có từ nhiễu ngẫu nhiên — dev đang quá dễ, em cần dev khó hơn."

---

## Slide 7 — Dùng thế nào khi chạy thật (45 giây)
> "Lúc chạy thì không train gì nữa. Mỗi câu 5 bước: encode audio một lần; tra âm vị của 2000 từ trong bảng đã tính sẵn; scorer chấm 2000 từ theo khúc 32 từ, mất khoảng nửa giây; giữ các từ có xác suất từ 0,5 trở lên, trung bình còn 3 từ, có thể còn 0; rồi đưa vào prompt của Qwen, xếp tăng dần để từ điểm cao nhất đứng cuối như Hou25 khuyên.
> Phía dưới là lệnh thật trong code: với câu ví dụ, file context chỉ còn đúng 'neverbend'."

---

## Slide 8 — Scorer chọn từ có đúng không (1 phút)
> "Đây là đo riêng scorer, chưa dính ASR. Precision là từ giữ lại có đúng không; Recall là từ đúng có bị lọc mất không. Ngưỡng 0,5.
> List 100 từ: mỗi câu còn 2,1 từ, đúng 98%, giữ được 95% từ đúng.
> List 2000 từ: còn 3 từ, vẫn giữ 95% từ đúng, nhưng độ chính xác tụt còn 67% — list dài thì lẫn nhiều từ nhiễu hơn.
> List không có từ đúng nào: trung bình giữ 1 từ, 46% câu trả về rỗng — tức scorer biết 'không chọn gì'.
> Chấm 2000 từ mất khoảng nửa giây mỗi câu."

**Nếu thầy hỏi:**
- *Câu rỗng 25% ở N100 là tốt hay xấu?* → "24% câu test-clean không có từ hiếm nào, nên rỗng 25% là gần đúng."
- *Test-other?* → "Cùng xu hướng nhưng thấp hơn: ở 2000 từ precision 48%, recall 89%."

---

## Slide 9 — Đưa vào Qwen3-ASR (1,5 phút) — chỉ vào ví dụ rồi xuống biểu đồ
> "Một câu thật trong test-clean: 'missus neverbend…'. List IS21 có gần 2000 từ, chỉ 'neverbend' là đúng. Scorer chấm cả list: neverbend 0,70, các từ khác dưới 0,15. Lọc ngưỡng 0,5 giữ đúng 1 từ.
> Không có list, Qwen chép 'never been' — sai. Có list đã lọc, Qwen chép đúng 'neverbend'.
> Trên cả 2.620 câu: đưa nguyên 2000 từ thì B-WER 7,40 — không giúp gì. Lọc lấy top-50 như Hou25: 5,71. Lọc theo ngưỡng: 4,78, giảm 35%, gần oracle 4,01.
> Test-other: 16,45 → 11,57 → 9,71; oracle 7,73."

**Nếu thầy hỏi:**
- *Vì sao ngưỡng tốt hơn top-50?* → "Top-50 luôn nhét 50 từ dù câu chỉ có 1–2 từ đúng, 48 từ còn lại là nhiễu. Ngưỡng chỉ giữ ~3 từ."
- *Chọn ngưỡng 0,5 thế nào?* → "Chọn trước, chưa tinh chỉnh. Em thử 0,7: 4,96, kém hơn chút."
- *Chấm điểm thế nào?* → "Scorer IS21 chính thức, trên toàn bộ tập test, chuẩn hoá Whisper giống baseline."
- *List sai có làm hại không?* → "List toàn từ sai, sau khi lọc: 7,39 — bằng không đưa gì (7,40)."

---

## Slide 10 — Hạn chế và câu hỏi (1 phút)
> "Hạn chế: list dài thì precision tụt; và vẫn thua PTC-Bias (4,10 / 7,76) — phương pháp không cần train, dò âm vị bằng CTC.
> Em có thử cho scorer xem thêm bản chép lần 1 của Qwen, để ưu tiên những từ Qwen đang chép sai. Nhưng model học tắt: nó dựa vào bản chép mà bỏ qua audio, kết quả kém hơn, 6,58. Em đang sửa, ví dụ che ngẫu nhiên bản chép khi train.
> Em muốn hỏi thầy 3 điều…" (đọc 3 câu trên slide)

**Nếu thầy hỏi:**
- *Học tắt là sao?* → "Lúc train, 93% từ đúng đã có trong bản chép lần 1, nên model học quy tắc 'có trong bản chép thì là có mặt' thay vì nghe audio. Gặp từ không có trong bản chép, nó không nhận ra."
- *Không có list thì sao?* → "Em thử chấm cả kho 209 nghìn từ: giữ được 88% từ đúng nhưng mỗi câu ~12 từ, đúng chỉ 16% → đưa vào Qwen làm hại (8,12). Cần thêm nguồn ngữ cảnh, ví dụ LLM."

---

## Số liệu dự phòng (không nằm trên slide)
| Qwen B-WER clean / other | |
|---|---|
| không context | 7,40 / 16,08 |
| N100 nguyên list | 5,30 / 11,61 |
| model rẻ pmaxsim, p ≥ 0,7 | 5,46 / 11,39 |
| chỉ giữ từ Qwen chưa chép ra | 5,01 / 10,39 |
| kho 209k, p ≥ 0,7 | 8,12 / 16,49 |
| list toàn từ sai, sau lọc | 7,39 / 16,18 |

Chi tiết: `scorer/results.md`.
