# Kịch bản — gặp thầy 06/11/2026: trình bày scorer (~13 phút + thảo luận)

> **Lưu ý:** số trang trong file này theo thứ tự cũ (trước khi chuyển 6 trang xuống phụ lục). Bản ngắn `kich-ban.md` có số trang đúng.

Slide: `hop-thay.pdf` (21 trang chính + 2 phụ lục).
Ký hiệu: 🔴 = câu bắt buộc nói rõ, chậm · ⏱ = thời gian · 💬 = thầy có thể hỏi + cách trả lời.

**Mạch cả buổi (thuộc lòng):**
> Scorer giữ khung Hou25, chỉ đổi một chỗ: biểu diễn từ bằng **âm vị** thay vì audio TTS. Mỗi lựa chọn thiết kế đều có lý do và có số đo. Kết quả là chính xác **gần bằng** Hou25, nhưng không cần TTS, nhanh gấp 3, có xác suất để lọc theo ngưỡng, và biết từ nằm ở đâu. Việc còn lại để chốt là so với cách âm vị–âm vị (S0).

---

## Phần 1 — Mở đầu

### Trang 1 — Tiêu đề (⏱ 10 s)
> "Hôm nay em trình bày scorer: nó làm gì, vì sao em thiết kế như vậy, kết quả tới đâu, và còn thiếu gì để chốt."

### Trang 2 — Tóm tắt (⏱ 1 phút)
> "Ô 1: scorer nhận audio cùng list 2000 từ, trả cho mỗi từ xác suất nó có được nói hay không. 🔴 **Em giữ khung Hou25, chỉ đổi đúng một chỗ: biểu diễn từ bằng âm vị thay vì audio TTS.**
> Ô 2: có 3 lựa chọn thiết kế, em sẽ đi qua từng cái, mỗi cái có lý do và số đo.
> Ô 3: kết quả nói thẳng: gần bằng Hou25 chứ không thắng. Bù lại không cần TTS, nhanh gấp 3, lọc theo ngưỡng được, có vị trí. Một câu hỏi chưa trả lời được là âm vị–âm vị, em để cuối."

### Trang 3 — Bài toán và hệ thống (⏱ 45 s)
> "List càng dài thì SpeechLLM càng bỏ sót từ đúng và chèn từ sai. Với Qwen, đưa list 2000 từ thì B-WER 7.40, y như không đưa list.
> Scorer đứng giữa: nó chấm list theo audio, chọn ra vài từ rồi đưa vào prompt. Mũi tên vàng là bước sau mà thầy gợi ý buổi trước: dùng thêm vị trí lúc decode."

---

## Phần 2 — Thiết kế: vì sao (phần chính, ~6 phút)

### Trang 4 — Hình kiến trúc (⏱ 1 phút) ⭐
Chỉ vào hình, đi từ trên xuống:
> "Hàng trên là audio: qua encoder Whisper-large-v3 **đóng băng**, giống hệt Hou25, rồi một lớp chiếu nhỏ.
> Hàng dưới là từ: ví dụ 'cabinet' qua g2p thành 7 âm vị K AE B AH N AH T, rồi embedding.
> Ba cột là ba cách so: pmaxsim nhanh nhất, **pxattn ở giữa là bản em đưa vào ASR**, bên phải là Hou25 dùng audio TTS.
> Phần xanh là phần train, khoảng 5 triệu tham số, train trên train-clean-100 giống Hou25."

### Trang 5 — Bảng từng bộ phận (⏱ 1 phút)
> "Bảng này trả lời 'tại sao như thế' cho từng bộ phận, cột cuối là bằng chứng đã đo.
> 🔴 **Em giữ encoder và dữ liệu giống Hou25, nên chênh lệch đến từ thiết kế chứ không phải từ dữ liệu.**
> Ba dòng tô xanh là 3 lựa chọn chính, em đi qua từng cái. Dòng 'phía audio' em nói thẳng là chưa ablation."

### Trang 6 — Lựa chọn (a): âm vị thay TTS (⏱ 1 phút)
> "Hou25 phải trả giá cho TTS: đọc cả kho 209 nghìn từ mất khoảng 80 phút và 21 GB feature; gặp từ mới thì phải TTS thêm (Hoàng đo được 2,7% từ test không có TTS); và chỉ có một giọng nữ cho mọi người nói.
> Âm vị thì g2p ra ngay cho từ bất kỳ và không có giọng.
> Bảng bên phải: tìm từ đúng ngang hoặc hơn, số từ cần giữ để đủ 99% từ đúng ít hơn hẳn (107 so với 256), nhanh gấp 3. Giọng nam và nữ không chênh nhau.
> 🔴 **Cái mất: dễ bị từ nghe giống lừa hơn, 21,6 so với 17,3.** Em đoán là CNN của Hou25 nhìn được thứ tự âm chi tiết hơn, nhưng chưa kiểm."

### Trang 7 — Sao không dùng ASR âm vị để hai phía cùng là âm vị? (⏱ 2 phút) ⭐ CÂU THẦY HỎI
> "Câu này em nghĩ thầy sẽ hỏi: phía từ đã là âm vị, sao phía audio không đổi luôn thành âm vị cho hai bên cùng một loại? Đó chính là cách của PTC-Bias. Em so hai cách theo 4 dòng.
> **Dòng 1, audio thành gì.** PTC biến mỗi frame audio thành xác suất của 71 âm vị, tức chỉ giữ thông tin 'đây là âm gì'. Em giữ nguyên vector của Whisper, không ép về âm vị.
> **Dòng 2, train bằng gì.** PTC train một bộ nhận dạng âm vị, bằng nhãn âm vị. Nhãn này lấy từ g2p của transcript, nên bản thân nó cũng có thể sai. Em train thẳng bằng nhãn 'từ này có được nói không', đúng cái việc cuối cùng cần làm.
> **Dòng 3, ra điểm thế nào.** PTC khớp chuỗi âm vị của từ lên ma trận xác suất, rồi dùng 5 tham số chỉnh tay để ra danh sách. Em học thẳng, chỉ có một ngưỡng p.
> **Dòng 4, khi phát âm thật khác từ điển.** Đường khớp của PTC phải đi đúng chuỗi g2p, lệch là bị trừ điểm. Attention của em học được cách chịu lệch.
> Có 2 con số ủng hộ cách của em. Một: PTC phụ thuộc chất lượng bộ âm vị: chính họ đổi WavLM sang encoder của Qwen thì B-WER tệ đi, 4,10 lên 4,45. Hai: em đo thấy 11–16% từ hiếm có phát âm g2p khác phát âm MFA, nên 'chuỗi âm vị chuẩn' không phải lúc nào cũng đúng.
> 🔴 **Nhưng em nói thẳng: hiện PTC tìm từ tốt hơn em (99,3 so với 98,6). Họ dùng WavLM, dữ liệu khác, nên chưa so công bằng. Vì vậy em đề xuất S0: làm đúng cách của PTC trên cùng encoder Whisper, cùng dữ liệu, với tiêu chí viết trước. Thắng thì em đổi.**"

💬 *"11–16% đó có phải người nói phát âm khác không?"* → Chưa chắc. MFA dùng từ điển riêng của nó, nên lệch có thể do hai từ điển khác nhau chứ không hẳn do người nói. Con số chỉ cho thấy phát âm của từ hiếm không có một đáp án duy nhất.

💬 *"Thế giữ vector thì có lợi gì?"* → Đây là lập luận, chưa có số tách riêng. Vector giữ cả thông tin ngoài 'âm gì', ví dụ âm nối, ranh giới từ. Ép về 71 âm vị là bỏ những thứ đó. S0 sẽ cho biết phần này có đáng giá không.

💬 *"Dòng 'chữ của Qwen → âm vị' trên trang cũ là gì?"* → Là lấy chữ Qwen nhận ra rồi đổi sang âm vị để so. Cách này chốt cứng qua chữ nên kém (95,4), và khác PTC vì PTC không chốt chuỗi. Em đưa nó vào trang 16 (đã thử và bỏ).

### Trang 8 — Lựa chọn (b): attention là gì (⏱ 1,5 phút) ⭐
Chỉ vào hình bên trái:
> "Em giải thích attention bằng hình này. Audio được chia thành khoảng 350 ô, mỗi ô 20 mili-giây. Mỗi âm vị của từ, ví dụ K và T, có **100 điểm chú ý** để rải lên các ô. Ô nào nghe giống mình thì nó dồn nhiều điểm: K dồn 70 điểm vào một ô, T dồn 75 vào ô khác.
> Rải xong, mỗi âm vị mang về thông tin của chỗ nó dồn điểm. Nếu cả 7 âm vị đều tìm được chỗ khớp, theo đúng thứ tự, thì p cao.
> Vì sao không dùng CNN như Hou25? Hou25 so audio với audio, nên vệt chéo có sẵn cho CNN nhìn. Bên em, ban đầu ma trận âm vị × ô audio không có vệt nào, nên phải để từng âm vị tự đi tìm.
> 🔴 **Điểm then chốt: model chỉ được dạy 'từ có được nói không', không ai dạy vị trí. Nhưng muốn đoán đúng thì phải rải điểm đúng chỗ, nên bảng rải điểm tự trở thành bản đồ vị trí. Các trang sau em cho thầy xem bảng đó.**"

### Trang 9 — Cách đọc hình attention lớp 1 (⏱ 1,5 phút) ⭐
Đọc chậm, chỉ tay vào từng mũi tên đỏ:
> "Đây là bảng rải điểm của từ 'bastard'.
> - **Hàng** là 6 âm vị của từ theo g2p: B, AE, S, T, ER, D.
> - **Cột** là các ô audio, mỗi ô 20 mili-giây. Frame 280 là khoảng giây 5,6.
> - **Màu**: tối là gần 0 điểm, vàng là nhiều điểm.
> - **Chữ phía trên và vạch trắng** là âm vị thật trong audio theo MFA, một công cụ căn chỉnh chuẩn. Scorer chưa bao giờ thấy các vạch này; em vẽ lên chỉ để kiểm tra.
> Thầy nhìn 3 điều:
> Một, cả câu tối, chỉ sáng ở frame 283 đến 311, đúng chỗ 'bastard' được nói.
> Hai, có **bậc thang**: B, AE sáng ở bên trái, rồi S, rồi T, rồi ER, D ở bên phải, đúng thứ tự phát âm.
> Ba, em nói thẳng là chưa hoàn hảo: B sáng ở đầu đoạn AE, ER sáng cùng chỗ với T, tức lệch sang đoạn kề bên. Đo trên cả tập thì 2/3 âm vị rơi đúng đoạn của mình."

💬 *"Sao lại lệch?"* → Chưa kiểm nguyên nhân. Model không được dạy vị trí, nó chỉ cần nhìn gần đúng chỗ là đoán được có/không, nên lệch một âm vị không bị phạt. Muốn chính xác hơn có thể thêm loss căn chỉnh bằng MFA, nhưng hiện chưa cần.

### Trang 10 — Thêm 3 từ khác (⏱ 45 s)
> "Để thầy thấy đây không phải hình chọn đẹp: ba từ khác, hai từ ở tập khó test-other. Hình do script tự lấy từ mẫu ngẫu nhiên, em không chọn tay.
> 'refer': R, rồi AH, rồi F, ER. 'uncommon' dài 7 âm vị: AH, N ở trái, K, AA ở giữa, M, AH, N ở phải. 'milner's': M, IH, rồi L, rồi N, ER, Z.
> Cả ba đều giống nhau: phần còn lại của câu tối, chỉ sáng ở từ, và sáng **lệch dần từ trái sang phải** theo thứ tự âm."

### Trang 11 — Đối chứng 1: lớp 2 (⏱ 45 s)
> "pxattn có 2 lớp attention. Đây là lớp thứ hai, cùng hai từ. Điểm chú ý rải rộng, có chỗ sáng ngoài từ: ở 'bastard', hàng S sáng ở frame 263, trước khi từ bắt đầu. Không còn bậc thang.
> Đo trên cả tập thì lớp 2 chỉ đúng chỗ khoảng 50%, so với 99% của lớp 1.
> 🔴 **Nên vị trí phải đọc ở lớp 1.** Em đoán lớp 2 nhìn rộng ra xung quanh để kiểm tra ngữ cảnh, nhưng chưa kiểm."

### Trang 12 — Đối chứng 2: cosine thô (⏱ 1 phút) ⭐
> "Đây là đối chứng quan trọng nhất. Có người sẽ hỏi: hình nào vẽ ra cũng thấy vị trí thôi, đâu có gì đặc biệt?
> Em lấy chính vector âm vị và vector audio **của pxattn**, tính độ giống trực tiếp bằng cosine, không qua attention. Kết quả: sáng lung tung, kể cả ngoài từ. Đo trên cả tập thì trúng chỗ 8,8 / 12,3%, ngang với đặt ngẫu nhiên.
> 🔴 **Tức là cùng một model, cùng những vector đó, nhưng chỉ phép so đã được học (attention) mới mang vị trí. Bậc thang ở trang 9 không phải tự có; model học ra nó.**"

### Trang 13 — Ma trận 39 × 39 (⏱ 1 phút)
> "Trang này hỏi sâu hơn: không chỉ 'đúng chỗ' mà 'đúng âm'.
> Cách tạo: với mỗi đoạn audio là âm X theo MFA, em xem âm Y của từ dồn bao nhiêu điểm vào đó, rồi cộng trên mọi từ của hai tập.
> Hàng là âm thật trong audio, cột là âm đang đi tìm, màu đỏ là nhìn vào nhiều.
> Thấy **đường chéo đỏ**: âm AE đi tìm thì dồn vào đoạn audio là AE, K vào K. 69% hàng có ô chéo lớn nhất; trung bình ô chéo đứng thứ 1,5 trên 39. Lớp 2 chỉ 5%, cosine thô 0%.
> Giới hạn: trong một từ, cùng loại âm thường cũng là cùng chỗ, nên hai thứ này lẫn vào nhau một phần."

### Trang 14 — Tổng hợp bằng số (⏱ 45 s)
> "Gộp lại bằng số, khoảng 1.000 từ mỗi tập: lớp 1 đặt đúng chỗ 99%, sai số 0,03 giây; cosine thô và đặt ngẫu nhiên chỉ khoảng 10%, sai hơn 2 giây. Cả những từ scorer còn nghi ngờ (p < 0,5) cũng được đặt đúng chỗ 94–100%.
> 🔴 **Ba lý do tin được: nhãn train chỉ là có/không; mốc MFA model chưa từng thấy; và trong cùng model, cosine thô ngang ngẫu nhiên.**
> Giới hạn: 400 câu mỗi tập, 1 checkpoint."

### Trang 15 — Lựa chọn (c): sigmoid → ngưỡng (⏱ 1 phút)
> "Hou25 lấy top-50, nên câu nào cũng nhận 50 từ, kể cả câu không có từ hiếm nào. Em dùng sigmoid nên mỗi từ có xác suất riêng: số từ thay đổi theo câu, có thể bằng 0.
> Trên Qwen: lọc p ≥ 0,5 cho 4,78 / 9,71, tốt hơn top-50 rõ và gần oracle. 🔴 **List toàn từ nhiễu thì scorer đưa Qwen về đúng mức không context: list nhiễu không kéo sai được.**
> Giới hạn: Whisper bị lặp chữ khi prompt chỉ 0–1 từ, nên trên Whisper em vẫn dùng top-50."

### Trang 16 — Đã thử và bỏ (⏱ 45 s)
> "Em cũng đã thử các hướng khác và bỏ có lý do:
> - cho scorer nhìn thêm hyp của Qwen: kém hẳn, nghi model học tắt theo hyp;
> - loss theo cả list: tệ mọi metric;
> - chốt cứng qua chữ ASR: kém;
> - pmaxsim: rẻ nhưng không xét thứ tự âm ('stone' gần giống 'notes'), nên giữ làm tầng lọc rẻ.
> Train 960 giờ thì tốt hơn, nhưng bảng chính em dùng 100 giờ để so công bằng với Hou25."

---

## Phần 3 — Kết quả (~3 phút)

### Trang 17 — Kết quả chính (⏱ 1 phút)
> "Whisper: nguyên list N2000 cho 9,15 / 18,69; scorer còn khoảng 5 / 10, giảm có ý nghĩa thống kê (phụ lục A).
> 🔴 **Em nói thẳng: buổi trước em nói thắng Hou25 cả 8 ô, câu đó chỉ đúng khi so với số trong paper. Chạy cùng pipeline với bản tái hiện của Hoàng thì em kém 0,2 tới 0,9.** Hiện mới 1 seed, chưa có khoảng tin cậy, vì đang chờ file từng câu của Hoàng."

### Trang 18 — Chốt đóng góp ở mức nào (⏱ 1 phút)
> "Nên claim em xin chốt là: **chính xác gần bằng Hou25**, đổi lại không cần TTS, nhanh gấp 3, lọc theo ngưỡng được, và có vị trí tới mức âm vị. Hai thứ cuối, p và vị trí, đúng là thứ cần cho bước can thiệp lúc decode thầy gợi ý."

### Trang 19 — Scorer đã gần trần (⏱ 45 s)
> "Thí nghiệm này cho thấy scorer đã làm gần hết phần của nó. Lỗi 'nghe giống', tức Whisper viết từ gần đúng, của scorer em ngang với khi đưa prompt oracle (4,69 so với 4,62).
> 🔴 **Prompt hoàn hảo cũng không sửa được nhóm lỗi này, nên cải thiện scorer thêm cũng không giảm được nó; phần còn lại nằm ở ASR.**"

---

## Phần 4 — Còn thiếu + xin ý kiến

### Trang 20 — Còn thiếu gì để chốt (⏱ 1 phút)
> "Để chốt scorer em còn 5 việc. Quan trọng nhất là **S0**: train một head CTC âm vị trên cùng encoder, cùng dữ liệu, để trả lời câu âm vị–âm vị. Tiêu chí em viết trước: chỉ đổi nếu tìm từ không kém, không bị từ nghe giống lừa hơn, và vị trí vẫn chính xác. Không đạt thì giữ pxattn.
> S1 là so ghép cặp với Hoàng, đang chờ file. S2–S4 là dùng vị trí để lọc từ nghe giống, decode lại, và chạy 3 seed."

### Trang 21 — Xin thầy góp ý
Đọc từng câu, **ghi lại câu trả lời của thầy**:
1. Claim "gần bằng Hou25, không TTS, nhanh gấp 3, có p, có vị trí" đã đủ thuyết phục chưa?
2. Lý do chọn âm vị + attention còn thiếu bằng chứng nào?
3. S0 chạy trước khi chốt, giới hạn khoảng 1 tuần?
4. Xin card GPU cho S0–S4 (khoảng 30 giờ, ước lượng).

---

## Câu hỏi khó có thể gặp

💬 *"Vậy scorer mới ở đâu, chỉ thay TTS bằng âm vị?"* → Đúng, thay đổi chính là biểu diễn từ. Nhưng thay đổi đó kéo theo 3 thứ Hou25 không có: không cần TTS (từ mới chấm ngay), xác suất dùng được để lọc theo ngưỡng, và vị trí tới mức âm vị. Em không claim chính xác hơn.

💬 *"Sao kém Hou25 0,2–0,9?"* → Nghi do từ nghe giống (21,6 so với 17,3). Nhưng phân rã lỗi cho thấy từ nhiễu viết đè lên từ đúng chỉ tốn 0,14 / 0,34 điểm, nên chưa giải thích hết. Cần S1 (file của Hoàng) để biết chắc.

💬 *"Sao lớp audio thêm 1 lớp Transformer, sao 2 lớp attention, sao 256 chiều?"* → Chưa ablation, em nói thẳng. S0 có thêm arm pxattn 1 lớp, vì chỉ lớp 1 định vị, lớp 2 thì không.

💬 *"g2p sai với tên riêng thì sao?"* → g2p tra CMUdict, từ ngoài từ điển thì đoán. Em chưa đo riêng lỗi do g2p; có thể tách các từ ngoài CMUdict ra để đo.

💬 *"Vệt chéo của Hou25 khác bậc thang của em thế nào?"* → Vệt của Hou25 có sẵn vì so audio với audio, và nó là đầu vào cho CNN. Bậc thang của em ban đầu không có (cosine thô là nhiễu, trang 12); attention tự học ra, và nó không đưa vào bộ phân loại.

💬 *"Giới tính?"* → Đã kiểm: không thấy chênh cho cả 3 scorer (CI chứa 0). Giới tính ước từ F0 nên chỉ là xấp xỉ.

## Sau buổi
- Ghi biên bản vào `meetings/2026-11-06.md` (câu trả lời cho 4 câu ở trang 21 + yêu cầu mới).
- Nhắc Hoàng gửi `exp/` để chạy S1.
