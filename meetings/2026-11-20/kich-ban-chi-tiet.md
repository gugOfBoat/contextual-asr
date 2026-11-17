# Kịch bản nói cho slide `bao-cao-chi-tiet.pdf` (29 trang)

> Số liệu lấy từ `scorer/analysis/REPORT-1115.md`. Phần giải thích từng thí nghiệm nằm ở `scorer/analysis/TIEN-DO-1115.md`.
> Thời lượng: bản đầy đủ khoảng **25 phút** (trang 1–26). Nếu bị giục thì đi **đường tắt 12 phút**, xem cuối file. Trang 27–29 là dự phòng, chỉ mở khi thầy hỏi.

## Chuẩn bị trước khi vào họp (5 phút)

- [ ] Mở `bao-cao-chi-tiet.pdf` ở chế độ trình chiếu. Mở sẵn `kich-ban-chi-tiet.md` này ở màn hình thứ hai hoặc trên điện thoại.
- [ ] Mở sẵn `scorer/analysis/REPORT-1115.md` để tra số khi thầy hỏi sâu.
- [ ] Thuộc 3 con số:
  - Sạch@1: **72.6 → 89.3**.
  - B-WER test-other: **9.69 → 7.53**, không prompt.
  - Whisper cần vị trí: **−0.42 WER**.
- [ ] Thuộc 2 câu phải nói thật:
  - **WER tổng chưa thắng có ý nghĩa** (−0.25 [−0.47; +0.03]).
  - **Hou25 là bản nhóm tự tái hiện**, chưa phải checkpoint của anh Hoàng.
- [ ] Mở sẵn một file trống để ghi ý kiến thầy (sau buổi chép vào `meetings/2026-11-20.md`).

## Ba quy tắc khi nói

1. Cứ đến slide phát hiện là đọc theo khung: **ô vàng Câu hỏi → Nghĩ → Thử → Thấy → ô xanh Nghĩa là**. Đọc ô vàng trước, ô xanh lá sau cùng.
2. Đã nói "thắng Hou25" thì phải kèm chữ **B-WER** (lỗi trên từ hiếm).
3. Ý vệt chéo là của **anh Hoàng**. Nhắc tên anh ở trang 8 và trang 11.

---

## Mở đầu (trang 1–3, ~2 phút)

**Trang 1 – Bìa.** "Em báo cáo 3 ngày 13–15/11. Câu hỏi chính: scorer biết từ nằm ở giây nào thì có lợi gì?"

**Trang 2 – Tóm tắt.** Đây là trang quan trọng nhất. Đọc cột trái là câu hỏi, cột phải là câu trả lời; chỉ cần nhấn 3 dòng:
- Đo kỹ hơn thì Hou25 chỉ đạt **72.6**, scorer âm vị đạt 87.5.
- Scorer đề xuất đạt **89.3**, lọc 2000 từ còn khoảng 2 từ mỗi câu, precision 91%.
- Đưa vào Whisper, không cần prompt: **B-WER 9.69 → 7.53**, có ý nghĩa thống kê. WER tổng **mới ngang**.
- Dòng cuối: Hou25 ở đây là bản nhóm tái hiện, train 2 epoch.

**Trang 3 – Mạch trình bày.** "Em đi 6 phần: đặt câu hỏi, rồi **vì sao lại là vị trí**, rồi scorer, Whisper, tổng hợp, đi tiếp. Mỗi phát hiện có cùng một khung: câu hỏi, nghĩ gì, thử gì, thấy gì, nghĩa là gì. Màu xanh là tốt, màu đỏ là xấu. Khoảng trong ngoặc vuông là khoảng tin cậy 95%; khoảng nào không chứa 0 là có ý nghĩa."

## Phần 0: Đặt câu hỏi (trang 4–7, ~4 phút)

Thầy đã biết bài toán và Hou25 từ các buổi trước, nên phần này đi nhanh.

**Trang 4 – Từ điển tên gọi.** Không đọc hết bảng. "Đây là tên các hệ em sẽ nhắc. Hệ đề xuất là dòng tô vàng **pdual2_dl**: một model, hai nhánh, có dạy vị trí. Cách đưa vào ASR được đề xuất là **nhắc đúng lúc**."

**Trang 5 – 3 quyết định, luật viết trước.**
- D1 chọn nền cho scorer, D2 hỏi Whisper có cần vị trí không, D3 chọn cách nhắc.
- Nhấn mạnh: "Em viết luật trước khi chạy, để không thấy kết quả rồi mới chọn luật. Ví dụ ở D1: Hou25 sửa xong phải đạt ≥ 85.7 thì mới giữ."
- Bảng bên phải là lịch chạy.

**Trang 6 – Thước đo Sạch@1.**
- "Lọt top-50 giống chấm trắc nghiệm mà cho khoanh 50 đáp án: hệ nào cũng 95–98%."
- Sạch@1 hỏi từ đúng có đứng trên mọi từ nhiễu không. Từ đúng khác đứng trên thì **không bị phạt**.

**Trang 7 – Đo lại.** Lọt top-50 thì Hou25 và âm vị cách nhau 3 điểm, Sạch@1 thì cách nhau **15**. "Cứ 4 từ đúng thì Hou25 để hơn 1 từ bị nhiễu vượt mặt. Câu hỏi tiếp theo: vì sao?"

## Phần 1: Vì sao vị trí (trang 8–12, ~6 phút)

Phần này trình bày như một mục **Phân tích** trong paper: nói kết luận trước, rồi đưa bằng chứng.

**Trang 8 – Kết luận phần 1.** Đọc ô vàng:
> "Hou25 sai không phải vì thiếu tín hiệu, mà vì cách gộp. Tín hiệu đúng là một vệt có thứ tự, và nó có sẵn. Trung bình bỏ qua thứ tự, nên ở câu dài nhiễu được đẩy lên. Đọc theo vệt và train lại thì sửa được phần lớn, và có thêm vị trí."

Sau đó chỉ vào bảng 4 dòng: "4 kết luận nhỏ, mỗi cái một thí nghiệm, em đi qua từng cái."
- Nói rõ: "mảng rời rạc cộng dồn" là em diễn giải, chưa đo trực tiếp.
- Ghi công: vệt chéo là phát hiện của **anh Hoàng**.

**Trang 9 – 1a. Lỗi ở đâu** (kết luận nhỏ 1).
- Nhóm từng nghĩ từ đúng bị pha loãng. Thực tế là **từ nhiễu nổi lên**: 96% số lần từ đúng bị vượt là do một từ khác hẳn.
- Câu càng dài càng báo nhầm nhiều: 1.4 lên 12.4 từ mỗi câu (đường xám).
- Ghi chú dưới hình: tiện tay sửa được lỗi encoder chỉ nhìn 30 s đầu, bằng cửa sổ trượt.

**Trang 10 – 1b. Vì sao** (ví dụ 8×3 để thấy cơ chế).
- Chỉ vào hai lưới: từ đúng có **3 ô nối chéo**, từ nhiễu có **4 ô rải rác**.
- Đọc bảng từ trên xuống:
  - **Trung bình:** chỉ đếm tổng độ sáng, nên nhiễu thắng (0.83 so với 0.63).
  - **Ô cao nhất:** chỉ cần chỗ sáng, nên hoà.
  - **Vệt chéo:** đòi các ô nối đúng thứ tự, nên từ đúng thắng (5.0 so với 1.67).
- Ô xanh: điểm bắt đầu t0 = 3 **chính là vị trí của từ**.
- Nói rõ: số trong ví dụ là tự đặt, nhưng cách tính giống hệt code.

**Trang 11 – 1c. Sửa** (kết luận nhỏ 2–4).
- Bảng: 3 bản Hou25 train chung một lượt, chỉ khác bước gộp. Vệt chéo **+9.5** [+7.9; +11.2], báo nhầm ở câu dài 6.1 xuống 1.9, định vị 99%.
- Đối chứng "ô cao nhất" không tăng (73.8). Vậy thứ có ích là **thứ tự**, khớp với ví dụ ở trang 10.
- Ô đỏ: không train lại thì hỏng (A1: R#10 −3.6). Model phải được dạy cho cách đọc mới.
- Nếu thầy hỏi vì sao 73.3 ở đây mà 72.6 ở trang 7: 73.3 là bản train **cùng lượt** với bản vệt chéo, để so cho công bằng; 72.6 là bản dùng cho các bảng khác.

**Trang 12 – 1d. Hệ quả.** Nối sang phần sau:
- Hou25 + vệt chỉ đạt 82.9, chưa tới 85.7, nên chuyển sang **nền âm vị**. Hou25 + vệt giữ lại làm đối thủ mạnh.
- "Thứ tự" trở thành nhánh khớp đúng thứ tự trong scorer âm vị.
- "Vị trí" dùng để lọc list và nhắc Whisper đúng lúc.
- Đọc hình bên phải từ trên xuống: chuỗi 4 bước, cũng chính là phần còn lại của bài.
- Câu cuối: "Nếu viết paper thì phần này là mục Phân tích, đặt trước Method. Em không claim vệt chéo là ý mới, vì nó gần với dò từ khoá bằng audio mẫu; em cần tra thêm."

## Phần 2: Scorer, 4 phát hiện (trang 13–17, ~6 phút)

**Trang 13 – Phát hiện 1, chốt D1.**
- Hou25 sửa đạt 82.9, chưa tới mức 85.7, nên theo luật đặt trước thì **chọn nền âm vị**.
- Với từ nghe giống, âm vị vẫn hơn giọng máy 9.7 điểm.
- Âm vị nhanh hơn khoảng 200 lần và không cần TTS.
- Hou25 vệt chéo vẫn có ích: dùng làm đối thủ mạnh trong paper.

**Trang 14 – Phát hiện 2.**
- Hai kiểu khớp bù cho nhau: khớp tự do thì ít sót, khớp đúng thứ tự thì ít nhiễu và cho biên chuẩn.
- Cộng hai model cùng kiểu được +1.1; cộng hai kiểu khác nhau được +2.6. Vậy 1.5 điểm là nhờ khác kiểu.
- Mức lợi đó ngang với train thêm 9.6 lần dữ liệu, nên em dồn thành **một model hai nhánh**.

**Trang 15 – Phát hiện 3, scorer đề xuất.**
- Hình bên trái: hai nhánh có bản đồ riêng, đầu ra là (từ, p, đầu, cuối).
- Dạy vị trí giúp nhánh đúng thứ tự nhiều nhất (+2.5).
- Dòng vàng: **89.3**. Nhóm từ Whisper nghe sai tăng từ 45.9 (Hou25) lên 70.9.
- Nói thật: bản cộng hai model vẫn hơn 1.5 điểm.

**Trang 16 – Phát hiện 4, tranh nhau.**
- Luật: hai từ cùng chỗ thì chỉ giữ từ điểm cao hơn. Ví dụ pileser với pillar.
- Bảng: Hou25 top-50 có precision 3–4%; scorer đề xuất sau tranh nhau đạt **91%**, mất khoảng 1–2 điểm từ đúng.
- Câu chốt: bài "tìm 1 trong 2000" thành nhiều bài nhỏ "tìm 1 trong vài từ cùng chỗ".

**Trang 17 – Hai câu thật.**
- Câu A: các nhiễu trùng chỗ đều bị loại.
- Câu B là **chỗ yếu**: flauve nghe như "fall", emperour nghe như "empire". Không có từ hiếm nào tranh chỗ với chúng nên chúng lọt.
- Từ đây ra việc số 1: dạy scorer nói "chỗ này không có từ nào".

## Whisper (trang 18–21, ~5 phút)

**Trang 18 – Cách nhắc.** Ba bước:
1. Đọc alignment heads của Whisper để biết nó đang nghe giây nào.
2. Từ nào có đoạn chứa giây đó thì cộng λ = 5 vào token đầu của từ.
3. Chặn lặp: mỗi từ chỉ nhắc một lần. Câu nào bị lặp thì decode lại không nhắc, và áp luật này cho mọi hệ.

Ô bên phải: Whisper hầu như không viết từ vào sai chỗ (≤ 0.36%), nhưng **viết thêm** thì nhiều. Nên vị trí chủ yếu để biết **lúc nào phải im lặng**.

**Trang 19 – Bảng Whisper.** Đọc dòng vàng so với dòng Hou25:
- test-other: B-WER 9.69 → **7.53**, Δ −2.16 [−2.89; −1.54].
- test-clean: 5.15 → 3.93, Δ −1.22.
- Từ thường không đổi.
- **Nói rõ (ô đỏ):** WER tổng −0.25, khoảng tin cậy chứa 0, nên chưa có ý nghĩa.
- Thêm prompt thì WER thắng có ý nghĩa, nhưng chèn sai gấp đôi.
- Trần là 5.99.

**Trang 20 – Ba phát hiện phía ASR.**
1. **Whisper cần vị trí:** cùng list, nhắc đúng lúc tốt hơn nhắc mọi lúc 0.42 WER, có ý nghĩa. Nhắc mọi lúc làm hỏng cả từ thường.
2. **Kênh prompt đã đầy:** scorer hơn Hou25 17 điểm, nhưng đưa top-50 vào prompt thì vẫn không hơn.
3. **Còn 1.5 điểm tới trần:** một nửa do vị trí, một nửa do list.

Câu chốt: "Scorer chỉ có lời khi ra (từ, p, vị trí) và được dùng lúc decode."

**Trang 21 – Theo cỡ list.** Precision giảm từ 98% xuống 70%, nhưng B-WER chỉ tăng từ 7.14 lên 7.65: cách làm bền khi list lớn. Ô đỏ: chưa có Hou25 ở N100–1000.

## Tổng hợp (trang 22–23, ~2 phút)

**Trang 22 – Giả thuyết so với thực tế.** Không đọc hết bảng. Nhấn 3 dòng đỏ:
- Ý "từ đúng bị pha loãng" là sai.
- Train với nhiều nhiễu hơn không giúp.
- List toàn nhiễu vẫn làm hại.

"Em giữ cả kết quả âm, để biết cái gì không giúp mà khỏi thử lại."

**Trang 23 – Câu chuyện cho paper.** Đọc hình bên phải từ trên xuống, ba ô: Hou25 bỏ vị trí → scorer giữ vị trí → ASR dùng vị trí. Ô dưới cùng là 3 cách đặt đóng góp, cũng là **câu hỏi số 1 xin ý kiến thầy**.

## Đi tiếp (trang 24–26, ~3 phút + thảo luận)

**Trang 24 – Chỗ còn yếu.** Nói thật cả 6 dòng. Nhấn 2 dòng:
- Audio không có từ nào trong list thì WER vẫn tệ đi (4.49 so với 4.25).
- Bản một model còn kém bản cộng điểm 1.5.

**Trang 25 – Bước tiếp theo.** Việc 1–4 là phía scorer, xong trước 20/12. Việc 5–7 là bảng chính, Qwen3-ASR và khảo sát cách chỉnh logit (ý thầy đã gợi). Nói "việc 7 là theo ý thầy".

**Trang 26 – Mốc và 3 câu hỏi.** Đọc từng câu, rồi **dừng lại chờ thầy trả lời**, ghi lại nguyên văn.
1. Đóng góp chính đặt ở scorer, ở cách nhắc, hay gộp cả hai?
2. Ngoài Hou25, có cần so với PTC-Bias và cách nhắc không theo vị trí không?
3. Whisper + Qwen3-ASR đã đủ chưa?

---

## Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn | Mở trang |
|---|---|---|
| Sạch@1 khác Recall thế nào? | Chỉ đếm từ **nhiễu** đứng trên, nên câu nhiều từ hiếm không bị phạt oan | 6, 27 |
| Sao không so top-50 như paper? | Có báo top-50 (95.4 / 98.4), nhưng mình chỉ đưa ~2 từ cho ASR nên phải đo ở đầu bảng | 7 |
| 73.3 và 72.6 khác gì nhau? | 73.3 là Hou25 train cùng lượt với bản vệt chéo (so công bằng); 72.6 là bản dùng cho các bảng khác | 11 |
| Vì sao tác giả Hou25 lấy trung bình? | Paper không nêu lý do (theo ghi chép của nhóm). Em đoán: bản đồ có kích thước thay đổi, mục tiêu của họ chỉ là top-50 cho prompt nên không cần vị trí, và train chỉ cần nhãn có/không | 10 |
| Vệt chéo có phải ý mới không? | Không claim. Gần với dò từ khoá bằng audio mẫu (DTW); cái mới là chỉ ra cách gộp là nguồn lỗi của scorer, và dùng vị trí cho ASR | 12 |
| Vệt dốc 1 giả định người nói cùng tốc độ giọng máy? | Đúng là giả định vậy. CNN nhìn được vùng 9×9 nên bù được phần nào, nhưng chưa đo riêng | 10 |
| Từ nghe giống thì sao? | List nghe giống: âm vị 64.7 so với giọng máy 51.4; hai nhánh đạt 67.7 | 13, 15 |
| Vì sao không dùng luôn bản cộng điểm 90.8? | Tốn 2 model; việc 3 là cho một model đuổi kịp | 15, 25 |
| Lấy giây đang nghe kiểu gì? | Alignment heads của Whisper, như Simul-Whisper (2406.10052 §2.1) | 18 |
| Tham số chọn ở đâu? | dev-other 400 câu, không chọn trên test | 5, 18 |
| WER tổng có thắng không? | Chưa: −0.25 [−0.47; +0.03]. Thêm prompt thì thắng có ý nghĩa nhưng chèn sai gấp đôi | 19 |
| List nhỏ hơn thì sao? | N100: Sạch@1 98.2, precision 97.8, WER 2.96, B-WER 7.14 | 21 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.49 so với 4.25. Đây là việc số 1 | 24, 25 |
| Đã có ai làm chưa? | Vị trí theo frame có ở CTC/transducer; PTC-Bias dùng vị trí để sửa sau decode. Chưa thấy ai nhắc đúng lúc cho decoder đóng băng, nhưng mới tra khoảng 10 lượt | 28 |
| So với số Hou25 công bố? | Đây là bản tái hiện 2 epoch; chưa có checkpoint của anh Hoàng để so cùng pipeline | 24 |
| Dùng được cho Qwen3-ASR không? | Chưa thử; Qwen không có alignment heads giống Whisper, cần cách khác để biết giây đang nghe (việc 6) | 25 |
| File, code ở đâu? | Dự phòng 3 | 29 |

## Đường tắt 12 phút (khi bị giục)

Trang **2** (tóm tắt) → **7** (đo lại) → **8** (kết luận phần 1) → **10** (ví dụ vệt chéo) → **11** (train lại +9.5) → **15** (scorer đề xuất) → **16** (tranh nhau) → **19** (bảng Whisper) → **20** (3 phát hiện ASR) → **24** (chỗ yếu) → **26** (3 câu hỏi).

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
