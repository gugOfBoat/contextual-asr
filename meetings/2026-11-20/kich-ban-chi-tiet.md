# Kịch bản nói cho slide `bao-cao-chi-tiet.pdf` (28 trang)

> Số liệu lấy từ `scorer/analysis/REPORT-1115.md`. Phần giải thích từng thí nghiệm nằm ở `scorer/analysis/TIEN-DO-1115.md`.
> Thời lượng: bản đầy đủ khoảng **20 phút** (trang 1–25). Nếu bị giục thì đi **đường tắt 12 phút**, xem cuối file. Trang 26–28 là dự phòng, chỉ mở khi thầy hỏi.

## Chuẩn bị trước khi vào họp (5 phút)

- [ ] Mở `bao-cao-chi-tiet.pdf` ở chế độ trình chiếu. Mở sẵn `kich-ban-chi-tiet.md` này ở màn hình thứ hai hoặc trên điện thoại.
- [ ] Mở sẵn `scorer/analysis/REPORT-1115.md` để tra số khi thầy hỏi sâu.
- [ ] Thuộc 3 con số:
  - Recall@1 (lọc): **72.6 → 89.3**.
  - B-WER test-other: **9.69 → 7.53**, không prompt.
  - Whisper cần vị trí: **−0.42 WER**.
- [ ] Thuộc 2 câu phải nói thật:
  - **WER tổng chưa thắng có ý nghĩa** (−0.25 [−0.47; +0.03]).
  - **Hou25 là bản nhóm tự tái hiện**, chưa phải checkpoint của anh Hoàng.
- [ ] Mở sẵn một file trống để ghi ý kiến thầy (sau buổi chép vào `meetings/2026-11-20.md`).

## Ba quy tắc khi nói

1. Mỗi trang có **một ô xanh đậm = một điều rút ra**. Đọc ô đó trước, rồi chỉ vào số làm bằng chứng. Không đọc hết chữ.
2. Đã nói "thắng Hou25" thì phải kèm chữ **B-WER** (lỗi trên từ hiếm).
3. Ý vệt chéo là của **anh Hoàng**. Nhắc tên anh ở trang 10.

---

## Mở đầu (trang 1–3, ~2 phút)

**Trang 1 – Bìa.** "Em báo cáo 3 ngày 13–15/11. Câu hỏi chính: scorer biết từ nằm ở giây nào thì có lợi gì?"

**Trang 2 – Tóm tắt.** Trang quan trọng nhất.
- **Insight:** "Giữ lại từ nằm ở giây nào có lợi ở cả hai phía: scorer chọn đúng từ hơn, Whisper viết đúng từ hiếm hơn, không cần prompt."
- Đọc 3 số to: Recall@1 (lọc) **72.6 → 89.3**; lọc 2000 từ còn **~2 từ/câu**, precision 91%; B-WER **9.69 → 7.53**, có ý nghĩa.
- Dòng đỏ: WER tổng mới ngang Hou25; list toàn nhiễu vẫn làm WER tệ hơn một chút.
- Dòng nhỏ cuối: Hou25 ở đây là bản nhóm tái hiện, train 2 epoch.

**Trang 3 – Mạch trình bày.** "Em đi 6 phần: đặt câu hỏi, rồi **vì sao lại là vị trí**, rồi scorer, Whisper, tổng hợp, đi tiếp. Mỗi trang có một ô xanh đậm là điều rút ra. Màu xanh là tốt, màu đỏ là xấu. Khoảng trong ngoặc vuông là khoảng tin cậy 95%; khoảng nào không chứa 0 là có ý nghĩa."

## Phần 0: Đặt câu hỏi (trang 4–8, ~4 phút)

Thầy đã biết bài toán và Hou25 từ các buổi trước, nên phần này đi nhanh.

**Trang 4 – Từ điển tên gọi.** Không đọc hết bảng.
- **Insight:** "Chỉ cần nhớ 2 cặp: scorer Hou25 là đối thủ, scorer đề xuất là của nhóm; đưa vào Whisper bằng prompt top-50 như Hou25, hoặc nhắc đúng lúc như nhóm."
- Tên trong code (pmaxsim, pmono, pdual2_dl) để ở dòng nhỏ cuối trang, chỉ nói khi thầy hỏi.

**Trang 5 – Ba câu hỏi, luật viết trước.**
- **Insight:** "Mỗi câu hỏi có luật đặt trước: qua ngưỡng thì chọn A, không qua thì chọn B. Để không thấy số rồi mới chọn luật."
- Đọc D1 làm ví dụ: Hou25 sửa xong phải đạt ≥ 85.7 thì mới giữ. Cột phải cho biết câu trả lời nằm ở trang nào (10, 18, 21).

**Trang 6 – Thước đo Recall@1 (lọc).** Đọc ô xanh đậm trước, rồi mới giải thích.
- **Điều cần nhấn:** "Bọn em chỉ đưa 1–3 từ cho ASR. Vậy điều quan trọng là từ đúng có đứng trên mọi từ nhiễu không. Nên đo ở hạng 1."
- Cách tính: xếp 2000 từ theo điểm, bỏ các từ đúng khác của câu ra, rồi xem từ đúng có đứng hạng 1 không. Ví dụ `pileser` đứng sau `assyria`, nhưng `assyria` cũng là từ đúng nên bỏ ra; `pileser` vẫn trúng.
- Vì sao không đo top-50: Hou25 đo top-50 vì họ nhét 50 từ vào prompt. Ở top-50 hệ nào cũng 95–98%, không phân biệt được.
- Không phải nhóm tự chế: đây là cách "filtered" của Bordes et al. 2013 (TransE, mục 4.2), ai làm xếp hạng knowledge graph cũng biết.

**Trang 7 – Hai scorer đem ra so.** Trang này để thầy biết "scorer âm vị" là gì trước khi thấy nó trong mọi bảng.
- **Insight:** "Cả hai cùng hỏi: từ này có được nói trong câu không? Hou25 cho máy đọc từ thành audio rồi so hai đoạn audio. Scorer âm vị của nhóm so audio thẳng với cách phát âm của từ."
- Đọc hai dòng hình, từ trái sang phải, cùng ví dụ `pileser`. Dòng trên: máy đọc, so ra bảng giống nhau, CNN, **lấy trung bình**. Dòng dưới: phiên âm P-AY-L-IY-Z-ER, mỗi âm vị tìm chỗ giống nhất trong câu, cộng lại.
- Nhấn: âm vị không cần máy đọc nên nhanh hơn khoảng 200 lần, và ra luôn từ nằm ở đâu.
- Câu nối: "Giờ đo hai scorer này ở hạng 1."

**Trang 8 – Đo lại ở hạng 1.**
- **Điều cần nhấn:** "Top-50 che mất khác biệt. Hou25 kém 3 điểm ở top-50 nhưng kém **15 điểm** ở hạng 1. Cứ 4 từ đúng thì hơn 1 từ bị từ nhiễu vượt."
- Câu nối: "Vậy ai đã vượt từ đúng? Đó là phần 1."
- Nói thật (dòng đỏ): chưa đo được Recall@1 cao hơn thì B-WER chắc chắn thấp hơn.

## Phần 1: Vì sao vị trí (trang 9–11, ~4 phút)

Mỗi trang có **một ô xanh đậm = một insight**. Đọc ô đó trước, rồi chỉ vào số làm bằng chứng. Không đọc hết chữ.

**Trang 9 – Hou25 thua vì đâu.**
- **Insight:** "Hou25 thua vì **chấm cao từ nhiễu**, không phải vì chấm thấp từ đúng. Câu càng dài càng nặng."
- Bằng chứng, đọc 3 số to bên phải: 96% người vượt là từ khác hẳn; 76% không giống âm nào trong câu; từ nhiễu điểm cao tăng từ 1.4 lên 12.4 mỗi câu khi câu dài ra.
- Câu nối: "Vì sao câu dài lại sinh nhiều nhiễu?"

**Trang 10 – Vì sao: trung bình bỏ mất thứ tự.**
- Chỉ vào hai lưới trước: "Mỗi từ có một bảng. Hàng là thời gian của câu, cột là thời gian của từ. Từ được nói thì ô sáng nối thành **vệt chéo**; từ không được nói thì ô sáng **rải rác**."
- "Hou25 lấy trung bình cả bảng, tức là chỉ đếm tổng ô sáng. Nhiễu có 4 ô rải rác nên thắng từ đúng có 3 ô nối nhau. Câu càng dài càng nhiều ô sáng tình cờ."
- **Insight:** "Bắt các ô phải nối nhau đúng thứ tự (vệt chéo, ý anh Hoàng) thì **+9.5** Recall@1, nhiễu ở câu dài **giảm 3 lần**, và **biết luôn từ nằm ở đâu** (99%)."
- Dòng "chỗ sáng nhất" không tăng: vậy cái giúp là **thứ tự**.
- Phải train lại; chỉ đổi cách đọc lúc chấm thì tệ đi.
- Nếu thầy hỏi vì sao 6.1 ở đây mà 12.4 ở trang 9: 6.1 là bản train cùng lượt với bản vệt chéo (Recall@1 73.3); 12.4 là bản dùng cho các bảng khác (72.6).

**Trang 11 – Quyết định D1: sửa Hou25 rồi vẫn thua âm vị.**
- **Insight:** "Vệt chéo kéo Hou25 từ 72.6 lên 82.9, nhưng vẫn dưới scorer âm vị (85.7). Luật đặt trước: không vượt thì bỏ Hou25 làm nền, xây tiếp trên âm vị."
- Chỉ trên trục số, từ trái sang: Hou25 72.6 → vệt chéo 82.9 → vạch luật 85.7 (vẫn ở bên trái vạch).
- Nếu thầy hỏi vì sao 85.7 mà trang 8 ghi 87.5: 85.7 là bản âm vị lúc em viết luật, 87.5 là bản train lại sau đó. Dùng số nào thì Hou25 vẫn thua.
- Ô trái: âm vị còn hơn ở từ nghe giống (64.7 so với 51.4) và nhanh hơn khoảng 200 lần. Ô phải: hai thứ mang sang phần sau: **thứ tự** thành một nhánh của scorer âm vị; **vị trí** để lọc từ và nhắc Whisper đúng lúc.
- Hou25 vệt chéo giữ lại làm đối thủ mạnh. Không claim vệt chéo là ý mới.

## Phần 2: Scorer (trang 12–16, ~5 phút)

**Trang 12 – Hai cách khớp bù nhau.**
- **Insight:** "Khớp tự do thì ít sót, khớp đúng thứ tự thì ít nhiễu. Cộng lại được +2.6, trong khi gộp hai model cùng kiểu chỉ +1.1. Nên em làm một model hai nhánh."
- Bảng trái: tự do hơn ở Recall@1, đúng thứ tự hơn ở precision và định vị.

**Trang 13 – Scorer đề xuất.**
- **Insight:** "89.3 Recall@1, so với 72.6 của Hou25. Với những từ Whisper tự nghe sai, đúng chỗ cần scorer nhất: 45.9 lên 70.9."
- Hình: hai nhánh, đầu ra là (từ, p, đầu, cuối). Lúc train, từ đúng chỉ được tính ở đúng đoạn nó được nói.
- Nói thật (dòng đỏ): cộng điểm hai model riêng vẫn hơn 1.5, nhưng tốn gấp đôi.

**Trang 14 – Lỗi câu dài không còn.**
- **Insight:** "Chỉ Hou25 lấy trung bình mới báo nhầm tăng theo độ dài câu. Scorer đề xuất: 0.9 từ nhiễu mỗi câu dài, so với 12.4."
- Chỉ vào đường xám đi lên, các đường khác nằm ngang.
- Nói thật (dòng đỏ): khớp tự do không đòi thứ tự mà vẫn thấp, nhưng nó khác Hou25 hai chỗ cùng lúc, nên chưa biết chỗ nào giúp.

**Trang 15 – Biết vị trí thì lọc được list.**
- **Insight:** "Hai từ không thể cùng nằm một chỗ, nên trùng chỗ thì chỉ giữ từ điểm cao hơn. Precision từ 3–4% của Hou25 top-50 lên 91%, mỗi câu còn khoảng 2 từ."
- Ví dụ pileser 0.8 và pillar 0.4 cùng chỗ: giữ pileser. Không train thêm gì.

**Trang 16 – Chỗ yếu.**
- **Insight:** "Từ nhiễu nghe giống một từ thường trong câu thì không có từ hiếm nào tranh chỗ, nên nó lọt."
- Câu B: flauve nghe như "fall", emperour nghe như "empire".
- Từ đây ra việc số 1: cho scorer chọn "chỗ này không có từ nào".

## Whisper (trang 17–20, ~4 phút)

**Trang 17 – Cách nhắc đúng lúc.**
- **Insight:** "Whisper hầu như không viết từ vào sai chỗ, nhưng hay viết thêm khi không cần. Vị trí dùng để biết lúc nào phải im lặng."
- Ba bước: đọc alignment heads để biết Whisper đang nghe giây nào; từ nào nằm ở giây đó thì cộng λ = 5; mỗi từ chỉ nhắc đến khi đã viết ra.

**Trang 18 – Kết quả.**
- **Insight:** "Không cần prompt, B-WER test-other 9.69 xuống 7.53, test-clean 5.15 xuống 3.93."
- **Phải nói rõ:** WER tổng −0.25, khoảng tin cậy chứa 0, nên chưa có ý nghĩa. Nói "thắng Hou25" thì luôn kèm chữ B-WER.

**Trang 19 – Ba điều rút ra phía ASR.**
- **Insight:** "Scorer chỉ có lời khi đưa ra (từ, p, vị trí) và được dùng lúc decode."
- Ba ô: cần vị trí (nhắc mọi lúc WER 3.56, đúng lúc 3.14); prompt đã đầy (scorer hơn 17 điểm mà đưa vào prompt vẫn không hơn); còn 1.5 tới trần, một nửa do vị trí.

**Trang 20 – Theo cỡ list.**
- **Insight:** "List dài hơn 20 lần, precision 98 xuống 70%, nhưng B-WER chỉ 7.14 lên 7.65."
- Nói thật: chưa có Hou25 ở N100–1000.
## Tổng hợp (trang 21–22, ~2 phút)

**Trang 21 – Giả thuyết so với thực tế.** Không đọc hết bảng.
- **Insight:** "Giả thuyết sai có ích nhất: Hou25 sai vì từ nhiễu nổi lên, không phải vì từ đúng bị loãng. Nhờ đó chuyển sang giữ thứ tự và vị trí."
- Chỉ thêm dòng cuối: list toàn nhiễu vẫn làm hại, chưa đạt.
- "Em giữ cả kết quả âm, để biết cái gì không giúp mà khỏi thử lại." Hai dòng bỏ cho gọn ghi ở chân trang.

**Trang 22 – Ba quyết định đã ra.**
- **Insight, câu chuyện một dòng:** "Hou25 bỏ vị trí nên thua; scorer giữ vị trí lên 89.3 và lọc list còn khoảng 2 từ; Whisper dùng vị trí thì không cần prompt."
- Bảng: mỗi quyết định có luật đặt trước và kết quả. D3 nói rõ là **chưa tối ưu**.
- Câu nối: "Đóng góp đặt ở đâu là câu hỏi số 1 em xin ý kiến thầy ở trang 25."

## Đi tiếp (trang 23–25, ~3 phút + thảo luận)

**Trang 23 – Chỗ còn yếu.** Nói thật cả 6 dòng.
- **Insight:** "Chỗ yếu lớn nhất: audio không có từ nào trong list thì vẫn hại WER (4.49 so với 4.25). Reviewer sẽ hỏi ngay: có viết bừa không?"
- Nhấn thêm: bản một model còn kém bản cộng điểm 1.5.

**Trang 24 – Bước tiếp theo.**
- **Insight:** "Việc 1 sửa đúng chỗ yếu lớn nhất: dạy scorer nói chỗ này không có từ nào. Việc 1–4 xong trước 20/12."
- Việc 5–7 là bảng chính, Qwen3-ASR và khảo sát cách chỉnh logit. Nói "việc 7 là theo ý thầy".

**Trang 25 – Mốc và 3 câu hỏi.** Đọc từng câu, rồi **dừng lại chờ thầy trả lời**, ghi lại nguyên văn.
1. Đóng góp chính đặt ở scorer, ở cách nhắc, hay gộp cả hai?
2. Ngoài Hou25, có cần so với PTC-Bias và cách nhắc không theo vị trí không?
3. Whisper + Qwen3-ASR đã đủ chưa?

---

## Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn | Mở trang |
|---|---|---|
| Recall@1 (lọc) khác Recall thế nào? | Chỉ đếm từ **nhiễu** đứng trên, nên câu nhiều từ hiếm không bị phạt oan | 6, 26 |
| Sao không so top-50 như paper? | Có báo top-50 (95.4 / 98.4), nhưng mình chỉ đưa ~2 từ cho ASR nên phải đo ở đầu bảng | 8 |
| 73.3 và 72.6 khác gì nhau? | 73.3 là Hou25 train cùng lượt với bản vệt chéo (so công bằng); 72.6 là bản dùng cho các bảng khác | 10 |
| Vì sao tác giả Hou25 lấy trung bình? | Paper không nêu lý do (theo ghi chép của nhóm). Em đoán: bản đồ có kích thước thay đổi, mục tiêu của họ chỉ là top-50 cho prompt nên không cần vị trí, và train chỉ cần nhãn có/không | 10 |
| Vệt chéo có phải ý mới không? | Không claim. Gần với dò từ khoá bằng audio mẫu (DTW); cái mới là chỉ ra cách gộp là nguồn lỗi của scorer, và dùng vị trí cho ASR | 11 |
| Vệt dốc 1 giả định người nói cùng tốc độ giọng máy? | Đúng là giả định vậy. CNN nhìn được vùng 9×9 nên bù được phần nào, nhưng chưa đo riêng | 10 |
| Từ nghe giống thì sao? | List nghe giống: âm vị 64.7 so với giọng máy 51.4; hai nhánh đạt 67.7 | 11, 13 |
| Vì sao không dùng luôn bản cộng điểm 90.8? | Tốn 2 model; việc 3 là cho một model đuổi kịp | 13, 24 |
| Lấy giây đang nghe kiểu gì? | Alignment heads của Whisper, như Simul-Whisper (2406.10052 §2.1) | 17 |
| Tham số chọn ở đâu? | dev-other 400 câu, không chọn trên test | 5, 17 |
| WER tổng có thắng không? | Chưa: −0.25 [−0.47; +0.03]. Thêm prompt thì thắng có ý nghĩa nhưng chèn sai gấp đôi | 18 |
| List nhỏ hơn thì sao? | N100: Recall@1 (lọc) 98.2, precision 97.8, WER 2.96, B-WER 7.14 | 20 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.49 so với 4.25. Đây là việc số 1 | 23, 24 |
| Đã có ai làm chưa? | Vị trí theo frame có ở CTC/transducer; PTC-Bias dùng vị trí để sửa sau decode. Chưa thấy ai nhắc đúng lúc cho decoder đóng băng, nhưng mới tra khoảng 10 lượt | 27 |
| So với số Hou25 công bố? | Đây là bản tái hiện 2 epoch; chưa có checkpoint của anh Hoàng để so cùng pipeline | 23 |
| Dùng được cho Qwen3-ASR không? | Chưa thử; Qwen không có alignment heads giống Whisper, cần cách khác để biết giây đang nghe (việc 6) | 24 |
| File, code ở đâu? | Dự phòng 3 | 28 |

## Đường tắt 12 phút (khi bị giục)

Trang **2** (tóm tắt) → **7** (hai scorer) → **8** (đo ở hạng 1) → **9** (Hou25 thua vì đâu) → **10** (vệt chéo +9.5) → **11** (luật D1) → **13** (scorer đề xuất) → **15** (lọc list) → **18** (kết quả Whisper) → **19** (3 điều phía ASR) → **23** (chỗ yếu) → **25** (3 câu hỏi).

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
