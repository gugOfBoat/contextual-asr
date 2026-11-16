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
3. Ý vệt chéo là của **anh Hoàng**. Nhắc tên anh ở trang 5 và trang 12.

---

## Mở đầu (trang 1–3, ~2 phút)

**Trang 1 – Bìa.** "Em báo cáo 3 ngày 13–15/11. Câu hỏi chính: scorer biết từ nằm ở giây nào thì có lợi gì?"

**Trang 2 – Tóm tắt.** Đây là trang quan trọng nhất. Đọc cột trái là câu hỏi, cột phải là câu trả lời; chỉ cần nhấn 3 dòng:
- Đo kỹ hơn thì Hou25 chỉ đạt **72.6**, scorer âm vị đạt 87.5.
- Scorer đề xuất đạt **89.3**, lọc 2000 từ còn khoảng 2 từ mỗi câu, precision 91%.
- Đưa vào Whisper, không cần prompt: **B-WER 9.69 → 7.53**, có ý nghĩa thống kê. WER tổng **mới ngang**.
- Dòng cuối: Hou25 ở đây là bản nhóm tái hiện, train 2 epoch.

**Trang 3 – Mạch trình bày.** "Em đi 6 phần. Mỗi phát hiện có cùng một khung: câu hỏi, nghĩ gì, thử gì, thấy gì, nghĩa là gì. Màu xanh là tốt, màu đỏ là xấu. Khoảng trong ngoặc vuông là khoảng tin cậy 95%; khoảng nào không chứa 0 là có ý nghĩa."

## Bài toán (trang 4–7, ~4 phút)

**Trang 4 – Thư ký, danh bạ, trợ lý.**
- ASR là thư ký: chép tốt từ thường, nhưng hay sai tên riêng ("Pileser" thành "pillar sir").
- Bias list là danh bạ 2000 từ. Mỗi câu chỉ có 1–3 từ thật sự được nói.
- Scorer là trợ lý: nghe audio trước rồi chọn vài từ đáng tin.
- Không đưa cả danh bạ cho thư ký được, vì đưa 2000 từ thì thư ký sẽ "nghe ra" những từ không hề có.

**Trang 5 – Hou25 và chỗ làm mất vị trí.** Chỉ vào hai ô đỏ trong hình:
1. Lấy trung bình cả bản đồ, nên vị trí của từ bị vứt đi.
2. Đưa top-50 vào prompt: 50 từ mà không biết từ nào ở đâu, và khoảng 48 trong số đó là nhiễu.

"Anh Hoàng thấy bản đồ ở bước 3 có một **vệt chéo** đúng chỗ từ được nói. Vị trí có sẵn ở đó, nhưng bị bỏ đi."

**Trang 6 – Ý tưởng.** Một câu: "Trợ lý trả ra (từ, độ tin, vị trí) chứ không chỉ một điểm." Chỉ vào hình trục thời gian: pileser nằm ở giây 3.0–3.6, nên chỉ nhắc thư ký ở giây 3.2; ở giây 1 và giây 5 thì im lặng. Vị trí mang lại hai lợi: ở scorer thì loại được nhiễu, ở ASR thì nhắc đúng lúc.

**Trang 7 – Từ điển tên gọi.** Không cần đọc hết bảng. "Đây là tên các hệ em sẽ nhắc. Hệ đề xuất là dòng tô vàng **pdual2_dl**: một model, hai nhánh, có dạy vị trí. Cách đưa vào ASR được đề xuất là **nhắc đúng lúc**."

## Đặt câu hỏi (trang 8–9, ~2 phút)

**Trang 8 – 3 quyết định, luật viết trước.**
- D1 chọn nền cho scorer, D2 hỏi Whisper có cần vị trí không, D3 chọn cách nhắc.
- Nhấn mạnh: "Em viết luật trước khi chạy, để không thấy kết quả rồi mới chọn luật. Ví dụ ở D1: Hou25 sửa xong phải đạt ≥ 85.7 thì mới giữ."
- Bảng bên phải là lịch chạy: 2 ngày, phần lớn chạy trên GPU.

**Trang 9 – Thước đo Sạch@1.**
- Ví dụ trắc nghiệm: "cho khoanh 50 đáp án thì ai cũng đúng 95–98%".
- Sạch@1 hỏi từ đúng có đứng trên mọi từ nhiễu không. Từ đúng khác đứng trên thì **không bị tính phạt**.

## Scorer: 6 phát hiện (trang 10–17, ~9 phút)

**Trang 10 – Đo lại.** Hai thước cho hai khoảng cách khác nhau: lọt top-50 thì cách nhau 3 điểm, Sạch@1 thì cách nhau 15. "Cứ 4 từ đúng thì Hou25 để hơn 1 từ bị nhiễu vượt mặt."

**Trang 11 – Phát hiện 1.** Trang này ngược với điều nhóm tin trước đây.
- Nhóm từng nghĩ từ đúng bị pha loãng. Thực tế là **từ nhiễu nổi lên**: 96% số lần từ đúng bị vượt là do một từ khác hẳn, không phải từ nghe giống.
- Câu càng dài càng báo nhầm nhiều: từ 1.4 lên 12.4 từ mỗi câu (đường xám trong hình).
- Ghi chú dưới hình: tiện tay sửa được lỗi encoder chỉ nhìn 30 s đầu, bằng cửa sổ trượt.

**Trang 12 – Phát hiện 2, giả thuyết của anh Hoàng.**
- Giữ nguyên Hou25, chỉ đổi bước gộp thành "đọc vệt chéo", rồi train lại: **+9.5 Sạch@1**, báo nhầm ở câu dài giảm 3 lần, định vị trúng 99%.
- Dòng "ô cao nhất" là đối chứng và không tăng. Vậy thứ có ích là **hình dạng vệt**.
- Nếu thầy hỏi vì sao trang 2 ghi 73.3 còn trang 10 ghi 72.6: 73.3 là bản Hou25 train **trong cùng lượt** với bản vệt chéo, để so cho công bằng; 72.6 là bản dùng cho các bảng còn lại.

**Trang 13 – Phát hiện 3, chốt D1.**
- Hou25 sửa đạt 82.9, chưa tới mức 85.7, nên theo luật đặt trước thì **chọn nền âm vị**.
- Với từ nghe giống, âm vị vẫn hơn giọng máy 9.7 điểm.
- Âm vị nhanh hơn khoảng 200 lần và không cần TTS.
- Hou25 vệt chéo vẫn có ích: dùng làm đối thủ mạnh trong paper.

**Trang 14 – Phát hiện 4.**
- Hai kiểu khớp bù cho nhau: khớp tự do thì ít sót, khớp đúng thứ tự thì ít nhiễu và cho biên chuẩn.
- Cộng hai model cùng kiểu được +1.1; cộng hai kiểu khác nhau được +2.6. Vậy 1.5 điểm là nhờ khác kiểu.
- Mức lợi đó ngang với train thêm 9.6 lần dữ liệu, nên em dồn thành **một model hai nhánh**.

**Trang 15 – Phát hiện 5, scorer đề xuất.**
- Hình bên trái: hai nhánh có bản đồ riêng, đầu ra là (từ, p, đầu, cuối).
- Dạy vị trí giúp nhánh đúng thứ tự nhiều nhất (+2.5).
- Dòng vàng: **89.3**. Nhóm từ Whisper nghe sai tăng từ 45.9 (Hou25) lên 70.9.
- Nói thật: bản cộng hai model vẫn hơn 1.5 điểm.

**Trang 16 – Phát hiện 6, tranh nhau.**
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
| Sạch@1 khác Recall thế nào? | Chỉ đếm từ **nhiễu** đứng trên, nên câu nhiều từ hiếm không bị phạt oan | 9, 27 |
| Sao không so top-50 như paper? | Có báo top-50 (95.4 / 98.4), nhưng mình chỉ đưa ~2 từ cho ASR nên phải đo ở đầu bảng | 10 |
| 73.3 và 72.6 khác gì nhau? | 73.3 là Hou25 train cùng lượt với bản vệt chéo (so công bằng); 72.6 là bản dùng cho các bảng khác | 12 |
| Từ nghe giống thì sao? | List nghe giống: âm vị 64.7 so với giọng máy 51.4; hai nhánh đạt 67.7 | 13, 15 |
| Vì sao không dùng luôn bản cộng điểm 90.8? | Tốn 2 model; việc 3 là cho một model đuổi kịp | 15, 25 |
| Lấy giây đang nghe kiểu gì? | Alignment heads của Whisper, như Simul-Whisper (2406.10052 §2.1) | 18 |
| Tham số chọn ở đâu? | dev-other 400 câu, không chọn trên test | 8, 18 |
| WER tổng có thắng không? | Chưa: −0.25 [−0.47; +0.03]. Thêm prompt thì thắng có ý nghĩa nhưng chèn sai gấp đôi | 19 |
| List nhỏ hơn thì sao? | N100: Sạch@1 98.2, precision 97.8, WER 2.96, B-WER 7.14 | 21 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.49 so với 4.25. Đây là việc số 1 | 24, 25 |
| Đã có ai làm chưa? | Vị trí theo frame có ở CTC/transducer; PTC-Bias dùng vị trí để sửa sau decode. Chưa thấy ai nhắc đúng lúc cho decoder đóng băng, nhưng mới tra khoảng 10 lượt | 28 |
| So với số Hou25 công bố? | Đây là bản tái hiện 2 epoch; chưa có checkpoint của anh Hoàng để so cùng pipeline | 24 |
| Dùng được cho Qwen3-ASR không? | Chưa thử; Qwen không có alignment heads giống Whisper, cần cách khác để biết giây đang nghe (việc 6) | 25 |
| File, code ở đâu? | Dự phòng 3 | 29 |

## Đường tắt 12 phút (khi bị giục)

Trang **2** (tóm tắt) → **5** (Hou25 mất gì) → **10** (đo lại) → **12** (vệt chéo của anh Hoàng) → **15** (scorer đề xuất) → **16** (tranh nhau) → **19** (bảng Whisper) → **20** (3 phát hiện ASR) → **24** (chỗ yếu) → **26** (3 câu hỏi).

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
