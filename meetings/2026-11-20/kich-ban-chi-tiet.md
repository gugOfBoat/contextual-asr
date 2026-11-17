# Kịch bản nói cho slide `bao-cao-chi-tiet.pdf` (32 trang)

> Số liệu lấy từ `scorer/analysis/REPORT-1115.md`. Phần giải thích từng thí nghiệm nằm ở `scorer/analysis/TIEN-DO-1115.md`.
> Thời lượng: bản đầy đủ khoảng **23 phút** (trang 1–27). Nếu bị giục thì đi **đường tắt 12 phút**, xem cuối file. Trang 28–32 là dự phòng, chỉ mở khi thầy hỏi.

## Chuẩn bị trước khi vào họp (5 phút)

- [ ] Mở `bao-cao-chi-tiet.pdf` ở chế độ trình chiếu. Mở sẵn `kich-ban-chi-tiet.md` này ở màn hình thứ hai hoặc trên điện thoại.
- [ ] Mở sẵn `scorer/analysis/REPORT-1115.md` để tra số khi thầy hỏi sâu.
- [ ] Thuộc 3 con số:
  - Recall@1 (lọc): **72.6 → 89.3**.
  - B-WER test-other: **9.69 → 7.65**, không prompt.
  - Whisper cần vị trí: **−0.44 WER**.
- [ ] Thuộc 2 câu phải nói thật:
  - **WER tổng chưa thắng có ý nghĩa** (−0.17 [−0.39; +0.12]).
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
- Đọc 3 số to: Recall@1 (lọc) **72.6 → 89.3**; precision khi lọc list **55% → 86%** (cùng ngưỡng), còn ~2 từ/câu; B-WER **9.69 → 7.65**, có ý nghĩa.
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
- Đọc D1 làm ví dụ: Hou25 sửa xong phải đạt ≥ 85.7 thì mới giữ. Cột phải cho biết câu trả lời nằm ở trang nào (11, 21, 25).

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

## Phần 2: Scorer (trang 12–20, ~8 phút)

**Trang 12 – Scorer đề xuất từ đâu ra.** Trang này trả lời "cái hai nhánh ở đâu ra".
- **Insight:** "Scorer đề xuất là scorer âm vị, nền tốt nhất theo D1, cộng với bài học vệt chéo ở phần 1, tức là đọc đúng thứ tự, gộp trong một model hai nhánh."
- Đọc 4 ô từ trái sang:
  1. Phần 1: thứ tự loại được nhiễu.
  2. Nền âm vị tốt nhưng không đòi thứ tự.
  3. Thêm cách khớp đúng thứ tự: ít nhiễu nhưng sót nhiều hơn.
  4. Hai cách bù nhau, nên gộp lại và dạy vị trí: 89.3.

**Trang 13 – Hai cách khớp: kiến trúc và ví dụ số.** Trang quan trọng của phần 2, đi chậm.
- **Insight:** "Tự do: mỗi âm vị lấy ô sáng nhất, nên từ nhiễu có ô sáng rời rạc vẫn được 0.75. Đúng thứ tự: ô phải nối thành đường, nên nhiễu còn 0.20, và đường đi cho luôn đầu–cuối của từ."
- Hàng trên là phần chung của cả hai cách:
  - audio qua encoder Whisper đóng băng, mỗi 20 ms một vector 1280 chiều, chiếu xuống 256 rồi qua 1 lớp Transformer;
  - từ tách thành âm vị, mỗi âm vị một vector (nhúng + vị trí trong từ, 2 lớp Transformer). Phía từ tính sẵn được cho cả kho 209k;
  - nhân hai bên ra **bảng giống nhau**: hàng = âm vị, cột = khung 20 ms, ô = cos.
- Ví dụ đồ chơi: 4 âm vị × 8 khung, ô càng đậm càng giống. Từ đúng có 4 ô sáng nối nhau chéo xuống; từ nhiễu có 4 ô sáng nhưng rời rạc, sai thứ tự.
- **Nhánh tự do** (ô viền vàng): mỗi hàng lấy ô sáng nhất. Từ đúng (.9+.8+.9+.8)/4 = 0.85; từ nhiễu (.8+.7+.8+.7)/4 = 0.75. Nhiễu gần bằng từ đúng.
- **Nhánh đúng thứ tự** (đường xanh): đi từ âm vị đầu tới cuối, mỗi bước sang phải 1–G khung. Từ đúng vẫn 0.85; từ nhiễu không có đường nào qua các ô sáng, đường tốt nhất chỉ 0.20. Thật: G = 15 khung = 0.3 s; ví dụ dùng G = 2.
- **Vị trí** (dòng dưới mỗi cột):
  - tự do: tâm = trung vị các khung được chọn, độ rộng đoán theo số âm vị. Với từ nhiễu, khung 0, 1, 5, 7 cho tâm rơi vào chỗ chẳng có gì;
  - đúng thứ tự: lần ngược đường đi, đầu = khung của âm vị đầu, cuối = khung của âm vị cuối (khung 2–6).
- Số trong ví dụ là số đồ chơi, chỉ để thấy cơ chế. Số đo thật ở trang sau.

**Trang 14 – Hai cách khớp đo thật: mỗi con số nghĩa là gì.**
- **Insight:** "Tự do ít sót hơn; đúng thứ tự ít nhận nhầm và định vị chuẩn hơn. Sai ở chỗ khác nhau, nên cộng lại được +2.6, gấp đôi gộp hai model cùng kiểu."
- Đọc bảng trái theo từng dòng:
  - Recall@1 87.5 so với 85.8 (−1.7, có ý nghĩa): đòi thứ tự thì sót hơn. Với list nghe giống còn kém hơn: từ gần âm cũng tạo được đường.
  - Precision ở p ≥ 0.5: 75.8 lên 82.3, đổi lại recall 91.7 xuống 88.4.
  - Câu dài: 1.3 xuống 0.9 từ nhiễu mỗi câu.
  - Sai số tâm: 0.05 s xuống 0.02 s, tức 2.5 khung xuống 1 khung.
  - Lệch đầu từ: tự do có khi mở sớm 0.12 s, có khi muộn 0.08 s; đúng thứ tự luôn mở muộn 0.03–0.10 s, không bao giờ mở sớm. Mở sớm thì Whisper bị nhắc trước khi từ được nói.
- Cột "Nghĩa là" là diễn giải của em, chưa đo riêng.
- Bảng phải là phép thử trước khi xây model: hai model tự do gộp lại chỉ +1.1; tự do + thứ tự +2.6. Phần hơn 1.5 chỉ có khi hai cách nhìn khác nhau.

**Trang 15 – Scorer đề xuất chấm một từ thế nào.** Đi theo hình từ trái sang.
- **Insight:** "Lập bảng giống nhau âm vị × thời gian, rồi đọc bằng hai nhánh: nhánh tự do để không sót, nhánh đúng thứ tự để không nhận nhầm và ra luôn vị trí."
- Giờ chỉ cần nói: đây là hai cách ở trang 13, gộp vào một model, cộng điểm hai nhánh. Vị trí lấy từ nhánh đúng thứ tự.
- Ba điểm khác: khác vệt chéo Hou25 vì cho phép nói nhanh, chậm; mỗi nhánh có bảng riêng (chung bảng chỉ 88.3); lúc train từ đúng chỉ được khớp trong đoạn nó thật sự được nói.

**Trang 16 – Vị trí lấy ra đúng tới đâu.**
- **Insight:** "So với căn chỉnh MFA, đề xuất đặt tâm từ lệch 0.02 s, tức 1 khung, và 99.1% tâm nằm trong đoạn thật. Hou25 gốc chỉ 44%."
- Cách đo (hình bên trái): MFA căn transcript đúng vào audio, cho đoạn thật của mỗi từ hiếm. Scorer chấm từ đúng, cho đoạn đoán. Đo khoảng cách hai tâm, lấy trung vị trên 5248 lượt; và đếm bao nhiêu % tâm đoán nằm trong đoạn thật nới ra 0.1 s mỗi bên.
- Bảng: Hou25 gốc lệch 0.69 s, chỉ 44% trúng; sửa thành vệt chéo thì 98.9%. Ba scorer âm vị đều khoảng 99%, nhưng hai bản có nhánh thứ tự lệch 1 khung thay vì 2.5.
- Đúng@chỗ: tại đoạn thật của từ, nó có thắng mọi từ nhiễu bị đặt chồng lên đó không. Đề xuất 96.1%.
- Vị trí này được dùng ở hai chỗ: lọc list (trang 19) và nhắc Whisper đúng lúc (trang 23, −0.44 WER).
- **Phải nói (chữ đỏ):** chỉ đo được trên từ đúng, vì từ nhiễu không có đoạn thật; 0.9% trượt chưa xem từng ca; lệch đầu/cuối của bản đề xuất chưa tách riêng; MFA cũng là ước lượng.

**Trang 17 – Kết quả: mỗi bước cộng thêm bao nhiêu.**
- **Insight:** "89.3 Recall@1, hơn mốc mạnh nhất là Hou25 vệt chéo 6.4 điểm. Với từ Whisper tự nghe sai: 60.9 lên 70.9."
- Đọc bảng từ trên xuống: mỗi dòng thêm một thứ.
- **Phải nói (chữ đỏ):** hai chỗ lệch có lợi cho nhóm. Hou25 train 2 epoch, scorer âm vị 6 epoch và chọn checkpoint trên dev-other. Bản đề xuất là bản tốt nhất trong 4 bản hai nhánh xếp theo test.

**Trang 18 – Lỗi câu dài không còn.**
- **Insight:** "Chỉ Hou25 lấy trung bình mới báo nhầm tăng theo độ dài câu. Scorer đề xuất: 0.9 từ nhiễu mỗi câu dài, so với 12.4."
- Chỉ vào đường xám đi lên, các đường khác nằm ngang.
- Nói thật: nhóm câu ≥ 20 s chỉ có 40 câu. Khớp tự do không đòi thứ tự mà vẫn thấp, chưa biết vì sao.

**Trang 19 – Biết vị trí thì lọc được list.**
- **Insight:** "Cùng ngưỡng p ≥ 0.5, cùng luật trùng chỗ thì giữ từ điểm cao hơn: đề xuất giữ 92% từ đúng, precision 86%; Hou25 chỉ 73% và 55%, vì nó không biết từ nằm đâu." Nối lại trang 16: vị trí Hou25 chỉ trúng 44%.
- Cách tính bên trái: lấy top-50, cho từ trùng chỗ tranh nhau, giữ p ≥ 0.5. Precision = từ đúng còn lại chia cho mọi từ còn lại, cộng trên cả 2939 câu.
- Nói rõ: số cũ "3–4% lên 91%" đã bỏ, vì 3–4% là top-50 không cắt (chọn 50 từ thì precision tối đa khoảng 3.6%), còn 91% là bản hai model.

**Trang 20 – Chỗ yếu.**
- **Insight:** "Từ nhiễu nghe giống một từ thường trong câu thì không có từ hiếm nào tranh chỗ, nên nó lọt."
- Câu B: flauve nghe như "fall", emperour nghe như "empire". Từ đây ra việc số 1.

## Whisper (trang 21–24, ~4 phút)

**Trang 21 – Cách nhắc đúng lúc.**
- **Insight:** "Whisper hầu như không viết từ vào sai chỗ, nhưng hay viết thêm khi không cần. Vị trí dùng để biết lúc nào phải im lặng."
- Ba bước: đọc alignment heads để biết Whisper đang nghe giây nào; từ nào nằm ở giây đó thì cộng λ = 5; mỗi từ chỉ nhắc đến khi đã viết ra.

**Trang 22 – Kết quả.**
- **Insight:** "Không cần prompt, B-WER test-other 9.69 xuống 7.65, test-clean 5.15 xuống 3.83." Đây là cùng một model với trang 17.
- B-WER chỉ tính lỗi trên từ hiếm của câu đúng, chấm bằng script IS21 chính thức.
- **Phải nói rõ:** WER tổng −0.17, khoảng tin cậy chứa 0, nên chưa có ý nghĩa. Chữ đỏ: chưa tách được lợi là do scorer hay do cách nhắc, vì thiếu dòng Hou25 vệt chéo + nhắc đúng lúc.

**Trang 23 – Ba điều rút ra phía ASR.**
- **Insight:** "Scorer chỉ có lời khi đưa ra (từ, p, vị trí) và được dùng lúc decode."
- Ba ô: cần vị trí (nhắc mọi lúc WER 3.66, đúng lúc 3.22); prompt đã đầy (scorer hơn 17 điểm mà đưa vào prompt vẫn không hơn); còn 1.7 tới trần, một nửa do vị trí.

**Trang 24 – Theo cỡ list.**
- **Insight:** "List dài hơn 20 lần, precision 98 xuống 70%, nhưng B-WER chỉ 7.14 lên 7.65."
- Nói thật: chưa có Hou25 ở N100–1000.

## Tổng hợp (trang 25, ~2 phút)

**Trang 25 – Ba câu hỏi đã có trả lời, còn gì chưa chắc.** Trang này gom cả phần tổng hợp vào một chỗ.
- **Insight, câu chuyện một dòng:** "Hou25 bỏ vị trí nên thua. Scorer giữ vị trí: 89.3, lọc còn khoảng 2 từ mỗi câu. Whisper dùng vị trí: B-WER 7.65, không cần prompt. Trước khi viết paper còn 4 việc phải sửa."
- Bên trái: D1, D2, D3, mỗi cái một luật đặt trước và kết quả. D3 nói rõ: ngưỡng và λ chọn trên dev, **nhưng** biến thể scorer lại chọn theo test.
- Bên phải (ô đỏ): 4 chỗ chưa chắc, mỗi chỗ chỉ sang một việc ở trang sau.
- Nếu thầy hỏi chi tiết so có công bằng không thì mở Dự phòng 1; giả thuyết so với thực tế thì mở Dự phòng 2.

## Đi tiếp (trang 26–27, ~2 phút + thảo luận)

**Trang 26 – Bước tiếp theo.**
- **Insight:** "Việc 1–3 cho ra scorer cuối, xong 20/12. Việc 4–6 làm đủ bảng cho paper, xong 17/01."
- Đọc cột "Vì sao": mỗi việc sửa một chỗ yếu ở trang 25. Nói "việc 6 là theo ý thầy".

**Trang 27 – Mốc và 3 câu hỏi.** Đọc từng câu, rồi **dừng lại chờ thầy trả lời**, ghi lại nguyên văn.
1. Đóng góp chính đặt ở scorer, ở cách nhắc, hay gộp cả hai?
2. Ngoài Hou25, có cần so với PTC-Bias và cách nhắc không theo vị trí không?
3. Whisper + Qwen3-ASR đã đủ chưa?

## Dự phòng (trang 28–32)

- **Trang 28 – Các con số lợi thế: tính thế nào, so có công bằng không.** Đọc cột đỏ: (1) số epoch và chọn bản trên test; (2) so cả hệ, prompt với nhắc; (3) câu dài chỉ 40 câu. Đây là việc số 4.
- **Trang 29 – Giả thuyết so với thực tế.** Giả thuyết sai có ích nhất: Hou25 sai vì từ nhiễu nổi lên, không phải vì từ đúng bị loãng.
- **Trang 30–32:** định nghĩa thước đo, liên quan và điểm mới, file và cách tái lập.

---

## Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn | Mở trang |
|---|---|---|
| Recall@1 (lọc) khác Recall thế nào? | Chỉ đếm từ **nhiễu** đứng trên, nên câu nhiều từ hiếm không bị phạt oan | 6, 30 |
| Sao không so top-50 như paper? | Có báo top-50 (95.4 / 98.4), nhưng mình chỉ đưa ~2 từ cho ASR nên phải đo ở đầu bảng | 8 |
| 73.3 và 72.6 khác gì nhau? | 73.3 là Hou25 train cùng lượt với bản vệt chéo (so công bằng); 72.6 là bản dùng cho các bảng khác | 10 |
| Vì sao tác giả Hou25 lấy trung bình? | Paper không nêu lý do (theo ghi chép của nhóm). Em đoán: bản đồ có kích thước thay đổi, mục tiêu của họ chỉ là top-50 cho prompt nên không cần vị trí, và train chỉ cần nhãn có/không | 10 |
| Vệt chéo có phải ý mới không? | Không claim. Gần với dò từ khoá bằng audio mẫu (DTW); cái mới là chỉ ra cách gộp là nguồn lỗi của scorer, và dùng vị trí cho ASR | 11 |
| Vệt dốc 1 giả định người nói cùng tốc độ giọng máy? | Đúng là giả định vậy. CNN nhìn được vùng 9×9 nên bù được phần nào, nhưng chưa đo riêng | 10 |
| Từ nghe giống thì sao? | List nghe giống: âm vị 64.7 so với giọng máy 51.4; hai nhánh đạt 67.7 | 11, 17 |
| Vì sao không dùng luôn bản cộng điểm 90.8? | Tốn 2 model; việc 3 là cho một model đuổi kịp | 17, 26 |
| Vị trí scorer đưa ra đúng tới đâu? | Tâm lệch 0.02 s so với MFA, 99.1% trúng đoạn thật ± 0.1 s; chỉ đo trên từ đúng | 16 |
| Lấy giây đang nghe kiểu gì? | Alignment heads của Whisper, như Simul-Whisper (2406.10052 §2.1) | 21 |
| Tham số chọn ở đâu? | dev-other 400 câu, không chọn trên test | 5, 21 |
| WER tổng có thắng không? | Chưa: −0.17 [−0.39; +0.12]. Thêm prompt thì thắng có ý nghĩa nhưng chèn sai gấp đôi | 22 |
| List nhỏ hơn thì sao? | N100: Recall@1 (lọc) 98.2, precision 97.8, WER 2.96, B-WER 7.14 | 24 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.49 so với 4.25. Đây là việc số 1 | 25, 26 |
| Đã có ai làm chưa? | Vị trí theo frame có ở CTC/transducer; PTC-Bias dùng vị trí để sửa sau decode. Chưa thấy ai nhắc đúng lúc cho decoder đóng băng, nhưng mới tra khoảng 10 lượt | 31 |
| So với số Hou25 công bố? | Đây là bản tái hiện 2 epoch; chưa có checkpoint của anh Hoàng để so cùng pipeline | 25 |
| Dùng được cho Qwen3-ASR không? | Chưa thử; Qwen không có alignment heads giống Whisper, cần cách khác để biết giây đang nghe (việc 6) | 26 |
| File, code ở đâu? | Dự phòng 5 | 32 |

## Đường tắt 12 phút (khi bị giục)

Trang **2** (tóm tắt) → **7** (hai scorer) → **8** (đo ở hạng 1) → **9** (Hou25 thua vì đâu) → **11** (luật D1) → **12** (scorer đề xuất từ đâu ra) → **13** (hai cách khớp) → **16** (vị trí đúng tới đâu) → **17** (kết quả) → **19** (lọc list) → **22** (kết quả Whisper) → **23** (3 điều phía ASR) → **25** (tổng hợp) → **27** (3 câu hỏi).

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
