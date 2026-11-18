# Kịch bản nói cho slide `bao-cao-chi-tiet.pdf` (36 trang)

> Số liệu lấy từ `scorer/analysis/REPORT-1115.md`, `e_dual_other.md`, `nms_fair_other.md`. Giải thích từng thí nghiệm: `scorer/analysis/TIEN-DO-1117.md`.
> Thời lượng: bản đầy đủ khoảng **25 phút**. Bị giục thì đi **đường tắt 12 phút** (cuối file).

## Chuẩn bị trước khi vào họp (5 phút)

- [ ] Mở `bao-cao-chi-tiet.pdf` ở chế độ trình chiếu, kịch bản này ở màn hình thứ hai.
- [ ] Mở sẵn `scorer/analysis/REPORT-1115.md` để tra số.
- [ ] Thuộc 3 con số:
  - Recall@1 (lọc): **72.6 → 89.3**.
  - B-WER test-other: **9.69 → 7.65**, không prompt.
  - Whisper cần vị trí: **−0.44 WER**.
- [ ] Thuộc 2 câu phải nói thật:
  - **WER tổng chưa thắng có ý nghĩa** (−0.17 [−0.39; +0.12]).
  - **Hou25 là bản nhóm tự tái hiện**, 2 epoch.
- [ ] Mở một file trống để ghi ý kiến thầy.

## Ba quy tắc khi nói

1. Mỗi trang: đọc **ô xanh đậm** trước, rồi chỉ vào con số làm bằng chứng.
2. Nói "thắng Hou25" thì phải kèm chữ **B-WER**.
3. Vệt chéo là ý của **anh Hoàng**: nhắc tên ở trang 12 và 13.

---

## Mở đầu (trang 1–3, ~2 phút)

**Trang 1 – Bìa.** "Em báo cáo 13–15/11. Câu hỏi chính: scorer biết từ nằm ở giây nào thì có lợi gì?"

**Trang 2 – Tóm tắt.**
- Insight: giữ lại "từ nằm ở giây nào" có lợi cả hai phía.
- Ba số:
  - Recall@1 (lọc) **72.6 → 89.3**.
  - Lọc list: precision **55% → 86%** cùng ngưỡng; mỗi câu vào 2000 từ, ra **1.9 từ** (giải thích ở trang 30).
  - B-WER **9.69 → 7.65**.
- Dòng đỏ: WER tổng mới ngang Hou25.

**Trang 3 – Mạch.** "Em đi 5 phần: đặt câu hỏi, vì sao lại là vị trí, scorer, Whisper, đúc kết."

## Phần 0: Đặt câu hỏi (trang 4–9, ~5 phút)

**Trang 4 – Từ điển.** Chỉ cần nhớ Hou25 (đối thủ) và scorer đề xuất; đưa vào Whisper bằng prompt top-50 hoặc nhắc đúng lúc.

**Trang 5 – Ba câu hỏi + luật đặt trước.** D1 trả lời ở trang 14, D2 ở trang 34, D3 ở trang 36.

**Trang 6 – Thước đo.** Đi theo ví dụ trong bảng:
- "Câu có 2 từ hiếm: assyria và pileser. List 2000 từ. Scorer xếp hạng."
- "pileser đứng hạng 2, nhưng người đứng trên là assyria, cũng đúng, nên không tính là lỗi → coi là hạng 1 → trúng."
- "Nếu flauve, một từ nhiễu, được 0.95 thì nó đứng trên pileser → trượt. Nghĩa là flauve sẽ tới ASR trước pileser."
- "Recall@1 (lọc) = số lượt trúng / 5248 lượt từ hiếm."
- Vì sao không top-50: Hou25 nhét 50 từ vào prompt; nhóm chỉ giữ ~2 từ/câu, nên từ đúng ở hạng 30 không bao giờ tới ASR.

**Trang 7 – Hai scorer.** Hou25 cho máy đọc từ rồi so hai đoạn audio; scorer âm vị so audio thẳng với cách phát âm.

**Trang 8 – Đo lại ở hạng 1.**
- "Cùng hai scorer, cùng 2939 câu, chỉ đổi thước đo."
- Hou25 = nhóm tái hiện, gộp bằng **trung bình** (trang 11). Khớp tự do = scorer âm vị bản đầu (trang 19).
- "Top-50: 95.4 và 98.4, gần như nhau. Hạng 1: 72.6 và 87.5. Với Hou25, 22.8 điểm từ đúng lọt top-50 nhưng có từ nhiễu đứng trên."

**Trang 9 – Bốn cột của biểu đồ.** Đọc bảng từ trên xuống; mỗi dòng chỉ đổi ô vàng so với dòng trên:
- Hou25: audio máy đọc, CNN, trung bình mọi ô → 72.6.
- Vệt chéo: chỉ đổi cách gộp, train lại → 82.9 (+10.3; mốc cùng lượt 73.3 nên chênh sạch là +9.5). Nếu bị hỏi "vệt chéo là gì": hàng = từng 20 ms audio máy đọc, cột = từng 20 ms câu; từ có trong câu thì đầu khớp đầu, sau khớp sau → ô sáng xếp chéo; chỉ lấy trung bình dọc đường chéo sáng nhất.
- Khớp tự do: đổi audio máy đọc sang dãy âm vị, mỗi âm vị lấy ô sáng nhất → 87.5.
- Đề xuất: thêm nhánh đúng thứ tự (bảng riêng) và dạy vị trí → 89.3, tức khoảng 4687 trên 5248 lượt từ đúng không bị nhiễu nào vượt.
- Nói luôn chỗ chưa công bằng: Hou25 train 2 lượt, scorer âm vị 6 lượt; mỗi hệ chỉ train 1 lần.

## Phần 1: Vì sao vị trí (trang 10–14, ~5 phút)

**Trang 10 – Hou25 thua vì đâu.**
- 96% người vượt là từ khác hẳn.
- 76% không giống âm nào trong câu.
- Từ nhiễu điểm cao tăng từ 1.4 (câu ngắn) lên 12.4 (câu ≥ 20 s).

**Trang 11 – Hou25 chấm thế nào.** Chỉ theo hình từ trái sang phải:
1. Máy đọc từ, rồi encoder.
2. Cross-attention.
3. 4 bảng cos (hàng = khung câu, cột = khung từ).
4. CNN.
5. **Trung bình mọi ô**, rồi Linear.

Rồi chỉ vào công thức: "Vì Linear cuối là tuyến tính, nên trung bình rồi Linear = Linear từng ô rồi trung bình. Vậy điểm = trung bình điểm của từng ô. Mỗi ô qua 4 lớp 3×3 chỉ nhìn 9×9 ô, khoảng 0.18 s, nên mảnh ở giây 1 và mảnh ở giây 7 được cộng như nhau."

**Trang 12 – Ví dụ số.**
- Trung bình: đúng (3 ô nối chéo) 0.20, nhiễu (4 ô rải rác) **0.23** → nhiễu thắng.
- Chứng minh: trung bình = tổng / 24; đổi chỗ ô thì tổng không đổi, nên điểm không chứa thứ tự.
- Vệt chéo (anh Hoàng): cộng dọc đường chéo, lấy đường tốt nhất: đúng 2.7, nhiễu 1.1; đường đó cho luôn chỗ của từ.

**Trang 13 – Đo thật.** Cùng lượt train, chỉ đổi cách gộp:
- Vệt chéo 73.3 → **82.9** (+9.5 [+7.9; +11.2]).
- "Chỗ sáng nhất" chỉ 73.8 → cái giúp là **thứ tự**, không phải chỗ sáng.
- Đọc ba cột theo phần "đọc số" bên dưới. Chỗ sáng nhất chỉ 4% tâm trúng vì các hàng sáng rải khắp câu.

**Trang 14 – D1.** Vệt chéo 82.9 vẫn dưới luật 85.7 → xây tiếp trên âm vị, mang "thứ tự" sang.

## Phần 2: Scorer (trang 15–31, ~10 phút) — phần chính

**Trang 15 – Từ đâu ra.** 4 bước: bài học thứ tự → nền âm vị → thêm thứ tự → gộp + dạy vị trí.

**Trang 16 – Tổng quan.** Bảng âm vị × thời gian, hai nhánh đọc, ra (từ, p, đầu, cuối). "Các trang sau đi từng bước."

**Trang 17 – Bảng M.**
1. G2P: pileser → P AY L IY Z ER.
2. Âm vị → vector, tính sẵn cho 209k từ.
3. Audio → 1 vector / 20 ms.
4. M = cos.

"Cả 2000 từ chỉ là một phép nhân ma trận, 4–7 ms/câu."

**Trang 18 – Nhìn bằng hình.** "Mỗi hình là một bảng: mỗi hàng là một âm của từ, mỗi cột là 20 ms audio, ô càng đậm là càng giống." Âm lấy từ từ điển phát âm (bartley = B AA R T L IY). Độ đậm = cosine giữa vector của âm và vector của 20 ms audio; điểm một cách khớp = trung bình các ô được chọn. Số trên hình là số giả.
- Hàng trên, từ thật bartley: cách tự do (vòng cam, mỗi âm chọn ô đậm nhất) và cách đúng thứ tự (đường xanh, âm sau phải nói sau âm trước) đều đi chéo → cả hai 0.87.
- Hàng dưới, từ sai ghetto: 4 ô đậm nằm lung tung. Cách tự do vẫn gom được → 0.79, gần bằng từ thật. Cách đúng thứ tự không nối được → 0.48, rớt xuống dưới bartley.
- "Ba trang sau tính tay đúng ý này trên bảng nhỏ."

**Trang 19 – Khớp tự do.**
- Công thức: mỗi hàng lấy max, rồi trung bình.
- Chỉ ô viền vàng:
  - từ đúng chọn khung 2, 3, 4, 6 → 0.85;
  - từ nhiễu chọn khung 5, 0, 7, 1 (lộn xộn) → **0.75**, gần bằng.
- Vị trí: trung vị các khung chọn. Với nhiễu, trung vị rơi vào giữa hai cụm, chỗ không có gì.

**Trang 20 – Khớp đúng thứ tự (DP).**
- Đọc công thức: "D[p,t] là đường tốt nhất qua âm vị 1 tới p, kết thúc ở khung t. Âm vị trước phải ở trước 1 tới G khung."
- Tính tay 3 ô xanh:
  - 1.7 = .8 + max(.9, .1);
  - 2.6 = .9 + max(1.7, .5);
  - 3.4 = .8 + max(2.0, 2.6).
- Kết quả: 3.4/4 = **0.85**.
- "Từ đúng thì tự do và thứ tự bằng nhau, vì ô sáng đã theo thứ tự."

**Trang 21 – Từ nhiễu + lần ngược.**
- Đường vàng đi qua ô sáng tới 1.8 ở khung 7, nhưng âm vị 4 phải ở sau khung 7 → hết chỗ → tốt nhất chỉ 0.8/4 = **0.20**.
- Lần ngược từ đúng: 3.4 ở khung 6 ← 2.6 ở khung 4 ← 1.7 ở khung 3 ← 0.9 ở khung 2 → từ ở khung 2–6.

**Trang 22 – Ra p.**
- logit = a·thứ tự + a2·tự do + c.
- Với giá trị khởi tạo (nói rõ là minh hoạ):
  - chỉ tự do: nhiễu p = 0.92, **lọt** ngưỡng 0.5;
  - hai nhánh: p = 0.38, **bị loại**.
- Luôn có tự do ≥ thứ tự. Chỉ dùng thứ tự thì sót, vì một âm vị khớp kém kéo cả đường xuống.

**Trang 23 – Train (1).** Đọc bảng 5 dòng:
- Câu có bartley → 1 cặp đúng + 4 cặp nhiễu (2 gần chính tả: bentley, bartlett; 2 ngẫu nhiên: kirkland, ghetto). Thật ra mỗi nhiễu bốc thăm: 40% lấy trong 20 từ gần chính tả nhất, 60% lấy ngẫu nhiên trong kho 209k; ví dụ này ra 2 + 2.
- Mỗi cặp một số phạt = −ln(xác suất máy cho đáp án đúng): đoán càng sai, phạt càng nặng. bentley (viết gần giống) bị phạt nặng nhất, −ln 0.60 = 0.51. Phạt của câu = trung bình = 0.21.
- Nếu bị hỏi "xác suất ở đâu ra": điểm thô = a × điểm thứ tự + b × điểm tự do + c (a, b, c học được), qua sigmoid ra 0–1.
- Mỗi bước 8 câu, 3568 bước là một lượt qua hết dữ liệu, chạy 6 lượt. Hết mỗi lượt chấm Recall@1 trên 500 câu dev, giữ bản cao nhất (vòng tròn trên hình).
- Nhấn: Recall@1 không dùng để train, chỉ để chọn bản. Dev 500 câu lên xuống 2–3 điểm nên bản được chọn có phần may rủi.

**Trang 24 – Train (2): dạy vị trí.** Hình trái: không che thì T "mượn" chữ t của từ sau ở khung 21 vẫn ăn điểm. Hình phải: MFA (công cụ căn chữ vào audio) cho biết từ nằm ở đâu; ngoài đoạn đó (nới 0.1 s) xoá hết, chỉ với từ đúng, chỉ lúc train. Đo trên khớp tự do: Recall@1 87.5 → 88.3. Chưa có số riêng cho "chỉ thêm thứ tự"; 89.3 là cả ba cộng lại.

**Trang 25 – Đo thật hai cách.**
- Thứ tự ít nhầm hơn: precision 75.8 → 82.3.
- Nhưng sót hơn: Recall@1 87.5 → 85.8, khoảng tin cậy −1.7 [−2.7; −0.7].
- Đọc cột "Vì sao" (đây là diễn giải).
- Ô dưới: cộng 2 model khác cách được +2.6; cùng cách chỉ +1.1 → hai cách bù nhau thật → gộp thành một model.

**Trang 26 – Vị trí ra từ đâu.** "Vị trí không học riêng, nó là chính con đường đã cho điểm."
- Đi ngược con đường từ âm cuối: IY ở cột 16, L 14, T 12, R 10, AA 8, B ở cột 6.
- Cột đánh số từ 0, cột t phủ t×0.02 → (t+1)×0.02 s: cột 6 bắt đầu 0.12 s, cột 16 kết thúc 0.34 s → từ nằm ở 0.12–0.34 s. Tâm = điểm giữa đoạn.
- Đáp án căn tay (MFA, công cụ căn chữ vào audio) nói 0.10–0.36 s; hai tâm đều 0.23 s → trúng.
- Cách tự do không có con đường nên phải đoán tâm bằng trung vị các ô đã chọn, dễ lệch hơn.

**Trang 27 – Vị trí đúng tới đâu.** Tâm lệch 0.02 s, 99.1% trúng; Hou25 gốc 44%. Chỉ đo trên từ đúng.

**Trang 28 – Từng bước.**
- Dạy vị trí +0.8.
- Thêm nhánh, chung bảng: Recall@1 đứng yên, nhưng nhóm nghe sai +2.9.
- Tách hai bảng +1.0.
- Tổng: **89.3**; nhóm Whisper nghe sai **70.9**.
- Đọc dòng đỏ: lệch có lợi cho nhóm.

**Trang 29 – Câu dài.** Hou25 12.4 nhiễu/câu, đề xuất 0.9.

**Trang 30 – Lọc list.** Đọc phép tính bên trái:
- Từ đúng còn lại: 92.4% × 5248 = 4849, tức 1.65 mỗi câu.
- Từ nhiễu còn lại: 0.27 × 2939 = 794.
- Tổng 5643 / 2939 = **1.92 từ/câu**. Câu thật có 1.79 từ hiếm, nên list sau lọc gần bằng danh sách thật. Precision 85.9%.
- Hou25: 1.30 đúng + 1.06 nhiễu. Bước "trùng chỗ" làm Hou25 mất 14 điểm, vì vị trí của nó sai.

**Trang 31 – Chỗ yếu.** Nhiễu nghe như một từ thường (flauve ≈ "fall") thì không có từ hiếm nào tranh chỗ, nên nó lọt.

## Whisper (trang 32–35, ~4 phút)

**Trang 32 – Cách nhắc đúng lúc.** Alignment heads cho biết giây đang nghe. Từ nằm ở giây đó thì được cộng λ = 5, mỗi từ một lần.

**Trang 33 – Kết quả.**
- B-WER 9.69 → **7.65**, test-clean 5.15 → 3.83.
- **Nói rõ:** WER tổng −0.17, chưa có ý nghĩa.

**Trang 34 – Ba điều phía ASR (D2).** Nhắc đúng lúc 3.22 so với nhắc mọi lúc 3.66: cần vị trí.

**Trang 35 – Theo cỡ list.** Đọc bảng:
- Từ N100 lên N2000, nhiễu lọt qua tăng 0.04 → 0.73 từ/câu.
- Nhưng B-WER chỉ 7.14 → 7.65, WER 2.96 → 3.22.
- "Mỗi câu thêm 0.7 từ nhiễu được nhắc, mà phần lớn Whisper không viết ra, vì nhiễu chỉ được nhắc ở đoạn của nó và audio không ủng hộ."
- Lưu ý: precision 70% ở đây là ngưỡng 0.3, khác 86% ở trang 30 (ngưỡng 0.5).
- Nói thật: chưa có Hou25 ở N100–1000.

## Đúc kết (trang 36, ~1 phút)

**Trang 36.** Đọc ba ô từ trái sang phải:
1. Hou25 thua vì trung bình.
2. Scorer giữ vị trí: 89.3, 0.02 s, 1.9 từ/câu.
3. Whisper dùng vị trí: 7.65.

Rồi đọc ô đỏ "chưa chắc". Sau đó **dừng lại để thầy góp ý.**

Nếu còn thời gian, hỏi miệng thầy 3 câu (ghi nguyên văn):
1. Đóng góp chính đặt ở scorer, ở cách nhắc, hay gộp cả hai?
2. Ngoài Hou25 có cần so với PTC-Bias không?
3. Whisper + Qwen3-ASR đã đủ chưa?

---

## Nếu thầy hỏi

| Câu hỏi | Trả lời ngắn | Mở trang |
|---|---|---|
| Recall@1 (lọc) khác Recall thế nào? | Chỉ đếm từ **nhiễu** đứng trên, nên câu nhiều từ hiếm không bị phạt oan | 6 |
| Sao không đo top-50 như paper? | Có báo top-50 (95.4 / 98.4), nhưng mình chỉ đưa ~2 từ cho ASR | 6, 8 |
| Vì sao nói trung bình mất thứ tự, khi CNN có nhìn hàng xóm? | CNN chỉ nhìn 9×9 ô (0.18 s). Linear cuối tuyến tính, nên điểm = trung bình điểm từng ô; các mảnh cách nhau > 9 ô đổi chỗ được mà điểm gần như không đổi | 10, 11 |
| 73.3 và 72.6 khác gì nhau? | 73.3 là Hou25 train cùng lượt với bản vệt chéo; 72.6 là lượt trước | 12 |
| Vệt chéo có phải ý mới không? | Không claim. Gần dò từ khoá bằng audio mẫu (DTW); cái mới là chỉ ra cách gộp là nguồn lỗi, và dùng vị trí cho ASR | 11 |
| G = 15 khung lấy từ đâu? | Nhóm tự đặt (loại C): hai âm vị liền nhau cách nhau tối đa 0.3 s |  18 |
| Hệ số a, c trong ví dụ có phải số thật? | Không, là giá trị khởi tạo, chỉ để minh hoạ cơ chế | 20 |
| G2P đoán sai thì sao? | Nhánh tự do chịu được (mỗi âm vị tính riêng); nhánh thứ tự bị kéo xuống | 20, 22 |
| Vì sao không dùng luôn bản cộng điểm 90.8? | Tốn 2 model; mục tiêu là một model đuổi kịp | 22, 24 |
| Vị trí đúng tới đâu? | Tâm lệch 0.02 s so với MFA, 99.1% trúng; chỉ đo trên từ đúng | 23 |
| 1.9 từ/câu tính sao? | (4849 đúng + 794 nhiễu) / 2939 câu | 26 |
| Lấy giây đang nghe kiểu gì? | Alignment heads của Whisper, như Simul-Whisper (2406.10052 §2.1) | 28 |
| WER tổng có thắng không? | Chưa: −0.17 [−0.39; +0.12] | 29 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.51 so với 4.25 | 32 |
| So với số Hou25 công bố? | Bản tái hiện 2 epoch; chưa có checkpoint của anh Hoàng | 24 |

## Đường tắt 12 phút (khi bị giục)

Trang **2** → **6** → **8** → **9** → **11** → **12** → **14** → **17** → **18** → **20** → **22** → **23** → **24** → **26** → **28** → **30** → **33** → **36**.

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
