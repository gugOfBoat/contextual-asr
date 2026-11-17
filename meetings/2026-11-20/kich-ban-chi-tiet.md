# Kịch bản nói cho slide `bao-cao-chi-tiet.pdf` (32 trang)

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
3. Vệt chéo là ý của **anh Hoàng**: nhắc tên ở trang 11 và 12.

---

## Mở đầu (trang 1–3, ~2 phút)

**Trang 1 – Bìa.** "Em báo cáo 13–15/11. Câu hỏi chính: scorer biết từ nằm ở giây nào thì có lợi gì?"

**Trang 2 – Tóm tắt.**
- Insight: giữ lại "từ nằm ở giây nào" có lợi cả hai phía.
- Ba số:
  - Recall@1 (lọc) **72.6 → 89.3**.
  - Lọc list: precision **55% → 86%** cùng ngưỡng; mỗi câu vào 2000 từ, ra **1.9 từ** (giải thích ở trang 26).
  - B-WER **9.69 → 7.65**.
- Dòng đỏ: WER tổng mới ngang Hou25.

**Trang 3 – Mạch.** "Em đi 5 phần: đặt câu hỏi, vì sao lại là vị trí, scorer, Whisper, đúc kết."

## Phần 0: Đặt câu hỏi (trang 4–8, ~4 phút)

**Trang 4 – Từ điển.** Chỉ cần nhớ Hou25 (đối thủ) và scorer đề xuất; đưa vào Whisper bằng prompt top-50 hoặc nhắc đúng lúc.

**Trang 5 – Ba câu hỏi + luật đặt trước.** D1 trả lời ở trang 13, D2 ở trang 30, D3 ở trang 32.

**Trang 6 – Thước đo.** Đi theo ví dụ trong bảng:
- "Câu có 2 từ hiếm: assyria và pileser. List 2000 từ. Scorer xếp hạng."
- "pileser đứng hạng 2, nhưng người đứng trên là assyria, cũng đúng, nên không tính là lỗi → coi là hạng 1 → trúng."
- "Nếu flauve, một từ nhiễu, được 0.95 thì nó đứng trên pileser → trượt. Nghĩa là flauve sẽ tới ASR trước pileser."
- "Recall@1 (lọc) = số lượt trúng / 5248 lượt từ hiếm."
- Vì sao không top-50: Hou25 nhét 50 từ vào prompt; nhóm chỉ giữ ~2 từ/câu, nên từ đúng ở hạng 30 không bao giờ tới ASR.

**Trang 7 – Hai scorer.** Hou25 cho máy đọc từ rồi so hai đoạn audio; scorer âm vị so audio thẳng với cách phát âm.

**Trang 8 – Đo lại ở hạng 1.**
- "Cùng hai scorer, cùng 2939 câu, chỉ đổi thước đo."
- Hou25 = nhóm tái hiện, gộp bằng **trung bình** (trang 10). Khớp tự do = scorer âm vị bản đầu (trang 17).
- "Top-50: 95.4 và 98.4, gần như nhau. Hạng 1: 72.6 và 87.5. Với Hou25, 22.8 điểm từ đúng lọt top-50 nhưng có từ nhiễu đứng trên."

## Phần 1: Vì sao vị trí (trang 9–13, ~5 phút)

**Trang 9 – Hou25 thua vì đâu.**
- 96% người vượt là từ khác hẳn.
- 76% không giống âm nào trong câu.
- Từ nhiễu điểm cao tăng từ 1.4 (câu ngắn) lên 12.4 (câu ≥ 20 s).

**Trang 10 – Hou25 chấm thế nào.** Chỉ theo hình từ trái sang phải:
1. Máy đọc từ, rồi encoder.
2. Cross-attention.
3. 4 bảng cos (hàng = khung câu, cột = khung từ).
4. CNN.
5. **Trung bình mọi ô**, rồi Linear.

Rồi chỉ vào công thức: "Vì Linear cuối là tuyến tính, nên trung bình rồi Linear = Linear từng ô rồi trung bình. Vậy điểm = trung bình điểm của từng ô. Mỗi ô qua 4 lớp 3×3 chỉ nhìn 9×9 ô, khoảng 0.18 s, nên mảnh ở giây 1 và mảnh ở giây 7 được cộng như nhau."

**Trang 11 – Ví dụ số.**
- Trung bình: đúng (3 ô nối chéo) 0.20, nhiễu (4 ô rải rác) **0.23** → nhiễu thắng.
- Chứng minh: trung bình = tổng / 24; đổi chỗ ô thì tổng không đổi, nên điểm không chứa thứ tự.
- Vệt chéo (anh Hoàng): cộng dọc đường chéo, lấy đường tốt nhất: đúng 2.7, nhiễu 1.1; đường đó cho luôn chỗ của từ.

**Trang 12 – Đo thật.** Cùng lượt train, chỉ đổi cách gộp:
- Vệt chéo 73.3 → **82.9** (+9.5 [+7.9; +11.2]).
- "Chỗ sáng nhất" chỉ 73.8 → cái giúp là **thứ tự**, không phải chỗ sáng.
- Đọc ba cột theo phần "đọc số" bên dưới. Chỗ sáng nhất chỉ 4% tâm trúng vì các hàng sáng rải khắp câu.

**Trang 13 – D1.** Vệt chéo 82.9 vẫn dưới luật 85.7 → xây tiếp trên âm vị, mang "thứ tự" sang.

## Phần 2: Scorer (trang 14–27, ~10 phút) — phần chính

**Trang 14 – Từ đâu ra.** 4 bước: bài học thứ tự → nền âm vị → thêm thứ tự → gộp + dạy vị trí.

**Trang 15 – Tổng quan.** Bảng âm vị × thời gian, hai nhánh đọc, ra (từ, p, đầu, cuối). "Các trang sau đi từng bước."

**Trang 16 – Bảng M.**
1. G2P: pileser → P AY L IY Z ER.
2. Âm vị → vector, tính sẵn cho 209k từ.
3. Audio → 1 vector / 20 ms.
4. M = cos.

"Cả 2000 từ chỉ là một phép nhân ma trận, 4–7 ms/câu."

**Trang 17 – Khớp tự do.**
- Công thức: mỗi hàng lấy max, rồi trung bình.
- Chỉ ô viền vàng:
  - từ đúng chọn khung 2, 3, 4, 6 → 0.85;
  - từ nhiễu chọn khung 5, 0, 7, 1 (lộn xộn) → **0.75**, gần bằng.
- Vị trí: trung vị các khung chọn. Với nhiễu, trung vị rơi vào giữa hai cụm, chỗ không có gì.

**Trang 18 – Khớp đúng thứ tự (DP).**
- Đọc công thức: "D[p,t] là đường tốt nhất qua âm vị 1 tới p, kết thúc ở khung t. Âm vị trước phải ở trước 1 tới G khung."
- Tính tay 3 ô xanh:
  - 1.7 = .8 + max(.9, .1);
  - 2.6 = .9 + max(1.7, .5);
  - 3.4 = .8 + max(2.0, 2.6).
- Kết quả: 3.4/4 = **0.85**.
- "Từ đúng thì tự do và thứ tự bằng nhau, vì ô sáng đã theo thứ tự."

**Trang 19 – Từ nhiễu + lần ngược.**
- Đường vàng đi qua ô sáng tới 1.8 ở khung 7, nhưng âm vị 4 phải ở sau khung 7 → hết chỗ → tốt nhất chỉ 0.8/4 = **0.20**.
- Lần ngược từ đúng: 3.4 ở khung 6 ← 2.6 ở khung 4 ← 1.7 ở khung 3 ← 0.9 ở khung 2 → từ ở khung 2–6.

**Trang 20 – Ra p.**
- logit = a·thứ tự + a2·tự do + c.
- Với giá trị khởi tạo (nói rõ là minh hoạ):
  - chỉ tự do: nhiễu p = 0.92, **lọt** ngưỡng 0.5;
  - hai nhánh: p = 0.38, **bị loại**.
- Luôn có tự do ≥ thứ tự. Chỉ dùng thứ tự thì sót, vì một âm vị khớp kém kéo cả đường xuống.

**Trang 21 – Train.**
- Mỗi câu: từ hiếm của câu là đúng, thêm 4 lần số đó từ nhiễu (40% gần âm). Loss BCE.
- Ví dụ: p = .9 → loss 0.11; nhiễu p = .38 → 0.48.
- Dạy vị trí: ngoài đoạn MFA ±0.1 s, bảng của từ đúng bị đặt −1.

**Trang 22 – Đo thật hai cách.**
- Thứ tự ít nhầm hơn: precision 75.8 → 82.3.
- Nhưng sót hơn: Recall@1 87.5 → 85.8, khoảng tin cậy −1.7 [−2.7; −0.7].
- Đọc cột "Vì sao" (đây là diễn giải).
- Ô dưới: cộng 2 model khác cách được +2.6; cùng cách chỉ +1.1 → hai cách bù nhau thật → gộp thành một model.

**Trang 23 – Vị trí đúng tới đâu.** Tâm lệch 0.02 s, 99.1% trúng; Hou25 gốc 44%. Chỉ đo trên từ đúng.

**Trang 24 – Từng bước.**
- Dạy vị trí +0.8.
- Thêm nhánh, chung bảng: Recall@1 đứng yên, nhưng nhóm nghe sai +2.9.
- Tách hai bảng +1.0.
- Tổng: **89.3**; nhóm Whisper nghe sai **70.9**.
- Đọc dòng đỏ: lệch có lợi cho nhóm.

**Trang 25 – Câu dài.** Hou25 12.4 nhiễu/câu, đề xuất 0.9.

**Trang 26 – Lọc list.** Đọc phép tính bên trái:
- Từ đúng còn lại: 92.4% × 5248 = 4849, tức 1.65 mỗi câu.
- Từ nhiễu còn lại: 0.27 × 2939 = 794.
- Tổng 5643 / 2939 = **1.92 từ/câu**. Câu thật có 1.79 từ hiếm, nên list sau lọc gần bằng danh sách thật. Precision 85.9%.
- Hou25: 1.30 đúng + 1.06 nhiễu. Bước "trùng chỗ" làm Hou25 mất 14 điểm, vì vị trí của nó sai.

**Trang 27 – Chỗ yếu.** Nhiễu nghe như một từ thường (flauve ≈ "fall") thì không có từ hiếm nào tranh chỗ, nên nó lọt.

## Whisper (trang 28–31, ~4 phút)

**Trang 28 – Cách nhắc đúng lúc.** Alignment heads cho biết giây đang nghe. Từ nằm ở giây đó thì được cộng λ = 5, mỗi từ một lần.

**Trang 29 – Kết quả.**
- B-WER 9.69 → **7.65**, test-clean 5.15 → 3.83.
- **Nói rõ:** WER tổng −0.17, chưa có ý nghĩa.

**Trang 30 – Ba điều phía ASR (D2).** Nhắc đúng lúc 3.22 so với nhắc mọi lúc 3.66: cần vị trí.

**Trang 31 – Theo cỡ list.** Đọc bảng:
- Từ N100 lên N2000, nhiễu lọt qua tăng 0.04 → 0.73 từ/câu.
- Nhưng B-WER chỉ 7.14 → 7.65, WER 2.96 → 3.22.
- "Mỗi câu thêm 0.7 từ nhiễu được nhắc, mà phần lớn Whisper không viết ra, vì nhiễu chỉ được nhắc ở đoạn của nó và audio không ủng hộ."
- Lưu ý: precision 70% ở đây là ngưỡng 0.3, khác 86% ở trang 26 (ngưỡng 0.5).
- Nói thật: chưa có Hou25 ở N100–1000.

## Đúc kết (trang 32, ~1 phút)

**Trang 32.** Đọc ba ô từ trái sang phải:
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

Trang **2** → **6** → **8** → **10** → **11** → **13** → **16** → **17** → **18** → **19** → **20** → **24** → **26** → **29** → **32**.

## Sau buổi họp

- Chép ý kiến thầy vào `meetings/2026-11-20.md`: quyết định, việc giao, hạn.
- Nếu việc tiếp theo đổi thì cập nhật `GOAL.md` mục 6, và ghi `log/2026-11-20.md`.
