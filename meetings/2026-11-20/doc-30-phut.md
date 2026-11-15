# Đọc 30 phút buổi sáng: scorer có vị trí, mình đã làm gì và đi đâu tiếp

> Dành cho buổi warm-up sáng. Đọc từ trên xuống, theo đồng hồ ở mỗi chặng.
> Không cần mở file nào khác. Muốn xem bảng đầy đủ thì cuối file có chỉ chỗ.

| Chặng | Nội dung | Thời gian |
|---|---|---|
| 1 | Bài toán, kể bằng ví dụ đời thường | 5 phút |
| 2 | Mình đặt ra câu hỏi gì, và vì sao phải đo lại | 5 phút |
| 3 | Đã làm gì, thấy gì (phía scorer) | 10 phút |
| 4 | Đưa vào Whisper thì sao | 4 phút |
| 5 | Chỗ còn yếu | 3 phút |
| 6 | Bước tiếp theo + 5 câu tự kiểm | 3 phút |

---

## Chặng 1. Bài toán (5 phút)

### Ví dụ đời thường

Hình dung một buổi họp được ghi âm. Thư ký (chính là **Whisper**) phải chép lại. Thư ký nghe tốt từ thường, nhưng hay viết sai **tên riêng, từ hiếm**: "Pileser" thành "pillar sir".

Để giúp thư ký, người ta đưa một **danh bạ 2000 tên**. Vấn đề: trong buổi họp chỉ có **1–3 tên** thật sự được nhắc. 1997 tên còn lại là **nhiễu**.

Nếu đưa nguyên 2000 tên, thư ký bị rối và bắt đầu "nghe ra" những tên không hề có. Nên cần một người trợ lý (chính là **scorer**) nghe trước, rồi chọn ra vài tên đáng tin.

### Hou25 (bài mốc ta phải thắng) làm thế nào

Trợ lý của Hou25:
1. Đọc to từng tên bằng giọng máy (TTS).
2. So giọng máy với đoạn ghi âm, ra một **bản đồ giống nhau** (thời gian câu × thời gian tên).
3. **Lấy trung bình cả bản đồ** thành một điểm.
4. Chọn **50 tên** điểm cao nhất, đưa cho thư ký dưới dạng **prompt** (ghi chú đầu trang).

Chỗ đáng tiếc: ở bước 2, bản đồ thật ra **biết** tên được nói ở giây nào (Hoàng thấy có một vệt chéo đúng chỗ đó). Nhưng bước 3 và 4 vứt thông tin này đi. Thư ký nhận 50 tên mà không biết tên nào ở đâu, và 48 trong số đó là nhiễu.

### Ý tưởng của mình, một câu

> **Giữ lại "tên này nằm ở giây nào".** Có vị trí thì trợ lý chọn tên chuẩn hơn, và ta nhắc thư ký **đúng lúc** họ đang nghe đoạn đó, không cần ghi chú đầu trang.

Ví dụ: trợ lý nói "`pileser` ở giây 3.0–3.6". Khi thư ký đang chép tới giây 3.2, ta khẽ nhắc "có thể là pileser". Ở giây 1 hay giây 5 thì im lặng.

---

## Chặng 2. Mình đặt ra câu hỏi gì (5 phút)

### Câu hỏi lớn

**Giữ lại vị trí thì có lợi gì không, ở phía trợ lý (scorer) và phía thư ký (Whisper)?**

Trước khi chạy, mình đã đặt sẵn **3 quyết định** cần ra, kèm luật quyết định viết trước (để không "thấy kết quả rồi mới chọn luật"):

| Quyết định | Luật đặt trước |
|---|---|
| D1. Trợ lý nên xây trên nền nào: sửa Hou25, hay scorer âm vị của nhóm? | Hou25 sửa xong phải đuổi kịp âm vị thì mới giữ |
| D2. Thư ký có thật sự cần vị trí không? | Cùng danh sách tên, nhắc đúng lúc phải tốt hơn nhắc mọi lúc, có ý nghĩa thống kê |
| D3. Nhắc thế nào cho hợp lý? | Chọn trên tập dev, không chọn trên test |

### Vì sao phải đo lại, không dùng thước của Hou25

Hou25 chấm trợ lý bằng câu hỏi: **"tên đúng có lọt top-50 không?"**. Giống như chấm thi trắc nghiệm mà cho học sinh khoanh 50 đáp án: ai cũng đạt 95–98%, nhìn như ngang nhau.

Mình đổi sang **Sạch@1**: tên đúng có **đứng đầu**, không bị tên nhiễu nào vượt mặt không. Thước này khó hơn nhiều, và nó mới đúng với cách mình dùng: nếu chỉ đưa 2–3 tên cho thư ký thì tên đúng phải nằm ở đầu.

Đo lại bằng Sạch@1, kết quả đầu tiên đã khác hẳn:

| | Top-50 (thước cũ) | **Sạch@1** (thước mới) |
|---|---|---|
| Hou25 | 95.4 | **72.8** |
| Scorer âm vị của nhóm | 98.4 | **85.7** |

Thước cũ: cách nhau 3 điểm. Thước mới: cách nhau **13 điểm**.

Ngoài Sạch@1, mình còn đo:
- **Định vị:** trợ lý đoán vị trí lệch bao nhiêu giây so với vị trí thật.
- **Nhóm từ Whisper nghe sai:** chỉ tính những từ mà thư ký tự viết sai. Đây là nhóm **cần** trợ lý nhất; từ thư ký đã nghe đúng thì nhắc cũng vô ích.
- **B-WER:** tỉ lệ lỗi chỉ tính trên từ hiếm. **WER** là tỉ lệ lỗi trên mọi từ.

---

## Chặng 3. Đã làm gì, thấy gì: phía scorer (10 phút)

Kể theo thứ tự câu hỏi nảy ra. Mỗi phát hiện có 4 bước: **mình nghĩ gì → thử gì → thấy gì → nghĩa là gì.**

### Phát hiện 1. Hou25 sai không phải vì "tên đúng bị chìm", mà vì "tên nhiễu nổi lên"

- **Nghĩ:** trước đây nhóm tin rằng lấy trung bình cả bản đồ làm tên đúng bị pha loãng, điểm thấp đi.
- **Thử:** xem kỹ những lần tên đúng bị vượt (CPU, checkpoint cũ).
- **Thấy:**
  - 96% số lần, tên vượt lên là tên **khác hẳn**, không phải tên nghe giống.
  - Số tên nhiễu điểm cao tăng theo độ dài câu: câu dưới 5 giây có 1.4 tên nhiễu mỗi câu, câu trên 20 giây có hơn 11.
- **Nghĩa là:** câu càng dài, bản đồ càng có nhiều mảng ngẫu nhiên "trông giống vệt", và trung bình cả bản đồ bị đánh lừa. Vấn đề là **nhiễu bị đẩy lên**, ngược với giả thuyết cũ.

Kèm một lỗi kỹ thuật nhặt được trên đường: encoder Whisper chỉ nhìn 30 giây đầu, nên từ nào nói sau giây 30 bị mất ở **mọi** hệ. Mình sửa bằng cửa sổ trượt (cắt câu dài thành nhiều đoạn 30 giây chồng nhau). Trước khi sửa, 5–6 trên 6 từ như vậy bị sót; sau khi sửa: 0 trên 6.

### Phát hiện 2. Giả thuyết của Hoàng đúng: đọc theo vệt chéo

- **Nghĩ (Hoàng):** thay vì trung bình cả bản đồ, chỉ đọc **vệt chéo** mạnh nhất, tức là đoạn mà âm của tên khớp với audio theo đúng trình tự.
- **Thử:** giữ nguyên Hou25, chỉ đổi bước gộp, **train lại**. Có thêm một bản đối chứng: lấy các ô điểm cao nhất mà không cần chúng thành vệt.
- **Thấy:**

| Cách gộp | Sạch@1 | Định vị trúng |
|---|---|---|
| Trung bình (Hou25 gốc) | 73.3 | 79% |
| Ô cao nhất, không cần thành vệt | 73.8 (không đổi) | 4% |
| **Vệt chéo** | **82.9** (+9.5) | **99%** |

- **Nghĩa là:** thứ có ích là **hình dạng vệt** (đúng trình tự, liền nhau), không phải cứ "đọc chỗ cao nhất". Và cùng một model, giờ biết luôn tên ở giây nào.

Lần trước (09/11) chỉ đổi cách gộp lúc chấm mà không train lại thì không thấy lợi. Lần này train lại thì thấy. Bài học: model phải được dạy cho cách đọc mới.

### Phát hiện 3. Nhưng nền tốt nhất vẫn là âm vị, không phải giọng máy

- **Nghĩ:** Hou25 dùng giọng máy (TTS), có thể phân biệt tên nghe giống tốt hơn so với cách chuyển tên thành **chuỗi âm vị** (kiểu phiên âm: `pileser` → P-AY-L-IY-Z-ER) như scorer của nhóm.
- **Thử:** một danh sách đặc biệt nhiều tên nghe giống.
- **Thấy:** âm vị vẫn hơn giọng máy 9.7 điểm. Và Hou25 sửa tốt nhất (82.9) vẫn thua âm vị (87.5), lại chậm hơn khoảng 200 lần (1.2 giây so với 5 mili giây mỗi câu).
- **Nghĩa là: chốt D1 = nền âm vị.** Hou25 + vệt chéo sẽ là "đối thủ mạnh" trong paper.

Một thử nghiệm thất bại cũng ghi lại: cho model thấy nhiều tên nhiễu hơn lúc train (16 thay vì 4) làm kết quả **tệ đi** (82.1 so với 87.7). Bỏ.

### Phát hiện 4. Hai cách khớp bù nhau, nên làm trợ lý hai nhánh

Có hai cách cho âm vị khớp với audio:
- **Khớp tự do** (`pmaxsim`): mỗi âm vị tự đi tìm chỗ giống nhất, ở đâu cũng được. Ưu: ít bỏ sót. Nhược: dễ ghép bừa.
- **Khớp đúng trình tự** (`pmono`): các âm vị phải khớp theo thứ tự, liền nhau, như một vệt. Ưu: ít báo nhầm, ra vị trí rất chuẩn. Nhược: xếp hạng kém hơn chút (85.8 so với 87.5).

- **Thử:** cộng điểm hai cách. Đối chứng: cộng hai model **cùng** cách (vì gộp hai model nào cũng hay hơn một chút).
- **Thấy:** cộng hai model cùng cách: +1.1. Cộng hai cách khác nhau: **+2.6**. Phần chênh +1.5 là do chúng nhìn khác nhau thật. Lợi này ngang với việc train thêm gấp 9.6 lần dữ liệu.
- **Nghĩa là:** thiết kế trợ lý gồm **hai nhánh trong một model**: nhánh tự do để không sót, nhánh đúng trình tự để loại nhiễu và cho vị trí.

### Phát hiện 5. Dạy vị trí lúc train giúp thêm

- **Thử:** lúc train, tên đúng chỉ được tính điểm **ở đúng đoạn nó được nói**. Đoạn đó lấy từ công cụ căn chỉnh tự động MFA.
- **Thấy:** nhánh đúng trình tự tăng từ 85.8 lên 88.3. Tăng mạnh nhất ở nhóm từ Whisper nghe sai (64 lên 69).

**Ghép lại, scorer đơn đề xuất** (`pdual2_dl`, một model, hai nhánh, dạy vị trí):

| | Sạch@1 | Nhóm từ Whisper nghe sai | Định vị |
|---|---|---|---|
| Hou25 (bản nhóm tái hiện) | 72.6 | 46 | không tin được |
| Âm vị một nhánh | 87.5 | 64.5 | 0.05 giây |
| **Hai nhánh + dạy vị trí (một model)** | **89.3** | **70.9** | **0.02 giây** |
| Cộng điểm hai model riêng | 90.8 | – | 0.02 giây |

### Phát hiện 6. Có vị trí thì lọc được danh sách

- **Nghĩ:** hai tên không thể cùng được nói ở một chỗ. Nếu hai tên tranh cùng một đoạn audio thì chỉ giữ tên điểm cao hơn.
- **Thấy:** từ 2000 tên, mỗi câu chỉ còn **khoảng 2 tên**, đúng **91%**, và vẫn giữ 91% tên đúng. So với Hou25: 50 tên, đúng 3–4%.
- **Nghĩa là:** vị trí biến bài "tìm 1 trong 2000" thành nhiều bài nhỏ "tìm 1 trong vài tên cùng chỗ".

Có một chỗ luật này chưa xử lý được. Ví dụ câu có từ thường "fall": tên nhiễu `flauve` nghe giống "fall", nên được điểm cao ở đúng chỗ đó. Không có tên hiếm nào tranh chỗ với nó, nên nó lọt qua. Trợ lý cần biết nói "ở đây không có tên nào cả" (chặng 6, việc số 1).

---

## Chặng 4. Đưa vào Whisper thì sao (4 phút)

### Cách nhắc

Mỗi khi Whisper viết một chữ, mình đọc một số "đầu chú ý" bên trong nó để biết nó **đang nghe giây nào**. Tên nào có đoạn chứa giây đó thì được cộng thêm điểm (λ = 5). Mỗi tên chỉ nhắc một lần. Nếu Whisper bắt đầu lặp chữ thì decode lại câu đó không nhắc.

### Kết quả chính (test-other, danh sách 2000 tên)

| | WER (mọi từ) | B-WER (từ hiếm) |
|---|---|---|
| Không giúp gì | 4.25 | 17.92 |
| **Hou25: đưa 50 tên vào prompt** | 3.39 | 9.69 |
| Scorer mới, đưa 50 tên vào prompt | 3.43 | 10.03 |
| Scorer mới, nhắc **mọi lúc** | 3.56 | 7.17 |
| **Scorer mới, nhắc đúng lúc, không prompt** | **3.14** | **7.53** |
| Mức trần (biết đúng tên + đúng chỗ) | 2.99 | 5.99 |

### Đọc bảng này thế nào

1. **Thắng Hou25 ở từ hiếm:** B-WER 9.69 xuống 7.53, không cần prompt, có ý nghĩa thống kê. test-clean cũng vậy (5.15 xuống 3.93).
2. **Whisper cần vị trí (chốt D2):** cùng danh sách, nhắc đúng lúc tốt hơn nhắc mọi lúc 0.42 WER. Nhắc mọi lúc thì Whisper chèn tên vào những chỗ không có, làm hỏng từ thường.
3. **Prompt đã "đầy":** scorer tốt hơn Hou25 17 điểm, nhưng đưa 50 tên vào prompt thì vẫn chỉ ngang Hou25 (10.03 so với 9.69). Nghĩa là trợ lý giỏi hơn mà đi qua cửa prompt thì không ích gì. **Muốn hưởng lợi phải đi qua cửa "vị trí + nhắc đúng lúc".**
4. **Còn cách trần khoảng 1.5 điểm B-WER.** Một nửa do vị trí đoán lệch, một nửa do danh sách còn sót hoặc lẫn nhiễu.

Một chi tiết thú vị (đo trên CPU): Whisper **không** hay viết tên vào sai chỗ, vì nó nghe audio. Lỗi khi thiếu vị trí là kiểu khác: nó **viết thêm** tên ở chỗ không có. Vậy vị trí cần để biết **lúc nào im lặng**, hơn là để biết đặt tên ở đâu.

---

## Chặng 5. Chỗ còn yếu (3 phút)

Phải nói thật với thầy, và cũng là việc phải làm tiếp:

1. **WER tổng chưa thắng chắc.** Không prompt chỉ thấp hơn Hou25 0.25, chưa có ý nghĩa thống kê. Nên khi nói "thắng Hou25" luôn phải kèm chữ **B-WER**.
2. **Audio không có tên nào thì vẫn bị hại.** Đưa danh sách toàn tên nhiễu: WER tăng từ 4.25 lên 4.49. Trợ lý chưa đủ "dám nói không".
3. **Một model còn kém bản cộng điểm 1.5** (89.3 so với 90.8).
4. **Hou25 là bản nhóm tự dựng lại** (2 epoch), chưa phải checkpoint của Hoàng. Có thể bị chê "đối thủ yếu".
5. **Mới chỉ có Whisper.** Chưa thử Qwen3-ASR, nên chưa nói được "dùng cho mọi ASR". Cách nhắc hiện tại dựa vào "đầu chú ý" riêng của Whisper.

---

## Chặng 6. Bước tiếp theo (3 phút)

### Hướng đi, một câu

> Trợ lý trả ra **(tên, độ tin, vị trí)**, rồi nhắc ASR đúng lúc trong khi decode. Không dựa vào prompt.

### Việc cụ thể, theo thứ tự

| # | Việc | Vì sao | Xong khi |
|---|---|---|---|
| 1 | Dạy trợ lý nói **"chỗ này không có tên nào"** | Chỗ yếu số 2 và ví dụ `flauve` / "fall" | Danh sách toàn nhiễu gần như không làm WER tệ đi |
| 2 | Nâng **nhóm từ Whisper nghe sai** (đang 70.9) | Nhóm quyết định B-WER | Tăng khoảng 3 điểm |
| 3 | Một model đuổi kịp bản cộng điểm | 89.3 so với 90.8 | ≥ 90.5 |
| 4 | Train trên 960 giờ thay vì 100 giờ | Lần trước tăng 1.8 điểm | Bản cuối cho paper |
| 5 | Đủ bảng chính: danh sách 100 / 500 / 1000 / 2000; Hou25 của Hoàng | Bảng của paper | Đủ ô |
| 6 | Thử **Qwen3-ASR**: tìm cách biết nó đang nghe giây nào mà không cần "đầu chú ý" | Để nói được "dùng cho mọi ASR" | Có một dòng Qwen3-ASR |
| 7 | Khảo sát các cách chỉnh logit khác (ý của thầy) | Chọn đối thủ so sánh | Bảng so sánh |

**Mốc:** 20/11 gặp thầy · 20/12 scorer cuối · 17/01 đủ bảng · 05/02 bản nộp · 09/02 hạn Interspeech.

### Ba câu sẽ hỏi thầy

1. Đóng góp chính đặt ở **trợ lý hai nhánh**, ở **cách nhắc đúng lúc**, hay gộp cả hai thành một hệ?
2. Chỉ so với Hou25, hay thêm PTC-Bias và cách nhắc không theo vị trí?
3. Whisper + Qwen3-ASR có đủ cho Interspeech không?

---

## 5 câu tự kiểm (đọc xong thử trả lời, đáp án ngay dưới)

1. Vì sao không dùng "top-50" để chấm trợ lý?
2. Hou25 sai chủ yếu vì tên đúng bị chìm hay vì tên nhiễu nổi lên?
3. Vì sao cần hai nhánh, không dùng một nhánh?
4. Bằng chứng nào cho thấy Whisper cần vị trí?
5. Nếu thầy hỏi "vậy thắng Hou25 chưa?", trả lời thế nào cho đúng?

<details>
<summary>Đáp án gợi ý</summary>

1. Ở top-50 hệ nào cũng 95–98%, khác biệt bị che. Mình chỉ đưa 2–3 tên cho ASR nên tên đúng phải đứng đầu: dùng Sạch@1 (Hou25 72.8, âm vị 85.7).
2. Tên nhiễu nổi lên, và càng nhiều khi câu càng dài (1.4 lên hơn 11 tên mỗi câu).
3. Khớp tự do ít sót, khớp đúng trình tự ít nhiễu và cho vị trí chuẩn. Gộp hai cách: +2.6, gộp hai model cùng cách chỉ +1.1.
4. Cùng danh sách tên, nhắc đúng lúc tốt hơn nhắc mọi lúc 0.42 WER, có ý nghĩa thống kê.
5. "Thắng về B-WER, tức lỗi ở từ hiếm (9.69 xuống 7.53, có ý nghĩa), mà không cần prompt. WER tổng mới ngang, chưa có ý nghĩa thống kê."

</details>

---

**Muốn đào sâu sau buổi sáng:**
- Giải thích từng thí nghiệm theo khung đầy đủ: `scorer/analysis/TIEN-DO-1115.md`.
- Mọi bảng số và khoảng tin cậy: `scorer/analysis/REPORT-1115.md`.
- Slide và kịch bản nói: `meetings/2026-11-20/hop-thay.pdf`, `kich-ban.md`.
