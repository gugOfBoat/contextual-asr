# Chương 8 — Đo lường và thiết kế thí nghiệm

> **Mục tiêu chương:** biết đo **đúng**. Trong contextual ASR, đo sai là cách phổ biến nhất để tự lừa mình — và cũng là chỗ reviewer bắt lỗi nhiều nhất. Chương này cho định nghĩa chính xác (đối chiếu bản gốc), ví dụ tính tay, và checklist thí nghiệm.

---

## 8.1 WER: nền tảng, và ba cái bẫy

$$\text{WER} = \frac{S + D + I}{N}$$

$S$ = thay thế, $D$ = xoá, $I$ = chèn, $N$ = số từ trong **reference**. Tính bằng **căn chỉnh Levenshtein** (quy hoạch động tìm đường sửa lỗi rẻ nhất).

**Bẫy 1 — WER có thể > 100%.** Vì $I$ không bị chặn bởi $N$. Hay gặp khi model hallucinate.

**Bẫy 2 — căn chỉnh không duy nhất.** Nhiều đường sửa lỗi cùng chi phí; công cụ khác nhau chọn đường khác nhau → phân bố $S/D/I$ khác nhau (dù tổng như nhau). Điều này **ảnh hưởng trực tiếp tới B-WER/U-WER** vì hai metric đó phân loại lỗi theo vị trí. Nên: **dùng một công cụ duy nhất cho toàn bộ luận văn**, và ghi rõ tên + phiên bản.

**Bẫy 3 — chuẩn hoá văn bản.** Đã nêu ở báo cáo `00` §7.4: cùng một model có thể cho O-WER 32% và N-WER 6.9%. Với tiếng Việt phải quyết định và ghi rõ:
- Unicode **NFC** (bắt buộc — chương 4 §4.3b).
- Hạ chữ thường? Bỏ dấu câu? Chuẩn hoá số? Gạch nối (`vắc-xin` vs `vắc xin`)?
- Đơn vị đếm: **âm tiết** (quy ước phổ biến cho tiếng Việt) — nói rõ.

> **Quy tắc:** áp **cùng một** hàm chuẩn hoá cho reference, hypothesis **và bias list**. Nếu bias list không được chuẩn hoá giống hệt, B-WER sẽ sai một cách âm thầm.

---

## 8.2 ★ B-WER và U-WER — định nghĩa gốc

*Le et al., Interspeech 2021 (arXiv 2104.02194). Trích nguyên văn từ bản full-text:*

> *"U-WER is measured on words **not in** the biasing list, B-WER on words **in** the biasing list. **Insertion errors are counted toward B-WER if the inserted word is in the biasing list and U-WER otherwise.**"*

Viết thành công thức. Gọi $\mathcal{B}$ = tập từ trong bias list:

$$\text{B-WER} = \frac{S_b + D_b + I_b}{N_b}, \qquad \text{U-WER} = \frac{S_u + D_u + I_u}{N_u}$$

- $N_b$ = số từ trong **reference** thuộc $\mathcal{B}$; $N_u = N - N_b$.
- $S, D$: phân loại theo **từ trong reference** (từ bị thay/bị xoá thuộc $\mathcal{B}$ hay không).
- $I$: phân loại theo **từ trong hypothesis** (từ bị chèn thuộc $\mathcal{B}$ hay không).

**Vì sao quy ước insertion như vậy là hợp lý:** over-biasing biểu hiện chính là **chèn thêm** từ trong bias list vào chỗ không có. Quy ước này đảm bảo hành vi đó **bị tính vào B-WER**, tức phương pháp biasing không thể "ăn gian" bằng cách rải thực thể khắp nơi.

Và ngược lại, một **thay thế** kiểu ref `thuốc` (không thuộc $\mathcal{B}$) → hyp `paracetamol` (thuộc $\mathcal{B}$) bị tính vào **U-WER** (vì từ ở reference không thuộc $\mathcal{B}$). Đây chính là kênh mà over-biasing làm U-WER xấu đi.

### Ví dụ tính tay #1 — biasing thất bại

Bias list: `{paracetamol, amoxicillin}`

| | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **REF** | bệnh | nhân | dùng | **paracetamol** | mỗi | ngày |
| **HYP** | bệnh | nhân | dùng | pa · ra · xê · ta · môn | mỗi | ngày |

Căn chỉnh: `paracetamol` → `pa` là **thay thế**; `ra xê ta môn` là **4 chèn**.

- $N = 6$, $N_b = 1$ (`paracetamol`), $N_u = 5$.
- $S_b = 1$; $I_b = 0$ (các từ chèn `ra, xê, ta, môn` **không** nằm trong bias list) → $I_u = 4$.

$$\text{WER} = \frac{1+4}{6} = 83.3\%, \quad \text{B-WER} = \frac{1}{1} = 100\%, \quad \text{U-WER} = \frac{4}{5} = 80\%$$

**Bài học từ ví dụ này:** một thực thể sai làm bẩn cả U-WER (qua các chèn). Nên khi bạn cải thiện biasing, **U-WER cũng có thể giảm** — đừng ngạc nhiên. Và ngược lại: U-WER tăng không phải lúc nào cũng là over-biasing, phải xem loại lỗi.

### Ví dụ tính tay #2 — over-biasing

Bias list: `{paracetamol}`

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **REF** | cho | tôi | xin | thuốc |
| **HYP** | cho | tôi | xin | **paracetamol** |

- $N_b = 0$ (reference không chứa từ nào trong bias list) → **B-WER không xác định** cho câu này (chuẩn: cộng dồn tử/mẫu trên toàn tập rồi mới chia, đừng lấy trung bình theo câu).
- $S_u = 1$ → U-WER = 1/4 = 25%.

⇒ Over-biasing hiện ra ở **U-WER**. Đó là lý do **bắt buộc phải báo cáo cả hai**. Một paper chỉ báo B-WER là paper đã giấu đi một nửa sự thật.

### Cạm bẫy khi cài đặt B-WER

1. **Cộng dồn toàn tập**, không lấy trung bình WER theo câu (macro-average làm câu ngắn có trọng số quá lớn).
2. **Khớp cụm nhiều từ**: bias list thường chứa cụm (`"Nguyễn Thị Hồng"`). Định nghĩa gốc làm việc ở **mức từ** — thực tế người ta coi mọi từ trong cụm là "biased word". Ghi rõ bạn làm thế nào.
3. **Chuẩn hoá bias list y hệt reference** (NFC, chữ thường, dấu câu).
4. **Bias list phải là bias list thật sự dùng lúc decode**, gồm cả distractor — không phải chỉ danh sách thực thể có trong câu.

Script tham chiếu: [`tools/bwer.py`](tools/bwer.py).

---

## 8.3 Metric mức thực thể

WER trừng phạt theo từ; nhưng ứng dụng thật quan tâm *"có bắt đúng tên thuốc không"*. Nên bổ sung:

- **Entity Recall** = (số thực thể trong reference được chép **đúng hoàn toàn**) / (tổng thực thể trong reference).
- **Entity Precision** = (số thực thể model sinh ra mà đúng) / (tổng thực thể model sinh ra) — **đây là metric bắt over-biasing sắc bén nhất**.
- **Entity F1** = trung bình điều hoà.
- Biến thể "khớp một phần" (partial match) khi bạn muốn phân biệt sai hoàn toàn vs sai một âm tiết.

> Với tiếng Việt, nên báo cáo thêm **phân loại lỗi ở thực thể**: sai thanh điệu / sai nguyên âm–phụ âm / sai hoàn toàn. Một hệ thống sai `paracetamol` → `paracetamon` khác hẳn về mặt sử dụng với hệ thống sai thành `pa ra xê ta môn`, nhưng WER không phân biệt được. Đây là loại phân tích làm luận văn có chiều sâu.

---

## 8.4 ★ Đo over-biasing cho tử tế

Ba mức, tăng dần độ nghiêm ngặt:

**Mức 1 — báo cáo U-WER.** Tối thiểu bắt buộc.

**Mức 2 — đường cong đánh đổi.** Quét trọng số biasing $\alpha \in \{0, 0.5, 1, 2, 4, 8\}$, vẽ:

```
B-WER
  │  α=0
  │   ●
  │     ╲
  │       ●  α=1
  │         ╲
  │           ● α=2
  │             ╲___● α=4
  │                   ●  α=8   ← B-WER hết giảm, U-WER vẫn tăng
  └────────────────────────────► U-WER
```

Một phương pháp tốt hơn = **cả đường cong nằm dưới-trái** đường của baseline, chứ không phải một điểm may mắn. Đây là cách trình bày thuyết phục nhất và ít ai làm — làm được là điểm cộng rõ trong luận văn.

**Mức 3 — tập kiểm thử đối kháng (anti-context / distractor-only).** Lấy các câu **không chứa** thực thể nào, nhưng vẫn đưa bias list đầy đủ (thậm chí toàn cụm phát âm gần giống nội dung câu). Đo WER. Nếu WER tăng nhiều → hệ thống của bạn nguy hiểm trong thực tế. **Metric cho việc này còn thiếu chuẩn hoá trong cộng đồng** (báo cáo `01` §3.6) — nên đây cũng là chỗ có thể đóng góp.

---

## 8.5 Đo tầng retrieval riêng (nếu bạn dùng họ 6)

- **Recall@k**: tỉ lệ thực thể đúng nằm trong top-$k$. Đây là **trần trên** của toàn hệ thống.
- **MRR / Precision@k**: chất lượng xếp hạng.
- **Đo phân rã hai tầng**: chạy thêm hai cấu hình
  1. **Oracle retrieval**: đưa thẳng thực thể đúng + vài distractor → cho biết module biasing tốt tới đâu nếu retriever hoàn hảo.
  2. **Retrieval thật**: cấu hình đầy đủ.

  Khoảng cách giữa (1) và (2) = **phần lỗi thuộc về retriever**. Không có phân rã này thì không biết nên cải thiện tầng nào — và đây là lỗi thiết kế thí nghiệm phổ biến.

---

## 8.6 Các thí nghiệm chẩn đoán bắt buộc

Đây là bộ "xét nghiệm" nên chạy sớm, trước khi đầu tư vào một phương pháp.

| Thí nghiệm | Cách làm | Cho biết điều gì |
|---|---|---|
| **No-context** | tắt hẳn ngữ cảnh | cận dưới |
| **Oracle context** | bias list = **chỉ** các thực thể đúng của câu đó | **cận trên** của mọi phương pháp biasing |
| **Adversarial context** | bias list = toàn cụm sai/đồng âm | đo tính an toàn |
| **Oracle n-best** | WER nếu luôn chọn giả thuyết tốt nhất trong n-best | trần trên của rescoring & post-hoc |
| **Ablation tokenizer** | in token của bias list vs reference | phát hiện lệch khoá tra cứu (chương 4 §4.4) |
| **Phonetic form** | bias list ghi theo cách đọc thay vì chính tả | kiểm chứng giả thuyết lệch chính tả–phát âm |

> **Về "context-utilization gap"**: cặp thí nghiệm *no-context* vs *oracle context* chính là cách đo nó. Nếu cho **oracle context** mà WER gần như không đổi → model **không dùng** ngữ cảnh, và mọi cải tiến cơ chế biasing sẽ vô ích cho tới khi bạn sửa được điều đó. ProfASR-Bench (báo cáo `01` §1.5) đã đo và thấy đúng hiện tượng này với Whisper và Qwen-Omni ⚠️. **Chạy thí nghiệm này đầu tiên** — nó rẻ và có thể tiết kiệm cho bạn vài tháng đi sai hướng.

---

## 8.7 Metric hiệu năng (đừng bỏ qua — reviewer sẽ hỏi)

- **RTF / RTFx**: thời gian xử lý / thời lượng audio (RTFx = nghịch đảo, càng cao càng nhanh).
- **Latency**: với streaming, đo cả trễ thuật toán và trễ tính toán; báo cáo p50/p90.
- **Chi phí theo $N$**: vẽ đồ thị RTF và bộ nhớ theo $N \in \{100, 1k, 10k, 100k\}$. Đây là chỗ các phương pháp neural lộ điểm yếu và các phương pháp ký hiệu (TurboBias) toả sáng.

---

## 8.8 Kiểm định thống kê

Chênh 0.2 điểm WER trên tập 2.000 câu **có thể là nhiễu**. Hai công cụ tối thiểu:

- **Bootstrap resampling**: lấy mẫu lại tập test có hoàn lại 1.000 lần, tính WER mỗi lần → khoảng tin cậy 95%. Đơn giản, không giả định gì, nên dùng mặc định.
- **Matched-pairs test (MAPSSWE)**: so từng câu giữa hai hệ thống — mạnh hơn vì loại được phương sai do độ khó câu.

Trong luận văn, báo cáo khoảng tin cậy cho ít nhất metric chính. Đây là dấu hiệu của công việc nghiêm túc.

---

## 8.9 Dựng benchmark biasing (nếu bạn làm dataset tiếng Việt)

Quy trình chuẩn de-facto là LibriSpeech-biasing của Le et al. (báo cáo `01` §2.1): **5.000 từ tần suất cao nhất = "common"**, **209.200 từ còn lại = "rare"**; với mỗi câu test, lấy tất cả từ rare trong reference làm positive rồi thêm **distractor lấy ngẫu nhiên từ kho rare** cho đủ $N \in \{100, 500, 1000, 2000\}$.

Nếu chuyển thể sang tiếng Việt, phải quyết định thêm 6 điều — và **chính những quyết định này là đóng góp khoa học**, không phải chi tiết kỹ thuật:

1. **Đơn vị "từ"**: âm tiết hay từ ghép? Thực thể tiếng Việt thường nhiều âm tiết (`Nguyễn Thị Hồng Nhung`) → nên định nghĩa bias list ở **mức cụm**, và nói rõ cách quy về mức từ khi tính B-WER.
2. **Ngưỡng "rare"**: 5.000 từ phổ biến là quy ước tiếng Anh. Tiếng Việt có ~7.000 âm tiết hợp lệ → ngưỡng theo âm tiết vô nghĩa. Nên định nghĩa rare theo **từ/cụm** hoặc theo **thực thể được gán nhãn**.
3. **Cách lấy distractor**: ngẫu nhiên là baseline. Nhưng distractor **đồng âm/khác thanh** mới là phép thử thật (`Hải`/`Hãi`/`Hài`). Đề xuất: báo cáo hai cấu hình — distractor ngẫu nhiên và distractor khó — hai con số sẽ chênh nhau đáng kể, và chênh lệch đó chính là kết quả nghiên cứu.
4. **Từ mượn/code-switching**: dạng chuẩn của `paracetamol` là gì? Nên có **cả hai** biến thể trong danh sách? Ghi rõ, vì nó quyết định con số.
5. **Chuẩn hoá**: NFC, chữ thường, dấu câu, gạch nối.
6. **Các mức $N$**: giữ $\{100, 500, 1000, 2000\}$ để so được với thế giới, cộng thêm mức lớn nếu làm retrieval.

---

## 8.10 Checklist trước khi viết chương thực nghiệm

- [ ] Đã ghi rõ hàm chuẩn hoá và áp cho cả bias list.
- [ ] Báo cáo **WER, B-WER, U-WER** cùng lúc, cộng dồn toàn tập.
- [ ] Có **entity recall/precision** (precision để bắt over-biasing).
- [ ] Quét ít nhất 4 mức $N$.
- [ ] Có baseline **shallow fusion đã tune** (và nếu là RNN-T, có ILME).
- [ ] Có **oracle context** và **no-context** làm hai mốc.
- [ ] Có **đường cong B-WER ↔ U-WER**, không phải một điểm.
- [ ] Có ablation từng tầng nếu hệ thống nhiều tầng.
- [ ] Có RTF/bộ nhớ theo $N$.
- [ ] Có khoảng tin cậy bootstrap cho metric chính.
- [ ] Có ít nhất một bảng **phân tích lỗi định tính** (10–20 ví dụ thật, phân loại).

Nếu đủ 11 mục này, phần thực nghiệm của bạn mạnh hơn phần lớn paper hội nghị.

---

## Bài tập tự kiểm tra

1. Tính tay WER/B-WER/U-WER cho: REF = `"tôi bị dị ứng với amoxicillin"`, HYP = `"tôi bị dị ứng với a mốc xi lin"`, bias list = `{amoxicillin, paracetamol}`.
2. Hệ thống A: B-WER 5.0, U-WER 4.0. Hệ thống B: B-WER 3.0, U-WER 5.5. Hệ thống nào tốt hơn? Câu trả lời phụ thuộc điều gì? Bạn cần thêm thông tin gì để quyết định?
3. Thiết kế tập adversarial-context cho miền y tế tiếng Việt: mô tả cách sinh distractor khó, và bạn kỳ vọng metric nào sẽ xấu đi.
4. Chạy `tools/bwer.py` trên 2 ví dụ ở §8.2 và xác nhận kết quả khớp với tính tay.

---

*Chương tiếp → [`09-thuc-hanh.md`](09-thuc-hanh.md): bắt tay chạy — toolkit nào làm được gì, và bốn baseline có thể dựng trong tuần đầu.*
