# Chương 6 — Bản chất bài toán Contextual ASR

> **Mục tiêu chương:** phát biểu chính xác bài toán, chỉ ra **bốn nguyên nhân gốc** khiến từ hiếm bị nhận sai (mỗi nguyên nhân dẫn tới một họ giải pháp khác nhau), và dựng **bản đồ 7 điểm tiêm ngữ cảnh** — khung để đọc bất kỳ paper nào trong lĩnh vực. Đây là chương "tư duy", nên đọc chậm.

---

## 6.1 Phát biểu bài toán

ASR thường: $\hat{y} = \arg\max_y P(y \mid x)$.

Contextual ASR: có thêm **ngữ cảnh** $C$ tại thời điểm suy luận:

$$\hat{y} = \arg\max_y P(y \mid x, C)$$

$C$ **không** có lúc huấn luyện model gốc, **thay đổi theo từng câu/người dùng/phiên**, và thường **chưa từng xuất hiện** trong dữ liệu huấn luyện.

### $C$ trông như thế nào trong thực tế

| Nguồn ngữ cảnh | Ví dụ | Kích thước |
|---|---|---|
| Danh bạ / playlist của người dùng | `["Nguyễn Văn Hải", "Mẹ", "Sếp Tuấn"]` | 10²–10³ |
| Từ điển miền | tên thuốc, mã bệnh ICD, tên linh kiện | 10³–10⁵ |
| Nội dung màn hình / slide đang chiếu | tên riêng trên slide | 10¹–10² |
| Lượt hội thoại trước đó | *"anh đang nói về dự án Sao Mai"* | văn bản tự do |
| Hồ sơ / metadata | chuyên khoa, vùng miền, ngôn ngữ | vài trường |
| Chính transcript trước đó (long-form) | đoạn đã chép của cùng file | văn bản tự do |

**Hai bài toán khác nhau bị gọi chung một tên** (báo cáo `01` §0.1 đã tách rất rõ — nhắc lại ở đây vì nó quyết định cách bạn đọc paper):

- **Dòng A — Contextual biasing / hotword customization**: $C$ = **danh sách cụm** rời rạc. Câu hỏi: *"làm sao model chịu phát ra đúng những cụm này khi chúng thực sự được nói?"* — chiếm ~85% paper.
- **Dòng B — Long-context / conversational ASR**: $C$ = **văn bản dài liền mạch** (các lượt trước, tài liệu). Câu hỏi: *"làm sao dùng ngữ cảnh ngữ nghĩa để giải mơ hồ?"*

Kỹ thuật, metric, và dataset của hai dòng **khác nhau**. Trộn lẫn là lỗi đọc paper phổ biến nhất của người mới.

---

## 6.2 ★ Bốn nguyên nhân gốc khiến từ hiếm sai

Đây là mục quan trọng nhất tài liệu này. Mỗi nguyên nhân → một họ giải pháp. Chẩn đoán sai nguyên nhân → chọn sai giải pháp → thí nghiệm không cải thiện mà không hiểu vì sao.

### Nguyên nhân 1 — Prior tần suất (internal LM chống lại từ hiếm)

Phân rã theo Bayes ở mức khái niệm:

$$\log P(y|x) \;\approx\; \underbrace{\log P(x|y)}_{\text{bằng chứng âm học}} + \underbrace{\log P(y)}_{\text{prior ngôn ngữ}}$$

Giả sử người ta nói `amoxicillin`, và ứng viên nhầm là `a-mốc-si-lin` (một chuỗi âm tiết tiếng Việt vô nghĩa nhưng "quen tai" hơn). Nếu trong dữ liệu huấn luyện, chuỗi đúng hiếm hơn chuỗi nhầm khoảng $10^3$ lần thì:

$$\Delta \log P(y) \approx \log 10^{-3} \approx -6.9 \text{ nats}$$

Bằng chứng âm học phải thắng được khoảng **7 nats** chỉ để hoà. Trong khi audio thật có nhiễu, giọng vùng miền, nói nhanh → chênh lệch âm học thường chỉ vài nats.

**Kết luận định lượng quan trọng:** biasing phải cung cấp một lượng "điểm thưởng" **đúng cỡ với khoảng cách log tần suất**. Đó là lý do:
- $\lambda$ phải khá lớn → và vì lớn nên **gây over-biasing** ở những câu không chứa thực thể. Đây là gốc rễ của đánh đổi ở §6.3.
- **Trừ ILM trước** (chương 5 §5.5c) giúp giảm chính khoảng cách 7 nats đó, nên cần $\lambda$ nhỏ hơn → ít tác dụng phụ hơn.

→ Họ giải pháp: **shallow fusion / boosting / ILME** (họ 1).

### Nguyên nhân 2 — Mơ hồ âm học và lệch chính tả–phát âm

Có những trường hợp **thông tin đơn giản là không có trong tín hiệu**:
- Đồng âm: `Hải` / `Hãi`, `bảo` / `bão`; `Đại` / `Đài`.
- Chính tả ≠ phát âm: `paracetamol` được đọc *pa-ra-xê-ta-môn* (chương 4 §4.3c).
- Giọng vùng miền lệch có hệ thống: giọng Nam nhập hỏi–ngã.

Ở đây boosting cũng bó tay nếu **khoá tra cứu sai dạng**: model không sinh ra ứng viên nào gần với chuỗi token trong bias list, nên không có gì để boost.

→ Họ giải pháp: **phonetic-aware biasing**, đa biến thể phát âm, retrieval theo âm vị (trục (a) của báo cáo `01` §1.8).

> Đây là nguyên nhân **quan trọng nhất cho tiếng Việt**, và cũng là nơi paper tiếng Anh ít động tới — nên là chỗ có khoảng trống thật.

### Nguyên nhân 3 — Băm token và cắt tỉa beam

Đã phân tích định lượng ở chương 4 §4.2 và chương 5 §5.1: cụm hiếm dài 6–8 token, xác suất trọn cụm suy giảm theo cấp số nhân, và chỉ cần một bước bị đẩy khỏi beam là mất.

→ Họ giải pháp: **trie/copy** (nâng $p$ ở các bước sau), **dynamic vocabulary** (giảm $L$ xuống 1), **boost mức token có rollback**.

### Nguyên nhân 4 — Thực thể chưa từng thấy (OOV) và lệch train–test

Model chưa bao giờ thấy `Ibrutinib` lúc huấn luyện. Với subword thì nó **biểu diễn được** (ghép từ mảnh), nhưng chưa bao giờ **được thưởng** vì sinh ra chuỗi đó.

Còn với module biasing thì có một lệch tinh vi hơn: lúc huấn luyện module biasing, danh sách bias thường được lấy **từ chính reference** (nên luôn chứa đáp án đúng, tức "oracle context"); lúc chạy thật, danh sách nhiễu, thiếu, hoặc thừa. Model học được thói quen "cứ có gì trong danh sách thì tin" → hỏng. Cộng đồng gọi hiện tượng này là **contextual exposure bias** và đang xử lý bằng context dropout, hard negative, DPO (báo cáo `01` §1.5).

→ Họ giải pháp: **cách lấy mẫu bias list lúc huấn luyện** (chương 7 §7.10), text injection, dữ liệu tổng hợp.

---

## 6.3 ★ Tam giác đánh đổi

Mọi phương pháp contextual ASR đều nằm trong tam giác này, và **không phương pháp nào tối ưu cả ba đỉnh**:

```
                    ĐỘ MẠNH BIASING
                (bắt được thực thể — B-WER thấp)
                          ▲
                         ╱ ╲
                        ╱   ╲
                       ╱     ╲
                      ╱       ╲
        AN TOÀN ◄────╱─────────╲────► KHẢ NĂNG SCALE
    (không phá câu             (danh sách 10⁵ vẫn chạy
     thường — U-WER              — thời gian & chất lượng)
     không tăng)
```

- Tăng $\lambda$ → B-WER ↓ nhưng U-WER ↑ (over-biasing: model nhét thực thể vào chỗ không có).
- Deep biasing (họ 2–4) mạnh ở đỉnh trên, nhưng gãy ở đỉnh phải khi $N > 1000$.
- Shallow fusion mạnh ở đỉnh phải, yếu ở đỉnh trên.
- **Retrieval (họ 6) là mẹo phá tam giác**: nó không cải thiện cơ chế biasing, nó **giảm $N$** trước khi biasing chạy — biến bài toán $N=200.000$ thành bài toán $N=10$ mà các cơ chế mạnh đã làm tốt.

### Phát biểu "cổng ngữ cảnh" (contextual gate)

Cách nghĩ thống nhất nhất về mọi phương pháp: model phải học/được cài một **cổng** $g_t \in [0,1]$ trả lời câu hỏi *"tại bước này, có nên nghe theo ngữ cảnh không?"*

$$P(y_t \mid \cdot) = (1-g_t)\,P_{\text{model}}(y_t) + g_t\,P_{\text{context}}(y_t)$$

- TCPGen gọi $g_t$ là **generation probability** $\hat{P}^{gen}$ (chương 7 §7.5).
- CLAS thực hiện điều tương tự bằng cách để attention chọn `<no-bias>` (§7.3).
- Shallow fusion cài $g$ **cứng** bằng tay qua $\lambda$ — nên nó không thích nghi theo bước.
- Speech-LLM để LLM tự quyết định ngầm — và như đã thấy, nó thường quyết định "phớt lờ" (context-utilization gap).

**Mọi thất bại của contextual ASR đều là thất bại của cái cổng này:** hoặc cổng đóng khi cần mở (bỏ sót thực thể), hoặc mở khi cần đóng (over-biasing). Nghiên cứu tốt thường là nghiên cứu làm cho cổng **thông minh hơn**, chứ không phải làm cho biasing "mạnh hơn".

---

## 6.4 Trục "quy mô danh sách" — vì sao $N$ quyết định mọi thứ

| $N$ | Kịch bản thực | Phương pháp còn sống |
|---|---|---|
| 10–100 | slide đang chiếu, vài tên riêng trong cuộc gọi | tất cả, kể cả prompt thuần |
| 100–1.000 | danh bạ cá nhân, thuật ngữ một buổi họp | deep biasing tốt nhất; prompt bắt đầu lung lay |
| 1.000–10.000 | từ điển miền (thuốc, linh kiện) | shallow fusion/trie; deep biasing gãy; **prompt sập** |
| 10.000–200.000 | toàn bộ danh mục sản phẩm, toàn bộ dược điển | **bắt buộc retrieval**; hoặc ký hiệu trên GPU (TurboBias) |

Con số cụ thể trong báo cáo `01` §1.5: prompt-based Speech-LLM đi từ WER 1.3% ($N$=500) lên **22.4%** ($N$=1000) rồi hỏng hoàn toàn ở $N$=2000. Không phải suy giảm dần — mà **sụp đổ**.

**Bài học phương pháp luận:** khi đọc/viết paper contextual ASR, câu hỏi thứ hai (sau "tiêm vào đâu?") luôn phải là **"$N$ bằng bao nhiêu?"**. Một kết quả tuyệt vời ở $N$=100 gần như không nói gì về $N$=5000.

---

## 6.5 ★ Bản đồ 7 điểm tiêm ngữ cảnh

Đây là bảng để dán lên tường. Mỗi họ trong báo cáo `01` = một hàng ở đây, nhưng nhìn theo **cơ chế** thay vì theo tên gọi.

| # | Tiêm ở đâu | Thay đổi cái gì về mặt toán | Đại diện | Cần train lại? | Chịu $N$ lớn |
|---|---|---|---|---|---|
| **1** | **Đặc trưng / encoder** | thêm vector ngữ cảnh vào $h^{enc}$ | contextual adapter, intermediate biasing loss | ✅ (adapter nhỏ) | trung bình |
| **2** | **Trạng thái decoder / prediction net** | thêm $c^z_t$ vào đầu vào của lớp softmax | **CLAS**, CATT, NAM | ✅ | kém khi $N$↑ |
| **3** | **Phân phối đầu ra** (nội suy) | $P = (1-g)P_{mdl} + g\,P_{ptr}$ | **TCPGen**, CopyNE, SeACo | ✅ | trung bình |
| **4** | **Từ vựng đầu ra** (mở rộng) | thêm cột vào ma trận softmax | **Dynamic Vocabulary**, DYNAC | ✅ | kém–trung bình |
| **5** | **Chuỗi đầu vào của LLM** (prompt) | điều kiện hoá tiền tố | Whisper `initial_prompt`, Speech-LLM | ❌ (hoặc LoRA) | **rất kém** |
| **6** | **Điểm số lúc tìm kiếm** | $+\alpha$ theo token, có rollback | shallow fusion, context graph, CTC-WS, TurboBias | ❌ | **tốt nhất** |
| **7** | **Sau khi ra văn bản** | sửa chuỗi đã sinh | post-hoc correction, LLM revision | ✅ (model sửa) | tốt |
| **(0)** | **Trước tất cả: lọc $C$** | $N$: 200.000 → 10 | **retrieval / RAG** | ✅ (riêng retriever) | **phá trần** |

Hai nhận xét đọc ra từ bảng:

1. **Càng tiêm sâu càng mạnh nhưng càng khó scale và càng cần train lại.** Đây là quy luật, không phải trùng hợp: cơ chế sâu dùng attention/softmax trên $N$ mục → chi phí và độ loãng đều tăng theo $N$.
2. **Điểm (0) trực giao với tất cả.** Nên công thức thắng của 2025–2026 là **kết hợp**: `retrieval (0) → deep biasing (2/3/4) → shallow fusion (6) lúc decode → (tuỳ chọn) sửa lỗi (7)`. Các paper mạnh nhất đều là tổ hợp, không phải một cơ chế đơn lẻ.

---

## 6.6 Các ràng buộc triển khai định hình lựa chọn

Trước khi chọn phương pháp, trả lời 5 câu:

1. **Có được train lại model không?** Không → chỉ còn (5), (6), (7).
2. **Có cần streaming không?** Có → loại (5) và phần lớn (7); (6) và CATT là lựa chọn.
3. **$N$ bao nhiêu, và có đổi theo từng câu không?** Đổi liên tục → cơ chế phải cho phép nạp nóng danh sách (trie/embedding tính lại rẻ).
4. **Có nhãn thực thể trong dữ liệu huấn luyện không?** Không → phải sinh (lấy từ reference) hoặc dùng TTS tổng hợp.
5. **Chấp nhận tăng bao nhiêu độ trễ?** Retrieval thêm 1 lượt encode; post-hoc thêm 1 lượt LLM.

---

## 6.7 Dòng B — long-context: bản chất khác ở chỗ nào

Với dòng A, $C$ là **tập hợp** (không thứ tự, mỗi phần tử ngắn), và câu hỏi là *"có/không xuất hiện"*.
Với dòng B, $C$ là **văn bản liền mạch**, và câu hỏi là *"nội dung này giúp giải mơ hồ thế nào"*.

Cơ chế điển hình: nối transcript các lượt trước vào tiền tố decoder (Whisper `<|startofprev|>`), hoặc bộ nhớ chạy dài, hoặc cho LLM thấy cả lịch sử hội thoại.

Đặc trưng riêng của dòng B:
- Lỗi **lan truyền**: transcript lượt trước sai → ngữ cảnh sai → lượt sau sai theo.
- Metric khác: không phải B-WER, mà WER tổng + các đo lường về tính nhất quán thực thể xuyên lượt.
- Rủi ro hallucination cao hơn hẳn.

Nếu đề tài của bạn là hội thoại nhiều lượt trong miền chuyên môn (khám bệnh, họp), bạn sẽ **chạm cả hai dòng** — và phải nói rõ trong luận văn mình đang giải cái nào.

> Dòng B được viết riêng thành **[chương 11](11-long-context-asr.md)**: bốn cơ chế đưa ngữ cảnh dài vào model, **hai chế độ hỏng đối nghịch** (phớt lờ ngữ cảnh ↔ bịa theo ngữ cảnh, có số đo), bộ metric NE-WER/NE-FNR/VIR và thiết kế ablation 5 điều kiện. Nếu bạn theo nhánh này, đọc chương 11 ngay sau chương này.

---

## 6.8 Checklist đọc một paper contextual ASR trong 5 phút

Hỏi đúng 8 câu, theo thứ tự:

1. Dòng A hay dòng B?
2. **Tiêm ở điểm nào** trong bảng §6.5?
3. Kiến trúc nền là gì (CTC / RNN-T / AED / NAR / Speech-LLM)?
4. **$N$ bằng bao nhiêu** trong thí nghiệm? Có quét $N$ không?
5. Có báo cáo **U-WER** không? (Nếu chỉ báo B-WER → nghi ngờ over-biasing bị giấu.)
6. Baseline shallow fusion có được tune tử tế không? Có ILME không?
7. Bias list lúc **huấn luyện** được lấy mẫu ra sao? Có distractor không?
8. Chi phí: RTF/latency/bộ nhớ tăng bao nhiêu theo $N$?

Nếu paper không trả lời được câu 4, 5, 7 → kết quả của nó khó tin, và đó cũng có thể là **khoảng trống để bạn khai thác**.

---

## Bài tập tự kiểm tra

1. Một thực thể xuất hiện trong corpus huấn luyện với tần suất thấp hơn từ nhầm khoảng 500 lần. Ước lượng lượng "điểm thưởng" tối thiểu (theo nats) cần cấp để hai ứng viên hoà nhau. Nếu bạn đã trừ ILM với $\mu=0.5$ thì con số đó đổi thế nào?
2. Cho 4 nguyên nhân gốc ở §6.2. Với mỗi nguyên nhân, thiết kế **một thí nghiệm chẩn đoán** để biết nó có phải thủ phạm chính trong hệ thống của bạn không.
3. Vì sao retrieval được gọi là "trực giao" với các họ khác? Nêu một cấu hình kết hợp 3 tầng và giải thích thứ tự.
4. Bạn đọc một paper báo cáo B-WER giảm từ 12% xuống 4% ở $N$=100, không báo U-WER, không quét $N$. Viết 3 câu hỏi bạn sẽ đặt cho tác giả tại poster.

---

*Chương tiếp → [`07-co-che-tiem-ngu-canh.md`](07-co-che-tiem-ngu-canh.md): chương nặng nhất — công thức và pseudo-code của từng cơ chế, đủ để bạn tự cài lại.*
