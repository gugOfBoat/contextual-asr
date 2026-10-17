# Chương 11 — Nhánh B: Long-context / conversational ASR

> **Mục tiêu chương:** nhánh B dùng ngữ cảnh **liền mạch** (lượt hội thoại trước, tài liệu, slide, hồ sơ) thay vì **danh sách cụm**. Cơ chế, chế độ hỏng và metric đều khác nhánh A. Chương này viết riêng vì đây có thể là nhánh bạn theo (thầy hướng dẫn gọi long-context ASR là "contextual ASR").
>
> Các con số trong chương này được đối chiếu trực tiếp từ bản full-text (arXiv HTML), không chép theo trí nhớ.

---

## 11.1 Khác nhánh A ở đâu — bốn khác biệt cấu trúc

| | **Nhánh A — biasing** | **Nhánh B — long-context** |
|---|---|---|
| $C$ là gì | tập hợp $N$ cụm rời rạc, không thứ tự | văn bản/tài liệu **liền mạch**, có cấu trúc |
| Câu hỏi model phải trả lời | *"cụm này có được nói không?"* (khớp bề mặt) | *"nội dung này giúp giải mơ hồ thế nào?"* (ngữ nghĩa) |
| Cơ chế điển hình | trie, attention danh sách, dynamic vocab, retrieval | điều kiện hoá tiền tố, cross-utterance attention, bộ nhớ, omni-modal LLM |
| Metric | B-WER / U-WER | NE-WER, NE-FNR, **VIR**, tcpWER, WER tổng |
| Trạng thái | LibriSpeech-biasing gần bão hoà (B-WER ~2.8%) | **chưa bão hoà** — hai chế độ hỏng lớn còn mở |

Điểm giống duy nhất, nhưng là điểm quan trọng nhất: **vẫn là bài toán cái cổng** (chương 6 §6.3). Model phải quyết định *khi nào tin ngữ cảnh*. Nhánh A hỏng vì cổng mở sai chỗ trên một danh sách; nhánh B hỏng vì cổng đóng cứng (phớt lờ) hoặc mở toang (bịa theo ngữ cảnh).

---

## 11.2 Các nguồn ngữ cảnh dài

| Nguồn | Ví dụ | Tính chất khó |
|---|---|---|
| **Lượt hội thoại trước** | transcript các lượt đã chép của cùng cuộc trò chuyện | **do chính hệ thống sinh ra** → lỗi lan truyền |
| **Transcript trước trong long-form** | 30 s trước của cùng file | như trên, cộng thêm biên cửa sổ |
| **Tài liệu miền** | hướng dẫn điều trị, bệnh án, tài liệu kỹ thuật | dài, phần lớn không liên quan |
| **Slide / màn hình** | slide bài giảng đang chiếu | **đa mô thức**, và là nguồn hỏng nặng nhất (§11.4b) |
| **Hồ sơ / metadata** | chuyên khoa, vai người nói, chủ đề | ngắn, ít rủi ro, ít lợi ích |
| **Video / hình ảnh cảnh** | môi trường, vật thể | tín hiệu yếu, nhiễu cao |

---

## 11.3 Bốn cơ chế đưa ngữ cảnh dài vào model

### (a) Điều kiện hoá tiền tố (prefix conditioning) — rẻ nhất

Nối văn bản ngữ cảnh vào **đầu chuỗi decoder**. Với Whisper là `<|startofprev|>` (chương 3 §3.5). Với Speech-LLM là prompt.

$$P(y \mid x, C) = \prod_u P\big(y_u \mid y_{<u},\, \underbrace{C}_{\text{tiền tố}},\, x\big)$$

- **Ưu**: không train lại; Whisper vốn đã được huấn luyện với dạng này nên "hiểu" nó tự nhiên hơn hẳn so với việc nhét một danh sách hotword.
- **Nhược**: giới hạn độ dài (Whisper: cửa sổ decoder 448 token); ngữ cảnh dài lấn át; và nếu ngữ cảnh sai thì model đi theo cái sai.

> **Ghi chú quan trọng cho luận văn:** đây là chỗ nhánh B *hợp* với `initial_prompt` còn nhánh A thì *không*. Với nhánh A, prompt bị lệch phân bố (danh sách 1000 từ không giống transcript). Với nhánh B, prompt **đúng là** thứ Whisper được huấn luyện để nhận. Nên kết quả prompt-based ở nhánh B đáng tin hơn nhiều — và điều này giải thích vì sao hai nhánh cho kết luận trái ngược về "prompt có hiệu quả không".

### (b) Cross-utterance encoder / concat

Đưa **cả audio lẫn text của các lượt trước** qua encoder, rồi cho decoder attend vào toàn bộ:

```
[audio lượt t-2][audio lượt t-1][audio lượt t] ─► encoder chung ─► decoder chép lượt t
                        hoặc
   text lượt t-1 ─► text encoder ─┐
   audio lượt t  ─► speech enc ───┴─► cross-attention hợp nhất ─► decoder
```

Mạnh hơn (a) vì ngữ cảnh đi qua tầng biểu diễn chứ không chỉ qua tiền tố; nhưng cần train lại và tốn bộ nhớ theo độ dài lịch sử.

### (c) Bộ nhớ chạy (recurrent memory / state carry-over)

Giữ một vector/tập vector trạng thái mang thông tin các lượt trước, cập nhật dần. Chi phí không tăng theo độ dài lịch sử — hợp streaming và audio rất dài. Đổi lại: thông tin bị nén, khó truy xuất chính xác một tên riêng đã nói 10 phút trước.

### (d) Omni-modal LLM (ngữ cảnh thị giác)

Slide/ảnh → vision encoder → token thị giác, ghép cùng token audio trong chuỗi đầu vào của LLM. Đây là dòng nóng nhất, và cũng là dòng lộ ra chế độ hỏng ngoạn mục nhất ở §11.4b.

---

## 11.4 ★ Hai chế độ hỏng đối nghịch — trục nghiên cứu chính của nhánh B

Đây là phần đáng giá nhất chương này. Nhánh B hỏng theo **hai hướng ngược nhau**, và chưa hướng nào được giải.

### (a) Dưới-dùng ngữ cảnh (context-utilization gap)

Cho model ngữ cảnh **oracle** (chứa đúng thông tin cần) mà WER gần như không đổi → model *nhận* nhưng không *dùng*.

**Cơ chế** (đã phân tích ở chương 3 §3.7): trong dữ liệu huấn luyện, ngữ cảnh phần lớn **vô ích**, nên chiến lược tối ưu mà model học được là phớt lờ nó và tin audio. Đây là shortcut learning kinh điển.

**Cách đo**: cặp thí nghiệm `no-context` vs `oracle-context` (chương 8 §8.6). Nếu khoảng cách ≈ 0 → bạn đang đối mặt với chế độ hỏng này, và **mọi cải tiến cơ chế đều vô nghĩa cho tới khi sửa được nó**.

### (b) Trên-dùng ngữ cảnh (visual interference / over-reliance)

Ngược lại hoàn toàn: model **chép cái nó nhìn thấy thay vì cái nó nghe thấy**.

*VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models* (arXiv 2510.08618, ACL 2026) định nghĩa và đo hiện tượng này:

> **Visual Interference**: *"model sinh ra văn bản có mặt trong ngữ cảnh thị giác nhưng **không** có trong tiếng nói."*

**VIR (Visual Interference Rate)** — metric họ đề xuất: xác định tập từ **chỉ có trên slide** (không có trong audio), rồi đếm tỉ lệ mẫu mà các từ đó xuất hiện trong đầu ra của model.

Các con số đã đối chiếu từ bản full-text:

| Quan sát | Số |
|---|---|
| Qwen3-Omni-30B-A3B trên SlideSpeech: **audio-only** → **có ảnh slide** | WER **11.71 → 24.13** (tệ đi hơn 2×) |
| VIR của MiniCPM-o-2.6 (test set) | **63.28%** |
| VIR của Qwen2.5-Omni-7B | **12.87%** |
| VAPO-7B (phương pháp của họ, có ngữ cảnh thị giác) | WER **10.31**, recall thực thể 97.32% |

**Đọc bảng này cho đúng:** cho model một nguồn ngữ cảnh giàu thông tin và **đúng** (slide của chính bài giảng đó) lại làm WER **tăng gấp đôi**. Không phải ngữ cảnh sai — mà là **model không biết cân bằng hai mô thức**. Đây là bằng chứng mạnh nhất hiện có cho luận điểm: *bài toán không phải "cấp thêm ngữ cảnh", mà là "dạy model quyết định khi nào nghe theo"*.

### Vì sao hai chế độ hỏng này là cùng một vấn đề

Cả hai đều là **hỏng cổng** $g$:
- (a) $g \approx 0$ luôn → ngữ cảnh vô dụng.
- (b) $g \approx 1$ khi gặp mô thức "mạnh" (chữ nhìn thấy được) → bịa.

Một đề tài tốt của nhánh B thường là: **làm cho $g$ có điều kiện và đo được**, thay vì làm cho ngữ cảnh "giàu hơn".

---

## 11.5 Lỗi lan truyền — đặc thù riêng của hội thoại

Ngữ cảnh của lượt $t$ là transcript của lượt $t-1$ **do chính hệ thống sinh ra**. Nên:

$$\text{lỗi ở lượt } t-1 \;\Rightarrow\; \text{ngữ cảnh sai ở lượt } t \;\Rightarrow\; \text{lỗi ở lượt } t \;\Rightarrow\; \dots$$

Ba hệ quả thực nghiệm phải kiểm soát:

1. **Phải so cả hai chế độ**: ngữ cảnh **oracle** (transcript đúng của lượt trước) vs ngữ cảnh **thật** (transcript do hệ thống sinh). Chênh lệch giữa hai cái = mức độ nhạy với lỗi lan truyền. Paper nào chỉ báo cáo oracle là paper đã báo cáo một kịch bản không tồn tại trong thực tế.
2. **Lỗi thực thể tự củng cố**: một tên riêng sai ở lượt 1 sẽ được "nhắc lại" đúng như sai ở các lượt sau — hệ thống trở nên **nhất quán trong cái sai**. Metric WER không bắt được điều này; cần đo tính nhất quán thực thể xuyên lượt.
3. **Contextual exposure bias** (chương 6 §6.2, nguyên nhân 4): train với ngữ cảnh sạch, test với ngữ cảnh bẩn. Cách chữa: huấn luyện với ngữ cảnh **do chính model sinh** (scheduled sampling), hoặc chèn nhiễu vào ngữ cảnh lúc huấn luyện.

---

## 11.6 Metric cho nhánh B

B-WER/U-WER **không dùng được** ở đây (không có danh sách bias). Bộ metric tương ứng:

| Metric | Đo gì | Ghi chú |
|---|---|---|
| **WER tổng** | mặt bằng chung | thường **gần như không đổi** khi thêm ngữ cảnh → báo cáo riêng WER là vô nghĩa |
| **NE-WER** (named-entity WER) | WER **chỉ trên các thực thể** | đây mới là chỗ ngữ cảnh có tác dụng |
| **NE-FNR** (false negative rate ở thực thể) | tỉ lệ thực thể bị bỏ sót | bổ sung cho NE-WER |
| **VIR** | tỉ lệ mẫu bị nhiễm ngữ cảnh (bịa theo cái nhìn thấy) | metric **an toàn** — bắt chế độ hỏng (b) |
| **tcpWER / cpWER** | nếu có nhiều người nói | công cụ `meeteval` |

> **Quy tắc trình bày cho luận văn nhánh B:** luôn báo cáo **một cặp**: một metric *lợi ích* (NE-WER hoặc NE-FNR) và một metric *an toàn* (VIR hoặc WER trên tập không có ngữ cảnh liên quan). Y hệt tinh thần cặp B-WER/U-WER ở nhánh A. Chỉ báo cáo một vế là giấu đi một nửa.

### Ablation 5 điều kiện — thiết kế chuẩn

Chạy cùng một model, chỉ đổi $C$:

| Điều kiện | $C$ là gì | Cho biết |
|---|---|---|
| **none** | không có | cận dưới |
| **real** | ngữ cảnh thật (có thể chứa lỗi) | con số dùng được trong thực tế |
| **oracle** | ngữ cảnh đúng hoàn hảo | **cận trên** / đo context-utilization gap |
| **adversarial** | ngữ cảnh của chủ đề khác / cố tình sai | đo tính an toàn |
| **shuffle** | đúng nội dung nhưng **xáo trộn thứ tự / ghép sai câu** | ★ phân biệt *"model dùng ngữ nghĩa"* với *"model chỉ ăn theo phân bố từ vựng"* |

Điều kiện **shuffle** là điều kiện tinh tế nhất và ít ai chạy: nếu kết quả với `shuffle` gần bằng `oracle`, nghĩa là model **không hề dùng cấu trúc ngữ nghĩa** — nó chỉ hưởng lợi từ việc thấy đúng tập từ. Đó là một phát hiện đáng viết ra.

---

## 11.7 Cạm bẫy benchmark: dữ liệu tổng hợp thổi phồng kết quả

Nhiều benchmark nhánh B được dựng bằng **LLM sinh văn bản → TTS sinh audio** (rẻ, kiểm soát được thực thể). Khoảng cách giữa tổng hợp và thật rất lớn — số từ chính VAPO:

| Tập | NE-WER của VAPO-7B |
|---|---|
| **SlideASR-S** (tổng hợp: LLM sinh text + TTS + render slide bằng Matplotlib) | **2.83** |
| **SlideASR-R** (thật: 60 video thuyết trình học thuật, 200 thực thể chuyên ngành) | **26.48** |

Chênh gần **9×** trên cùng một model, cùng một metric. Bài học:

1. Kết quả trên benchmark tổng hợp **không** dự đoán được hiệu năng thật.
2. Nếu bạn dựng benchmark tiếng Việt, **bắt buộc** phải có phần audio thật, dù nhỏ.
3. Ngược lại, đây cũng là **cơ hội**: dữ liệu tổng hợp dùng để *huấn luyện* thì tốt (VAPO train trên phần tổng hợp rồi vẫn thắng trên tập thật) — chỉ đừng dùng để *kết luận*.

---

## 11.8 Nhánh B với tiếng Việt: chỗ nào có thể là đóng góp

Ba quan sát nối kiến thức chương này với bối cảnh tiếng Việt (báo cáo `00`):

1. **Thanh điệu là loại lỗi mà ngữ cảnh ngữ nghĩa *nên* sửa được.** `bảo`/`bão`/`báo` khác nhau hoàn toàn về nghĩa nhưng gần nhau về âm. Đây đúng là loại mơ hồ mà một LM/ngữ cảnh mạnh giải được, trong khi tăng chất lượng âm học thì không. → Câu hỏi nghiên cứu sạch: *ngữ cảnh hội thoại sửa được bao nhiêu phần trăm lỗi thanh điệu?* Chưa thấy ai trả lời ⚠️.
2. **Phải phân tích lỗi theo loại**, không chỉ WER: thanh / nguyên âm–phụ âm / thực thể. Nếu ngữ cảnh chỉ sửa được lỗi thực thể mà không sửa được lỗi thanh, đó cũng là một kết quả có giá trị (và giải thích được: thanh điệu là lỗi âm học, không phải lỗi ngữ nghĩa).
3. **Lỗi lan truyền có thể nặng hơn với tiếng Việt** vì một lỗi thanh làm đổi hẳn nghĩa từ, khiến ngữ cảnh lượt sau bị đầu độc mạnh hơn so với một lỗi chính tả nhẹ trong tiếng Anh ⚠️ *(giả thuyết, chưa ai kiểm chứng — kiểm được bằng thí nghiệm oracle-vs-real ở §11.5)*.

---

## 11.9 Tóm tắt chương

- Nhánh B khác nhánh A ở **dạng của $C$**, kéo theo khác cả cơ chế lẫn metric.
- Bốn cơ chế: tiền tố · cross-utterance · bộ nhớ · omni-modal.
- **Hai chế độ hỏng đối nghịch** — phớt lờ ngữ cảnh (context-utilization gap) và bịa theo ngữ cảnh (visual interference, VIR tới 63.28%). Cả hai đều là hỏng **cổng**, cả hai đều chưa được giải.
- Cấp thêm ngữ cảnh **đúng** vẫn có thể làm WER tăng gấp đôi (11.71 → 24.13 với Qwen3-Omni khi thêm slide) — bằng chứng rằng vấn đề nằm ở cân bằng mô thức, không nằm ở lượng thông tin.
- Metric phải đi theo **cặp lợi ích + an toàn**; ablation 5 điều kiện, đặc biệt là **shuffle**.
- Benchmark tổng hợp thổi phồng kết quả tới ~9× (NE-WER 2.83 vs 26.48).

---

## Bài tập tự kiểm tra

1. Với model tiếng Việt của bạn, chạy ablation 5 điều kiện trên 200 câu hội thoại. Khoảng cách `none` → `oracle` là bao nhiêu? Nếu ≈ 0, bạn kết luận gì và làm gì tiếp?
2. Thiết kế một metric kiểu VIR nhưng cho ngữ cảnh **văn bản** (không phải slide). Định nghĩa chính xác tập "từ chỉ có trong ngữ cảnh" và cách đếm.
3. Vì sao điều kiện `shuffle` lại phân biệt được "dùng ngữ nghĩa" với "ăn theo phân bố từ vựng"? Cho một ví dụ cụ thể bằng tiếng Việt.
4. Bạn quan sát: ngữ cảnh oracle cải thiện NE-WER 30% nhưng WER tổng chỉ giảm 0.3 điểm. Giải thích hiện tượng, và nói xem bạn sẽ báo cáo con số nào trong abstract của luận văn.

---

**Nguồn đã đối chiếu cho chương này:**
- [VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models (arXiv 2510.08618, ACL 2026)](https://arxiv.org/abs/2510.08618) — định nghĩa Visual Interference, metric VIR, các con số 11.71→24.13, VIR 63.28%/12.87%, VAPO-7B 10.31 WER, NE-WER 2.83 (tổng hợp) vs 26.48 (thật), cách dựng SlideASR-Bench.

*Quay lại mục lục → [`README.md`](README.md)*
