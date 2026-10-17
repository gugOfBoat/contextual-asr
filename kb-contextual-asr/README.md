# KNOWLEDGE BASE — CONTEXTUAL ASR

### Nền tảng cơ chế: từ sóng âm → embedding → kiến trúc ASR → cách ngữ cảnh chui vào model

*Biên soạn 29/07/2026. Đây là tài liệu **"how it works"**, bổ sung cho hai báo cáo khảo sát đã có (`00-*.md` = nền tảng + hiện trạng tiếng Việt, `01-*.md` = bản đồ phương pháp/dataset/metric/khoảng trống). Hai báo cáo kia trả lời **"ai đã làm gì"**; tập tài liệu này trả lời **"nó chạy bằng cách nào"**.*

---

## Tài liệu này giải quyết vấn đề gì

Khi đọc paper contextual ASR, bạn sẽ liên tục gặp những câu kiểu:

> *"We inject the bias embeddings into the joint network of the transducer, and combine the pointer distribution with the model distribution via a learned generation probability."*

Nếu chưa nắm gốc thì câu đó là một chuỗi ký hiệu vô nghĩa. Sau khi đọc xong tập này, bạn phải **đọc câu đó như đọc tiếng Việt**: biết joint network nằm ở đâu, bias embedding là vector gì, "pointer distribution" là phân phối trên tập nào, và vì sao phải nội suy hai phân phối thay vì thay thế.

Nguyên tắc biên soạn:

1. **Đi từ dưới lên.** Không dùng khái niệm nào trước khi định nghĩa nó.
2. **Mỗi cơ chế đều có công thức + pseudo-code + chế độ hỏng.** Biết một phương pháp *hỏng khi nào* quan trọng hơn biết nó tốt bao nhiêu — vì đề tài nghiên cứu sinh ra từ chỗ hỏng.
3. **Công thức của các phương pháp lõi đã được đối chiếu với bản full-text gốc** (arXiv/ar5iv), không chép từ trí nhớ. Chỗ nào chưa kiểm chứng tận gốc đều đánh ⚠️.
4. Thuật ngữ tiếng Anh **giữ nguyên** (vì bạn sẽ đọc paper), giải thích tiếng Việt ở lần xuất hiện đầu.

---

## Bản đồ 12 chương

| # | File | Nội dung | Bạn sẽ trả lời được câu hỏi |
|---|---|---|---|
| 1 | [`01-tin-hieu-den-dac-trung.md`](01-tin-hieu-den-dac-trung.md) | Sóng âm → số hoá → framing → STFT → mel → log-mel. SpecAugment, CMVN, các cạm bẫy tiền xử lý | "80 chiều trong log-mel 80 chiều là 80 cái gì?" |
| 2 | [`02-embedding-cho-audio.md`](02-embedding-cho-audio.md) | ★ **Chương embedding.** Bốn trục phân loại; wav2vec2/HuBERT/WavLM/Whisper-encoder; discrete token & codec; speaker embedding; **không gian chung audio–text** (nền của retrieval biasing); phonetic embedding | "Bias list là chữ, audio là sóng — làm sao so sánh chúng trong cùng một không gian?" |
| 3 | [`03-kien-truc-asr.md`](03-kien-truc-asr.md) | ★ **Bản chất model ASR.** Noisy-channel → hybrid → CTC / RNN-T / AED / NAR / Speech-LLM. Toán của từng loại, internal LM, streaming, Conformer/Zipformer | "Vì sao CTC không có mô hình ngôn ngữ nội tại, và vì sao điều đó lại quan trọng với biasing?" |
| 4 | [`04-token-va-tu-vung.md`](04-token-va-tu-vung.md) | Đơn vị đầu ra: char/BPE/phoneme/âm tiết. Tokenizer băm vụn thuật ngữ — nguyên nhân sâu xa khiến biasing khó. Đặc thù tiếng Việt, Unicode NFC/NFD | "Vì sao 'amoxicillin' khó hơn 'nhà' gấp nhiều lần, một cách định lượng?" |
| 5 | [`05-giai-ma-va-lm.md`](05-giai-ma-va-lm.md) | Beam search (CTC prefix / RNN-T / AED), shallow fusion, density ratio, ILME, WFST & Aho–Corasick, rescoring | "Cộng điểm thưởng vào lúc nào, cho token hay cho từ, và làm sao rút lại khi đoán hụt?" |
| 6 | [`06-ban-chat-contextual-asr.md`](06-ban-chat-contextual-asr.md) | ★ **Bản chất bài toán.** $P(y\|x,C)$; 4 nguyên nhân gốc khiến từ hiếm sai; đánh đổi recall ↔ over-biasing ↔ scale; **bảng "7 điểm tiêm ngữ cảnh"** | "Tại sao không thể chỉ ném danh sách vào prompt là xong?" |
| 7 | [`07-co-che-tiem-ngu-canh.md`](07-co-che-tiem-ngu-canh.md) | ★ **Chương nặng nhất.** Công thức + pseudo-code từng họ: shallow fusion, CTC-WS, CLAS, CATT, adapter, trie/TCPGen, dynamic vocabulary, NAM, prompt/Speech-LLM, retrieval, post-hoc. Kèm cách huấn luyện & nghịch lý distractor | "Viết lại được CLAS/TCPGen/Dynamic-Vocab bằng ký hiệu của chính mình" |
| 8 | [`08-do-luong-va-thuc-nghiem.md`](08-do-luong-va-thuc-nghiem.md) | B-WER/U-WER (định nghĩa gốc, ví dụ tính tay), entity F1, đo over-biasing, đo retriever, oracle/upper-bound, kiểm định thống kê, checklist thí nghiệm | "Số của tôi có so được với paper không?" |
| 9 | [`09-thuc-hanh.md`](09-thuc-hanh.md) | Toolkit nào làm được gì, code chạy được cho 4 baseline, dựng bias list, script tính B-WER, checklist debug | "Tuần này chạy được cái gì trên 1 GPU?" |
| 10 | [`10-thuat-ngu-va-doc-them.md`](10-thuat-ngu-va-doc-them.md) | Bảng ký hiệu, ôn toán tối thiểu, từ điển thuật ngữ EN–VI, danh sách đọc 3 tầng | Tra cứu |
| 11 | [`11-long-context-asr.md`](11-long-context-asr.md) | ★ **Nhánh B riêng.** Ngữ cảnh dài/hội thoại/slide: 4 cơ chế, **hai chế độ hỏng đối nghịch** (phớt lờ ngữ cảnh ↔ bịa theo ngữ cảnh), metric NE-WER/VIR, ablation 5 điều kiện | "Cho model ngữ cảnh ĐÚNG mà WER tăng gấp đôi — vì sao?" |
| 12 | [`12-doc-paper-nghien-cuu.md`](12-doc-paper-nghien-cuu.md) | ★ **Phòng đọc paper.** Cách đi từ câu hỏi của tác giả → method → experiment → evidence → limitation; reading map frontier 2025–2026 | "Đọc xong paper này, tôi thực sự biết thêm điều gì và paper kế tiếp phải trả lời câu nào?" |

Kèm theo: [`tools/bwer.py`](tools/bwer.py) — script tính WER / B-WER / U-WER theo đúng định nghĩa gốc, chạy được ngay.

---

## Bốn lộ trình đọc

**Lộ trình A — "tôi cần hiểu để đọc paper trong 2 ngày"**
`06` → `03` (§CTC, §RNN-T, §AED) → `07` → `08`.
Bỏ qua chương 1–2 nếu bạn đã quen mel-spectrogram; nhưng **phải** đọc §2.7 (không gian chung audio–text) trước khi đọc phần retrieval của chương 7.

**Lộ trình B — "tôi đi nhánh long-context (thầy gọi là contextual ASR)"**
`06` (bản chất, đặc biệt §6.1 và §6.7) → `03` §3.5 + §3.7 (AED/Whisper prev-text và Speech-LLM) → **`11`** → `08` §8.6 (thí nghiệm chẩn đoán) → **`12`** (đọc paper frontier).
Chương 7 đọc lướt để biết nhánh A làm gì — vì reviewer sẽ hỏi bạn tại sao không dùng biasing.

**Lộ trình C — "tôi muốn nắm gốc, không vội"**
Đọc tuần tự `01` → `12`. Chương 12 chỉ bắt đầu sau khi đã nắm chương 6–8 và 11.

**Lộ trình D — "tôi cần bắt tay code ngay"**
`09` → chạy 3 baseline → khi gặp chỗ không hiểu thì quay lại tra `03`/`05`/`07`.

**Lộ trình E — "tôi đang chuẩn bị báo cáo paper, chưa chọn đề tài"**
`12` → chọn một tuyến đọc → mở hồ sơ trong [`../4-papers/frontier-2026/`](../4-papers/frontier-2026/) → tự trả lời checklist cuối mỗi hồ sơ. Không nhảy sang proposal trước khi hoàn thành ít nhất một tuyến.

---

## Sợi chỉ đỏ xuyên suốt

Toàn bộ contextual ASR có thể tóm vào **một câu hỏi duy nhất**:

> Model có sẵn một niềm tin tiên nghiệm (prior) rằng chuỗi từ nào dễ xuất hiện. Niềm tin đó học từ dữ liệu huấn luyện, nên nó **luôn chống lại từ hiếm**. Câu hỏi: làm sao **sửa prior đó tại thời điểm suy luận**, chỉ cho đúng những chỗ cần sửa, mà không phá phần còn lại?

Bảy họ phương pháp trong báo cáo `01` chẳng qua là **bảy chỗ khác nhau để cắm cái sửa đó vào**:

```
audio ──►[đặc trưng]──►[ENCODER]──►[DECODER/JOINT]──►[phân phối token]──►[BEAM SEARCH]──►text──►[SỬA LỖI]
                            ▲             ▲                  ▲                 ▲                    ▲
                            │             │                  │                 │                    │
                       ② adapter/    ② CLAS,CATT      ③ TCPGen (copy)    ① shallow fusion    ⑦ post-hoc
                        deep bias      ⑤ prompt        ④ dynamic vocab      context graph      correction
                            ▲
                            └── ⑥ retrieval: lọc 200k cụm → 10 cụm rồi mới đưa vào bất kỳ chỗ nào ở trên
```

Nhớ sơ đồ này thì đọc paper nào cũng biết ngay nó đứng ở đâu. Chương 6 §6.5 mở rộng sơ đồ này thành bảng đầy đủ.

---

## Quy ước ký hiệu (dùng thống nhất cả 12 chương)

| Ký hiệu | Nghĩa |
|---|---|
| $x = (x_1,\dots,x_T)$ | chuỗi đặc trưng audio, $T$ frame |
| $y = (y_1,\dots,y_U)$ | chuỗi token đầu ra, $U$ token |
| $h^{enc} \in \mathbb{R}^{T' \times d}$ | đầu ra encoder (sau subsampling, $T' \approx T/4$) |
| $C = \{c_1,\dots,c_N\}$ | ngữ cảnh: danh sách $N$ cụm bias (hotword) |
| $h^{bias} \in \mathbb{R}^{N \times d}$ | embedding của các cụm bias |
| $\mathcal{V}$ | tập token (vocabulary), $\|\mathcal{V}\| = K$ |
| $\varnothing$ | ký hiệu blank (CTC/RNN-T) |
| $\lambda, \alpha$ | trọng số fusion/boosting |
| $\tau$ | temperature trong contrastive loss |

---

*Chương 1 bắt đầu tại đây → [`01-tin-hieu-den-dac-trung.md`](01-tin-hieu-den-dac-trung.md)*
