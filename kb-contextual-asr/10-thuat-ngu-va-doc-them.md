# Chương 10 — Tra cứu: toán, thuật ngữ, danh sách đọc

---

## 10.1 Ôn toán tối thiểu (đủ để đọc mọi paper trong lĩnh vực)

### Xác suất

- **Quy tắc chuỗi**: $P(y_1..y_U) = \prod_u P(y_u \mid y_{<u})$ — đây là toàn bộ "tính tự hồi quy".
- **Bayes**: $P(y|x) \propto P(x|y)P(y)$ — nền của noisy channel và của lập luận "prior chống lại từ hiếm" (chương 6 §6.2).
- **Log-domain**: nhân xác suất → cộng log. Mọi decoder làm việc trong miền log để tránh tràn số.
- **log-sum-exp**: $\log\sum_i e^{a_i} = a^* + \log\sum_i e^{a_i - a^*}$ với $a^*=\max a_i$. Dùng ở khắp nơi (CTC forward, gộp beam).

### Softmax và temperature

$$P_k = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}$$

$\tau \to 0$: nhọn (gần argmax). $\tau \to \infty$: phẳng (gần đều). Trong contrastive learning, $\tau$ nhỏ = phạt nặng negative gần.

**Cộng một hằng số vào logit = nhân xác suất với một hệ số** (trước khi chuẩn hoá lại). Đây chính là lý do "boosting" hoạt động: $z_k \mathrel{+}= \alpha \Rightarrow P_k$ được nhân $e^\alpha$.

### Attention

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

Đọc thành lời: *"query hỏi mỗi key một câu; câu trả lời (điểm) được chuẩn hoá thành trọng số; kết quả là trung bình có trọng số của các value."*
Trong biasing: **query = trạng thái model**, **key/value = embedding các cụm bias**. Cả CLAS, CATT, TCPGen đều chỉ là biến thể của câu này.

### Quy hoạch động

CTC forward–backward, RNN-T lattice, Levenshtein alignment — cùng một khuôn: *"nghiệm tối ưu của bài toán con được tái sử dụng"*. Nếu bạn viết được thuật toán Levenshtein từ đầu, bạn hiểu đủ.

### Học tương phản (contrastive)

$$\mathcal{L} = -\log \frac{e^{\text{sim}(a, t^+)/\tau}}{e^{\text{sim}(a,t^+)/\tau} + \sum_{t^-} e^{\text{sim}(a,t^-)/\tau}}$$

Chất lượng của không gian học được **phụ thuộc gần như hoàn toàn vào chất lượng của $t^-$** (negative). Đây là lý do "hard negative mining" quan trọng hơn kiến trúc.

### Độ phức tạp cần nhớ

| Thao tác | Chi phí |
|---|---|
| Self-attention trên $T$ frame | $O(T^2 d)$ |
| Cross-attention vào $N$ cụm bias | $O(U N d)$ ← **tăng tuyến tính theo $N$** |
| Beam search | $O(U \cdot B \cdot \text{cost}_{step})$ |
| Trie matching | $O(1)$ mỗi bước (không phụ thuộc $N$!) ← lý do phương pháp ký hiệu scale tốt |
| ANN search trên 200k vector | $O(\log N)$ xấp xỉ |

Bảng này giải thích bằng một cái nhìn vì sao deep biasing gãy còn shallow fusion và retrieval thì không.

---

## 10.2 Bảng ký hiệu dùng trong knowledge base này

| Ký hiệu | Nghĩa | Xuất hiện ở |
|---|---|---|
| $x, T$ | đặc trưng audio, số frame | ch.1, 3 |
| $y, U$ | chuỗi token đích, độ dài | ch.3 |
| $h^{enc}$ | đầu ra encoder | ch.3 |
| $\varnothing$ | blank | ch.3 |
| $C, N$ | danh sách bias, kích thước | ch.6, 7 |
| $c_n, L_n$ | cụm bias thứ $n$ và độ dài token | ch.7 |
| $h^z_i$ | embedding cụm bias (CLAS) | ch.7 §7.3 |
| $h^{bias}=[h^{start};h^{cont}]$ | vector trạng thái trie | ch.7 §7.4 |
| $P^{ptr}, P^{gen}, \hat{P}^{gen}$ | phân phối con trỏ, cổng, cổng đã hiệu chỉnh | ch.7 §7.5 |
| $\mathcal{V}^b$ | từ vựng động | ch.7 §7.6 |
| $\alpha$ | trọng số boosting | ch.5, 7 |
| $\lambda, \mu$ | trọng số LM ngoài, trọng số trừ ILM | ch.5 §5.5 |
| $\tau$ | temperature | ch.2, 10 |
| $g_t$ | cổng ngữ cảnh | ch.6 §6.3 |

---

## 10.3 Từ điển thuật ngữ EN → VI

| Tiếng Anh | Tiếng Việt / giải thích ngắn |
|---|---|
| Acoustic model | mô hình âm học — $P(x\|y)$ |
| Alignment | căn chỉnh giữa frame audio và token |
| Beam search | tìm kiếm chùm, giữ $B$ giả thuyết |
| Bias list / hotword list | danh sách cụm cần ưu tiên |
| B-WER / U-WER | WER trên từ **có** / **không có** trong bias list |
| Blank | ký hiệu rỗng của CTC/RNN-T |
| CIF (Continuous Integrate-and-Fire) | cơ chế "tích luỹ rồi bắn" sinh 1 embedding/token |
| CMVN | chuẩn hoá trung bình–phương sai đặc trưng |
| Contextual biasing | định hướng nhận dạng theo ngữ cảnh |
| Context graph | đồ thị/trie ngữ cảnh dùng lúc decode |
| Contextual exposure bias | lệch do train với ngữ cảnh oracle, test với ngữ cảnh nhiễu |
| Distractor | cụm nhiễu thêm vào bias list |
| Fertility | số token trung bình/từ của tokenizer |
| Forced alignment | căn chỉnh cưỡng bức khi đã biết transcript |
| Hallucination | model bịa nội dung không có trong audio |
| Internal LM (ILM) | mô hình ngôn ngữ nội tại của model E2E |
| ILME | ước lượng và trừ bớt ILM |
| Joint network | mạng hợp nhất trong RNN-T |
| Over-biasing | chèn thực thể vào chỗ không có |
| Peaky behavior | CTC phát blank hầu hết, spike nhọn tại token |
| Pointer/copy mechanism | cơ chế trỏ/sao chép từ danh sách |
| Prediction network | mạng dự đoán (LM nội tại) của RNN-T |
| Recall@k | tỉ lệ thực thể đúng nằm trong top-$k$ retrieval |
| Rescoring | chấm điểm lại n-best/lattice |
| Shallow fusion | cộng log-prob LM ngoài lúc beam search |
| SSL (self-supervised learning) | học tự giám sát |
| Subsampling | giảm tần số khung ở encoder |
| Teacher forcing | huấn luyện với tiền tố đúng |
| Trie / prefix tree | cây tiền tố |
| WFST | máy trạng thái hữu hạn có trọng số |

---

## 10.4 Danh sách đọc — ba tầng

### Tầng 1 — bảy bài phải đọc, đúng thứ tự này

| # | Bài | Vì sao đọc | Đọc xong nắm được |
|---|---|---|---|
| 1 | **Le et al., *Contextualized Streaming End-to-End Speech Recognition with Trie-Based Deep Biasing and Shallow Fusion***, Interspeech 2021 (arXiv 2104.02194) | **định nghĩa B-WER/U-WER + benchmark LibriSpeech-biasing** mà cả lĩnh vực dùng | ngôn ngữ chung để đọc mọi bài sau |
| 2 | **Pundak et al., CLAS — *Deep Context***, SLT 2018 (arXiv 1808.02480) | bài khai sinh deep biasing | cơ chế attention-vào-danh-sách (ch.7 §7.3) |
| 3 | **Zhao et al., *Shallow-Fusion End-to-End Contextual Biasing***, Interspeech 2019 | chỉ ra vì sao biasing khó với E2E (beam cắt sớm) | vì sao phải boost mức token, không phải mức từ |
| 4 | **Sun et al., TCPGen**, ASRU 2021 (arXiv 2109.00627) | cơ chế đẹp nhất về toán | trie mask + cổng học được (ch.7 §7.5) |
| 5 | **Sudo et al., *Dynamic Vocabulary***, Interspeech 2024 (arXiv 2405.13344) | giải quyết vấn đề tokenizer băm vụn | mở rộng từ vựng lúc suy luận (ch.7 §7.6) |
| 6 | **BR-ASR**, Interspeech 2025 (arXiv 2505.19179) | mở kỷ nguyên retrieval, scale 200k | vì sao retrieval phá được tam giác đánh đổi |
| 7 | **Andrusenko et al., CTC-WS**, Interspeech 2024 (arXiv 2406.07096) | baseline ký hiệu nhanh, đã vào NeMo | biasing không cần train lại |

*(Bảy bài này trùng với đề xuất ở báo cáo `01` §6.3 — giữ nguyên vì thứ tự đó hợp lý.)*

### Tầng 2 — theo hướng bạn chọn

- **Nếu đi hướng phonetic-aware (khuyến nghị cho tiếng Việt):** PROCTER (ICASSP 2023) → Automatic Text Pronunciation Correlation (ICASSP 2025, arXiv 2501.00804) → PARCO (ASRU 2025, arXiv 2509.04357) → zero-shot trie đa phát âm (APSIPA 2025, arXiv 2508.17796).
- **Nếu đi hướng retrieval:** GLCLAP → PSC-Joint (TASLP 2025) → COALA (Interspeech 2026) → H-PRM.
- **Nếu đi hướng Speech-LLM:** CTC-Assisted LLM (SLT 2024, arXiv 2411.06437) → các bài DPO/GRPO 2026 → ProfASR-Bench (context-utilization gap).
- **Nếu đi hướng benchmark/metric:** Le et al. (lại) → PIER (2025) → Contextual Earnings-22.
- **Nếu đi nhánh B (long-context — xem [chương 11](11-long-context-asr.md)):** **VAPO** (arXiv 2510.08618, ACL 2026 — Visual Interference + metric VIR + SlideASR-Bench) → **ProfASR-Bench** (context-utilization gap) → SlideSpeech (ICASSP 2024) → VHASR (EMNLP 2024). Về dữ liệu hội thoại tiếng Việt: **MLC-SLM Challenge (Interspeech 2025)** được ghi nhận có ~100 giờ hội thoại tiếng Việt tự nhiên kèm baseline mở ⚠️ *(ghi từ khảo sát trước — kiểm lại trang challenge trước khi dựa vào)*.

Danh sách nguồn đầy đủ kèm mã arXiv nằm ở **báo cáo `01` Phần VII** — không lặp lại ở đây.

### Tầng 3 — nền tảng (đọc khi cần tra)

- CTC: Graves et al. 2006.
- RNN-T: Graves 2012; ILME: Meng et al. 2021.
- Conformer: Gulati et al. 2020.
- wav2vec 2.0: Baevski et al. 2020; HuBERT: Hsu et al. 2021; WavLM: Chen et al. 2022.
- Whisper: Radford et al. 2022.
- SpecAugment: Park et al. 2019.
- Sách/khoá: *Speech and Language Processing* (Jurafsky & Martin) ch. ASR; tài liệu ESPnet/icefall.

---

## 10.5 Bản đồ liên kết ba tài liệu của bạn

```
00-bao-cao-khao-sat-asr-hoi-thoai-phuc-tap-tieng-viet.md
      → hiện trạng tiếng Việt, 4 trục khó, dữ liệu, WER mặt bằng
            (trả lời: "bối cảnh nào, tiếng Việt đang ở đâu")

01-bao-cao-khao-sat-contextual-asr.md
      → 7 họ phương pháp, dataset, metric, khoảng trống, 5 hướng đề tài
            (trả lời: "ai đã làm gì, còn trống chỗ nào")

kb-contextual-asr/  ← tài liệu này
      → cơ chế: embedding, kiến trúc, giải mã, công thức từng phương pháp, cách đo
            (trả lời: "nó chạy bằng cách nào, tôi cài lại và đo thế nào")
```

Khi làm việc: dùng `01` để **chọn** hướng, dùng knowledge base này để **thực hiện** và **đọc hiểu**, dùng `00` để **đặt vào bối cảnh tiếng Việt** khi viết luận văn.

---

## 10.6 Ba câu hỏi tự kiểm tra cuối cùng

Nếu trả lời trôi chảy cả ba, bạn đã nắm phần lớn kiến thức nền của contextual ASR:

1. **Vẽ lại từ đầu**: sơ đồ 7 điểm tiêm ngữ cảnh (ch.6 §6.5), và với mỗi điểm nêu một phương pháp đại diện + chế độ hỏng của nó.
2. **Giải thích cho người ngoài ngành trong 3 phút**: vì sao một hệ thống nhận dạng tốt 95% lại vẫn sai tên thuốc, và vì sao "chỉ cần đưa danh sách tên thuốc vào" lại không giải quyết xong vấn đề.
3. **Thiết kế thí nghiệm**: cho một model tiếng Việt bất kỳ và một danh sách 5.000 thuật ngữ y khoa, mô tả 5 thí nghiệm bạn sẽ chạy theo thứ tự, và với mỗi thí nghiệm nêu kết quả nào sẽ khiến bạn **đổi hướng**.

---

*Hết knowledge base. Quay lại mục lục → [`README.md`](README.md)*
