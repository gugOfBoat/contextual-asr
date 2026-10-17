# Chương 7 — Cơ chế tiêm ngữ cảnh: công thức và pseudo-code

> **Mục tiêu chương:** đủ chi tiết để bạn **tự cài lại** từng cơ chế, và quan trọng hơn — biết mỗi cơ chế **hỏng ở đâu**. Công thức của CLAS, TCPGen, Dynamic Vocabulary và trie-biasing trong chương này đã được **đối chiếu với bản full-text gốc** (ar5iv), không chép theo trí nhớ.
>
> Chương này giả định bạn đã nắm chương 3 (kiến trúc), 5 (giải mã) và 6 (bản đồ 7 điểm tiêm).

---

## 7.1 Khung chung: mọi thứ đều là một cái cổng

Nhắc lại §6.3. Đặt:
- $C = \{c_1,\dots,c_N\}$: danh sách cụm bias; mỗi $c_n$ là chuỗi token $(c_{n,1},\dots,c_{n,L_n})$.
- $s_t$: trạng thái model tại bước $t$ (decoder state, hoặc encoder frame).
- $P_{mdl}$: phân phối gốc của model trên $\mathcal{V}$.

Mọi cơ chế đều tạo ra một cặp (**phân phối ngữ cảnh** $P_{ctx}$, **cổng** $g$), rồi hợp nhất:

$$P(y_t \mid \cdot) = (1-g_t)\,P_{mdl}(y_t) + g_t\,P_{ctx}(y_t) \qquad \text{(hợp nhất kiểu nội suy)}$$

hoặc hợp nhất **cộng trong miền log** (kiểu shallow fusion):

$$\log P(y_t) \propto \log P_{mdl}(y_t) + \alpha \cdot b(y_t, s_t)$$

Khác biệt giữa hai kiểu:
- **Nội suy** giữ tổng xác suất bằng 1, học được $g$, mượt hơn → dùng trong các cơ chế neural.
- **Cộng log** đơn giản, không cần train, nhưng $\alpha$ phải chỉnh tay và dễ over-biasing.

Bảy mục tiếp theo là bảy cách cụ thể hoá $(P_{ctx}, g)$.

---

## 7.2 Cơ chế ①② — Boosting lúc giải mã (họ 1)

### Dạng cơ bản

$$b(y_t, s_t) = \begin{cases} +\alpha & \text{nếu } y_t \text{ đi tiếp được trong trie từ trạng thái } s_t \\ +\alpha + \alpha_{\text{end}} & \text{nếu vừa hoàn tất một cụm} \\ -\text{(tổng đã cộng)} & \text{nếu trượt khỏi trie (rollback)} \end{cases}$$

Chi tiết cài đặt và pseudo-code ở chương 5 §5.6. Ba biến thể đáng biết:

**(a) Context graph trong k2/icefall & WeNet.** Trie + failure link kiểu Aho–Corasick, chạy song song với beam search. Trạng thái trie được lưu **trong mỗi giả thuyết** của beam — nghĩa là cùng một prefix văn bản có thể ứng với các trạng thái trie khác nhau, phải quản lý cẩn thận.

**(b) CTC Word Spotter (CTC-WS, Interspeech 2024).** Khác về triết lý: không boost trong beam mà **dò cụm trước**:

```
1. Chạy CTC greedy → chuỗi frame-level (giữ cả blank & xác suất)
2. Duyệt chuỗi đó bằng trie của bias list, chấm điểm mỗi lần khớp
   (khớp "mềm": cho phép lệch vài frame/token)
3. Nếu điểm vượt ngưỡng → ghép cụm vào hypothesis, thay cho đoạn tương ứng
```
Ưu: cực nhanh (không cần beam search rộng), không train lại, đã có trong NeMo. Nhược: phụ thuộc CTC greedy có sinh ra cái gì đó "gần đúng" hay không — nếu âm học lệch hoàn toàn thì không có gì để khớp.

**(c) TurboBias / GPU phrase-boosting.** Cùng ý tưởng cây tiền tố nhưng cài đặt **song song trên GPU**, dùng chung cho CTC + RNN-T + AED, xử lý 20.000 cụm với chi phí thêm 2–5% RTFx (báo cáo `01` §1.1). Đây hiện là baseline ký hiệu mạnh nhất — nếu bạn đề xuất phương pháp neural, **phải** so với loại này, không phải với shallow fusion cẩu thả.

### Chế độ hỏng
- $\alpha$ lớn → over-biasing (chèn thực thể vào câu không có nó).
- Cụm ngắn (1–2 token) rất nguy hiểm: dễ khớp nhầm khắp nơi. Thực tế thường **lọc bỏ cụm quá ngắn** hoặc giảm $\alpha$ theo độ dài.
- Không cứu được khi acoustic model không sinh ra ứng viên gần đúng (nguyên nhân gốc số 2, §6.2).

---

## 7.3 Cơ chế ③ — CLAS: attention vào danh sách bias (họ 2, bài khai sinh)

*Pundak et al., SLT 2018, "Deep Context: End-to-End Contextual Speech Recognition" (arXiv 1808.02480). Công thức dưới đây đối chiếu từ bản full-text.*

### Kiến trúc

```
mỗi cụm z_i (chuỗi subword) ──► bias-encoder (LSTM nhiều tầng) ──► lấy TRẠNG THÁI CUỐI ──► h^z_i ∈ R^d
                              thêm một vector học được h^z_0 = h^z_nb  ("no-bias")
```

Tại mỗi bước decode $t$, với trạng thái decoder $d_t$:

$$u^z_{it} = (v^z)^\top \tanh\!\big(W^z_h h^z_i + W^z_d d_t + b^z_a\big)$$

$$\alpha^z_t = \mathrm{softmax}(u^z_t), \qquad c^z_t = \sum_i \alpha^z_{it}\, h^z_i$$

Nối context vector của audio và của bias, rồi tính phân phối đầu ra:

$$c_t = [\,c^x_t \,;\, c^z_t\,], \qquad P(y_t \mid h^x, y_{<t}) = \mathrm{softmax}\big(W_s [\,c_t ; d_t\,] + b_s\big)$$

### Ba chi tiết thiết kế phải hiểu

1. **`<no-bias>` là bắt buộc.** Nếu không có, softmax buộc phải phân bổ trọng số cho *một* cụm nào đó ngay cả khi không cụm nào liên quan → biasing luôn bật → over-biasing kinh niên. $h^z_{nb}$ chính là **cổng $g$** của CLAS, dưới dạng ngầm: khi $\alpha^z_{0t} \approx 1$ thì $c^z_t$ ≈ vector "trung tính".
2. **Cụm được nén thành MỘT vector** (trạng thái cuối LSTM). Đây vừa là điểm mạnh (xử lý được cụm chưa từng thấy — OOV lúc inference) vừa là điểm yếu (mất thông tin thứ tự token bên trong cụm, và attention trên $N$ vector loãng dần khi $N$ tăng).
3. **Cách lấy mẫu bias list lúc huấn luyện** (rất quan trọng, xem thêm §7.10): danh sách được **sinh ngẫu nhiên theo từng batch** — chọn $k \sim U[1, N_{phrases}]$ cụm, mỗi cụm là một n-gram lấy từ reference với bậc $n \sim U[1, N_{order}]$, và một tỉ lệ câu **không** có cụm nào (để model học "im lặng"). Trong chuỗi đích, chèn token `</bias>` sau cụm khớp để dạy model biết mình vừa hoàn tất một cụm.

### Chế độ hỏng
- $N$ lớn → attention loãng, hiệu quả sụp. Đây là giới hạn cố hữu của mọi cơ chế attention-trên-danh-sách.
- Cần dữ liệu huấn luyện có thực thể; nếu lấy mẫu n-gram ngẫu nhiên thì phân bố "cụm bias" lúc train không giống thực thể thật lúc test.

---

## 7.4 Cơ chế ④ — Deep biasing bằng vector trie (Le et al. 2021) và các biến thể transducer

*Le et al., Interspeech 2021 (arXiv 2104.02194) — cũng chính là bài định nghĩa B-WER/U-WER.*

Khác CLAS ở chỗ: không nén cả cụm thành một vector, mà mã hoá **trạng thái trie hiện tại**.

Tại mỗi bước, từ nút trie hiện tại, tạo **vector nhị phân**:

$$h^{bias} = [\,h^{start} \,;\, h^{cont}\,] \in \{0,1\}^{2D}$$

- $h^{start}_k = 1$ nếu token $k$ có thể **bắt đầu** một cụm bias nào đó.
- $h^{cont}_k = 1$ nếu token $k$ có thể **đi tiếp** cụm đang dở.

Rồi chiếu vào model:

$$h^{bias}_{\text{RNNT}} = f^{\text{RNNT}}_{bias}(h^{bias}), \qquad h^{bias}_{\text{LM}} = f^{\text{LM}}_{bias}(h^{bias})$$

và cộng/nối vào prediction network (hoặc vào NNLM dùng cho shallow fusion). Kết hợp thêm shallow fusion WFST lúc decode → đây là **cấu hình "deep + shallow"** kinh điển, và bài học lớn của paper: **hai tầng bổ sung cho nhau**, dùng cả hai tốt hơn hẳn dùng một.

### Biến thể quan trọng

| Tên | Ý chính | Vì sao đáng nhớ |
|---|---|---|
| **CATT** (ASRU 2021) | cross-attention vào bias embedding, query lấy từ **cả audio encoder lẫn label encoder** | đưa deep biasing vào **transducer streaming** |
| **Contextual Adapter** (ICASSP 2022) | chỉ train adapter nhỏ, **đóng băng model gốc** | quan trọng nhất về mặt thực tiễn: không phá model đã có |
| **NAM** (ICASSP 2022) | bộ nhớ liên kết key–value: key = embedding cụm, value = vector "đẩy"; truy xuất bằng tích vô hướng rồi **cộng thẳng vào biểu diễn** | thích nghi nhanh trên thiết bị; **Deferred NAM** chỉ encode top-K để giảm latency |
| **Intermediate biasing loss** (IS 2024) | thêm giám sát biasing ở **tầng giữa** encoder | ép thông tin ngữ cảnh vào sớm thay vì chỉ ở output |
| **Guided attention** (ICASSP 2024) | loss phụ dạy attention chỉ đúng cụm | trực tiếp chữa bệnh "cổng mở sai chỗ" |

---

## 7.5 Cơ chế ⑤ — TCPGen: nội suy phân phối bằng con trỏ (họ 3)

*Sun et al., ASRU 2021 (arXiv 2109.00627). Công thức dưới đây trích từ bản full-text.*

Đây là cơ chế **đẹp nhất về mặt toán** trong toàn bộ lĩnh vực, và là mẫu để hiểu mọi phương pháp copy/pointer.

### Bước 1 — thu hẹp không gian bằng trie

Tại bước $i$, từ nút trie hiện tại, ta có tập token hợp lệ $\mathcal{Y}^{trie}_i \subset \mathcal{V}$ (thường chỉ vài token), cộng thêm ký hiệu **OOL** (out-of-list) nghĩa là "không nhánh nào phù hợp".

### Bước 2 — attention có mặt nạ trên tập đó

$$\mathbf{q}_i = W^q_c \mathbf{c}_i + W^q_y \mathbf{y}_{i-1}, \qquad \mathbf{k}_j = W^K \mathbf{y}_j, \qquad \mathbf{v}_j = W^V \mathbf{y}_j$$

($\mathbf{c}_i$ = context vector từ cross-attention audio; với RNN-T thì thay bằng $\mathbf{h}^{enc}_t$.)

$$P^{ptr}(y_i \mid y_{1:i-1}, x) = \mathrm{Softmax}\Big(\mathrm{Mask}\big(\mathbf{q}_i \mathbf{K}^\top / \sqrt{d}\big)\Big)$$

$$\mathbf{h}^{ptr}_i = \sum_j P^{ptr}(y_i = j)\,\mathbf{v}_j$$

**Mask** chính là ràng buộc trie: mọi token ngoài $\mathcal{Y}^{trie}_i$ bị đặt $-\infty$. Đây là điểm mấu chốt — không gian tìm kiếm của con trỏ chỉ gồm vài token thay vì cả từ vựng.

### Bước 3 — cổng và hợp nhất

$$P^{gen} = \sigma\big(W^{gen}[\mathbf{h}^{dec}; \mathbf{h}^{ptr}]\big), \qquad \hat{P}^{gen} = P^{gen}\big(1 - P^{ptr}(\text{OOL})\big)$$

$$\boxed{\;P(y_i) = P^{mdl}(y_i)\,\big(1 - \hat{P}^{gen}\big) \;+\; P^{ptr}(y_i)\,\hat{P}^{gen}\;}$$

### Vì sao thiết kế này thông minh

- $P^{gen}$ là **cổng học được** — model tự quyết định khi nào tin con trỏ. Không phải chỉnh tay như $\alpha$.
- Nhân với $(1 - P^{ptr}(\text{OOL}))$ là chi tiết tinh tế: nếu con trỏ tự thấy "chẳng nhánh nào hợp", nó **tự tắt cổng**. Tức cơ chế có **hai** tầng an toàn: một tầng ký hiệu (trie mask) và một tầng học được (gate).
- Trie mask giải quyết nguyên nhân gốc số 3 (§6.2): sau khi vào nhánh, xác suất đi tiếp đúng gần như bằng 1.

### Họ mở rộng
- **+ GNN encoding** (IS 2022, TASLP): mã hoá nút trie bằng graph neural net → nút "biết" về cây con phía sau mình, quyết định sớm tốt hơn.
- **MBWE**: huấn luyện tối thiểu hoá trực tiếp lỗi trên từ bias, thay vì cross-entropy chung.
- **CopyNE** (ACL 2024): copy **cả thực thể một lần** thay vì từng token → không còn "nửa thực thể".
- **SeACo-Paraformer** (ICASSP 2024): áp cho model **non-autoregressive**, đã tích hợp trong FunASR → dùng được ngay, là baseline thực dụng rất tốt cho tiếng Trung/Việt.

### Chế độ hỏng
- Phụ thuộc tokenizer (chương 4 §4.4): trie dựng sai tokenizer → vô hiệu, im lặng.
- Trie quá lớn → mask không còn thu hẹp được nhiều → suy biến về attention thường.
- Cụm dài dễ đứt giữa chừng nếu một bước bị đẩy khỏi beam.

---

## 7.6 Cơ chế ⑥ — Dynamic Vocabulary: thêm token mới lúc suy luận (họ 4)

*Sudo et al., Interspeech 2024 (arXiv 2405.13344). Công thức đối chiếu từ bản full-text.*

### Ý tưởng

Thay vì bắt decoder đi 8 bước để ghép `paracetamol`, **thêm một token duy nhất** `<paracetamol>` vào từ vựng, chỉ tồn tại trong phiên suy luận này.

$$\mathcal{V}^b = \{\langle b_1\rangle, \dots, \langle b_N\rangle\}, \qquad \mathcal{V}_{\text{total}} = \mathcal{V} \cup \mathcal{V}^b \quad (K + N \text{ token})$$

### Sinh embedding cho token mới

$$\mathbf{G} = \mathrm{TransformerEnc}\big(\mathrm{Embedding}(\mathbf{B})\big), \qquad \mathbf{V} = \text{mean-pool}(\mathbf{G}) = [\mathbf{v}_1,\dots,\mathbf{v}_N]$$

$\mathbf{v}_n$ vừa đóng vai **embedding đầu vào** vừa đóng vai **vector trọng số ở lớp output** cho token $\langle b_n\rangle$.

### Mở rộng lớp output

$$\boldsymbol{\alpha}^n = \mathrm{Linear}(\mathbf{u}'_s), \qquad \boldsymbol{\alpha}^b = \frac{\mathrm{Linear}(\mathbf{u}'_s)\cdot \mathrm{Linear}(\mathbf{V})^\top}{\sqrt{d}}$$

$$P(\cdot) = \mathrm{Softmax}\big(\mathrm{Concat}(\boldsymbol{\alpha}^n, \boldsymbol{\alpha}^b)\big)$$

Nghĩa là: điểm cho token thường tính như cũ bằng ma trận softmax; điểm cho token bias tính bằng **tích vô hướng giữa trạng thái decoder và embedding cụm** — đúng kiểu "softmax với lớp output động".

### Embedding đầu vào ở bước sau

Khi bước trước sinh ra một bias token, đầu vào decoder lấy chính vector cụm:

$$\text{emb}(y'_{s-1}) = \begin{cases} \mathrm{Extract}(\mathbf{V},\,y'_{s-1}) & \text{nếu } y'_{s-1} \in \mathcal{V}^b \\ \mathrm{Embedding}(y'_{s-1}) & \text{ngược lại} \end{cases}$$

Sau khi decode xong, thay `<b_n>` bằng chuỗi chữ tương ứng.

### Ưu / nhược
**Ưu**: $L$ từ 8 xuống 1 (giải quyết trực diện nguyên nhân gốc 3); cụm dài không còn khó hơn cụm ngắn; đo lường sạch (một token = một cụm, dễ tính precision/recall thực thể).
**Nhược**: chất lượng hoàn toàn phụ thuộc vào việc $\mathbf{v}_n$ có **khớp với biểu diễn âm học** hay không — mà $\mathbf{v}_n$ sinh từ **chữ viết**. → Khi chính tả ≠ phát âm, cơ chế này mất hiệu lực **âm thầm**. Đây chính là giả thuyết giải thích kết quả ViMedCSS (36.31 → 36.29) trong báo cáo `00`, và là thí nghiệm chẩn đoán đáng làm nhất của bạn (chương 2 §2.8).

**Biến thể**: DYNAC (non-autoregressive, nhanh hơn), OWSM-Biasing (gắn lên model kiểu Whisper mở), Dynamic Vocabulary Prediction & Activation (chuyển sang encoder tự dự đoán cụm nào có mặt rồi mới kích hoạt — tức là tự sinh cổng).

---

## 7.7 Cơ chế ⑦ — Prompt và Speech-LLM (họ 5)

### Ba mức can thiệp

**Mức 1 — prompt thuần.**
```
Các thuật ngữ có thể xuất hiện: paracetamol, amoxicillin, ibuprofen.
Hãy chép lại chính xác đoạn ghi âm sau: <speech>
```
Với Whisper là `initial_prompt` (chèn sau `<|startofprev|>` — chương 3 §3.5). Miễn phí, nhưng:
- **Sập khi $N \gtrsim 500$–1000** (số đo ở báo cáo `01` §1.5).
- Prompt dài lệch phân bố → Whisper hallucinate hoặc lặp lại prompt.
- Bị giới hạn cửa sổ 448 token.

**Mức 2 — can thiệp logit / constrained decoding.**
Không thay prompt mà cộng thẳng vào logit của LLM khi token nằm trong cụm bias (giống §7.2 nhưng ở tầng LLM), hoặc ràng buộc decode để chuỗi sinh ra phải nằm trong tập cho phép. Ổn định hơn prompt, và **kiểm soát được** — nhưng cần truy cập logit (không dùng được với API đóng).

**Mức 3 — huấn luyện để model biết dùng ngữ cảnh.**
Đây là chỗ nghiên cứu 2025–2026 tập trung, vì hai vấn đề ở §3.7:
- **Instruction tuning với ngữ cảnh nhiễu**: đưa vào cả danh sách có đáp án lẫn danh sách không có, cả distractor đồng âm.
- **DPO/GRPO**: thay vì cross-entropy, tối ưu theo **preference/reward** — thưởng khi model dùng đúng ngữ cảnh, phạt khi copy bừa. Xử lý trực tiếp *contextual exposure bias*.

### Vì sao prompt sập — giải thích cơ chế (quan trọng để viết phần thảo luận trong luận văn)

1. **Loãng attention**: 1000 hotword ≈ 3000–5000 token prompt; tín hiệu audio chỉ ~100–200 token. Tỉ lệ tín hiệu/nhiễu trong chuỗi đầu vào rơi thẳng đứng.
2. **Thiên lệch copy**: LLM được huấn luyện để dùng lại nội dung trong ngữ cảnh → với danh sách dài, luôn tồn tại một mục "nghe na ná" → chèn bừa.
3. **Lệch phân bố huấn luyện**: model chưa từng thấy chuỗi kiểu danh sách 1000 mục trong dữ liệu ASR.

Cả ba đều **không** sửa được bằng cách viết prompt khéo hơn. Phải giảm $N$ (retrieval) hoặc đổi cơ chế.

---

## 7.8 Cơ chế ⓪ — Retrieval: lọc trước khi tiêm (họ 6)

Nền toán đã ở chương 2 §2.7. Ở đây là phần hệ thống.

```
                 ┌── offline ────────────────────────────────┐
                 │ 200k cụm ─f_text─► T ∈ R^{200k×d} ─► FAISS │
                 └───────────────────────────────────────────┘
audio ─f_audio─► a ─► top-k (cosine) ─► 10–50 cụm ─► [bất kỳ cơ chế nào ở §7.2–7.7]
```

### Bốn quyết định thiết kế

1. **Truy vấn ở mức nào?** Cả câu (đơn giản, nhưng loãng khi câu dài/nhiều thực thể) hay theo cửa sổ trượt (chính xác hơn, đắt hơn). GLCLAP dùng cả global lẫn local vì lý do này.
2. **Negative lấy thế nào?** Đây là chỗ quyết định chất lượng. In-batch negative là tối thiểu; **hard negative đồng âm** mới là thứ tạo khác biệt. BR-ASR dùng curriculum tăng dần độ khó.
3. **Có tầng phonetic không?** Với tiếng Việt: gần như bắt buộc (§2.8).
4. **Cascade**: tầng 1 rẻ lấy top-1000 (ANN), tầng 2 đắt hơn xếp hạng lại top-50 (cross-encoder). Tiêu chuẩn trong IR, còn ít được khai thác trong contextual ASR ⚠️ — có thể là chỗ trống.

### Ràng buộc cứng

$$\text{B-WER khả thi} \;\ge\; \text{hàm của } (1 - \text{Recall@}k)$$

Nếu retriever miss thì mọi thứ phía sau vô nghĩa. **Luôn báo cáo Recall@k song song với WER** — nếu paper nào không làm, đó là lỗ hổng đánh giá.

---

## 7.9 Cơ chế ⑧ — Sửa lỗi sau giải mã (họ 7)

```
audio ─► ASR (đóng, không sửa được) ─► "bệnh nhân dùng pa ra xê ta môn"
                                              │
                     bias list ──────────────►│ model sửa lỗi
                                              ▼
                                    "bệnh nhân dùng paracetamol"
```

Hai cách sinh ứng viên sửa:
- **Khoảng cách âm vị**: chuyển cả hypothesis và bias list sang chuỗi âm vị, tìm đoạn có khoảng cách sửa lỗi nhỏ → thay thế. Rẻ, kiểm soát được, và **hợp với tiếng Việt** vì lỗi thường là lỗi phát âm hoá.
- **LLM revision**: đưa transcript + bias list cho LLM, bảo nó sửa. Mạnh nhưng dễ hallucinate — bắt buộc phải ràng buộc "chỉ được thay bằng mục có trong danh sách" và đo cả U-WER.

**Ưu**: dùng được với API đóng (Whisper API, Google STT) — đây là lý do thực dụng lớn.
**Nhược**: không cứu được thông tin đã mất ở tầng âm học; thêm độ trễ; trần trên bị chặn bởi oracle n-best (chương 5 §5.7).

---

## 7.10 ★ Huấn luyện module biasing: chỗ quyết định thành bại

Phần này ít được viết rõ trong paper nhưng quyết định kết quả nhiều hơn kiến trúc.

### Cách dựng bias list lúc huấn luyện

```python
# Sơ đồ chuẩn (theo tinh thần CLAS / Le et al.)
def make_train_bias_list(ref_tokens, rare_vocab, N, p_no_bias=0.3, n_distr=(0, 100)):
    positives = [w for w in extract_ngrams(ref_tokens) if w in rare_vocab]
    if random.random() < p_no_bias:
        positives = []                       # câu KHÔNG có cụm nào → dạy model im lặng
    k = random.randint(0, len(positives))
    positives = random.sample(positives, k)
    n = random.randint(*n_distr)
    distractors = random.sample(rare_vocab, n)          # ← cực kỳ quan trọng
    return shuffle(positives + distractors)
```

Bốn siêu tham số ảnh hưởng lớn:

| Tham số | Quá thấp | Quá cao |
|---|---|---|
| Tỉ lệ câu **không** có cụm | model luôn bật biasing → over-biasing | model học phớt lờ ngữ cảnh |
| Số **distractor** | không chịu nổi danh sách dài lúc test | biasing yếu đi trên danh sách ngắn |
| Độ dài cụm lấy mẫu | không khớp thực thể thật | nhiễu |
| **Context dropout** (thỉnh thoảng bỏ hẳn $C$) | model phụ thuộc mù quáng vào $C$ | không học dùng $C$ |

### ★ Nghịch lý distractor

Đây là một trong những vấn đề mở đáng giá nhất (báo cáo `01` §1.2 và §5.2):

> Trộn **nhiều** distractor lúc huấn luyện giúp model chịu được danh sách dài lúc test, **nhưng làm giảm hiệu quả trên danh sách ngắn**. Trộn **ít** thì ngược lại. Chưa ai giải sạch.

Cách nghĩ về nó: model đang học một **ngưỡng tin cậy ngầm**. Nhiều distractor → ngưỡng cao → cẩn thận → bỏ sót. Ít distractor → ngưỡng thấp → hăng hái → chèn bừa. Vì ngưỡng này bị "nướng" cứng vào trọng số lúc huấn luyện, nó không thích nghi được với $N$ lúc test.

**Các hướng tấn công khả dĩ** (đây là chất liệu đề tài):
- Cho model **biết $N$** như một đầu vào (điều kiện hoá theo kích thước danh sách) → cổng tự hiệu chỉnh.
- Curriculum theo $N$ trong lúc huấn luyện.
- Hiệu chỉnh cổng **sau huấn luyện** (calibration trên tập dev theo từng $N$) — rẻ và ít ai làm ⚠️.
- Tách hẳn quyết định "có thực thể hay không" thành một module riêng có ngưỡng chỉnh được lúc chạy.

### Dữ liệu tổng hợp
Khi không có audio chứa thực thể hiếm: dùng **TTS** sinh audio cho câu chứa thực thể (text injection). Cạm bẫy đã nêu ở chương 1 §1.7: audio TTS quá sạch → module biasing học trên phân bố sai. Bắt buộc augment (nhiễu, RIR, codec) và **luôn đo trên audio thật**.

---

## 7.11 Ghép tầng: cấu hình "thắng" hiện nay

```
                 ┌─────────────────────────────────────────────────────┐
 danh mục 200k ─►│ (0) RETRIEVER: contrastive audio↔text (+ âm vị)      │─► top-20
                 └─────────────────────────────────────────────────────┘
                                          │
      ┌───────────────────────────────────┴─────────────────────────────┐
      │ (2/3/4) DEEP BIASING: CLAS/TCPGen/dynamic vocab trên top-20      │
      └───────────────────────────────────┬─────────────────────────────┘
                                          │
      ┌───────────────────────────────────┴─────────────────────────────┐
      │ (6) SHALLOW FUSION / context graph lúc beam search (α nhỏ)       │
      └───────────────────────────────────┬─────────────────────────────┘
                                          │
      ┌───────────────────────────────────┴─────────────────────────────┐
      │ (7) POST-HOC: sửa theo khoảng cách âm vị, chỉ thay bằng mục trong│
      │     danh sách đã retrieve (an toàn, kiểm soát được)              │
      └─────────────────────────────────────────────────────────────────┘
```

Lý do thứ tự này hiệu quả: mỗi tầng sửa một **nguyên nhân gốc khác nhau** (§6.2). Retrieval sửa vấn đề scale; deep biasing sửa prior; shallow fusion cứu các cụm mà deep biasing bỏ lỡ ở tầng tìm kiếm; post-hoc quét nốt lỗi chính tả/phát âm còn lại.

**Nhưng cẩn thận với luận văn:** ghép 4 tầng thì rất khó biết cái gì đóng góp bao nhiêu. Phải có **ablation từng tầng**, nếu không đóng góp khoa học sẽ mờ.

---

## 7.12 Bảng tra nhanh: cơ chế ↔ chế độ hỏng

| Cơ chế | Hỏng khi | Triệu chứng nhận biết |
|---|---|---|
| Shallow fusion / context graph | $\alpha$ lớn; cụm ngắn; âm học lệch hoàn toàn | U-WER tăng; xuất hiện thực thể ở chỗ vô lý |
| CTC-WS | CTC greedy không sinh gì gần đúng | không bắt được, nhưng U-WER không tăng |
| CLAS / CATT / adapter | $N$ lớn; phân bố cụm train ≠ test | B-WER cải thiện ít dần khi tăng $N$ |
| TCPGen / trie | tokenizer lệch; trie quá lớn | biasing "chạy" mà số không đổi ⇒ nghi tokenizer |
| Dynamic vocabulary | chính tả ≠ phát âm; embedding cụm kém | **cải thiện ~0** dù cấu hình đúng ⇒ nghi khoá tra cứu |
| Prompt / Speech-LLM | $N > 500$; prompt lệch phân bố | WER **bùng nổ**, hallucination, lặp |
| Retrieval | Recall@k thấp; đồng âm | B-WER chững lại dù biasing mạnh |
| Post-hoc | thực thể không có trong n-best; LLM bịa | U-WER tăng, xuất hiện "sửa quá tay" |

Bảng này dùng theo chiều **ngược**: quan sát triệu chứng → suy ra nguyên nhân → biết phải đo cái gì tiếp theo. Đó là kỹ năng thực nghiệm quan trọng nhất trong lĩnh vực này.

---

## Bài tập tự kiểm tra

1. Viết lại công thức hợp nhất của TCPGen bằng ký hiệu của riêng bạn, rồi giải thích **bằng lời** vai trò của $\big(1 - P^{ptr}(\text{OOL})\big)$. Điều gì xảy ra nếu bỏ nhân tử này?
2. So sánh CLAS và Dynamic Vocabulary theo 4 tiêu chí: nơi tiêm, độ dài chuỗi cần sinh, phụ thuộc tokenizer, khả năng chịu $N$ lớn.
3. Bạn cài context-graph boosting nhưng quên rollback. Mô tả chính xác loại lỗi sẽ xuất hiện trong transcript, và cách phát hiện nó bằng metric.
4. Thiết kế một thí nghiệm 1 ngày để kiểm chứng giả thuyết "Dynamic Vocabulary thất bại trên ViMedCSS vì lệch chính tả–phát âm". Nêu rõ: biến can thiệp, nhóm đối chứng, metric, và kết quả nào sẽ **bác bỏ** giả thuyết.
5. Đề xuất một cách tấn công nghịch lý distractor (§7.10) và nêu thí nghiệm tối thiểu để chứng minh nó hoạt động.

---

*Chương tiếp → [`08-do-luong-va-thuc-nghiem.md`](08-do-luong-va-thuc-nghiem.md): có cơ chế rồi thì phải đo. Và trong lĩnh vực này, **đo sai là cách phổ biến nhất để tự lừa mình**.*
