# Chương 3 — Bản chất các kiến trúc ASR

> **Mục tiêu chương:** hiểu **toán** của CTC, RNN-T, AED, NAR và Speech-LLM đủ sâu để biết mỗi loại **mở ra cửa nào cho ngữ cảnh chui vào**. Đây là chương bản lề: nếu chương 7 (cơ chế biasing) có chỗ nào khó hiểu, gần như chắc chắn là do chưa chắc chương này.

---

## 3.1 Bài toán, phát biểu chính xác

Cho chuỗi đặc trưng $x = (x_1,\dots,x_T)$, tìm chuỗi token $y = (y_1,\dots,y_U)$:

$$\hat{y} = \arg\max_{y} P(y \mid x)$$

Ba khó khăn cấu trúc, và **mọi kiến trúc ASR chẳng qua là một cách khác nhau để xử lý ba khó khăn này**:

1. **Độ dài không khớp**: $T \gg U$ (25–100 frame/giây vs ~3 token/giây), và **không biết trước** frame nào ứng với token nào → bài toán **căn chỉnh (alignment)**.
2. **Không gian đầu ra khổng lồ**: $|\mathcal{V}|^U$ chuỗi khả dĩ → phải tìm kiếm gần đúng (beam search).
3. **Phụ thuộc theo thứ tự**: audio và text đều **đơn điệu** (monotonic) — token thứ 5 không thể phát ra trước token thứ 4. Ràng buộc này là món quà (khác dịch máy), và các kiến trúc khai thác nó khác nhau.

### Góc nhìn noisy channel — nguồn gốc của mọi chuyện

Thời trước (và vẫn đúng về mặt khái niệm):

$$\hat{y} = \arg\max_y \underbrace{P(x\mid y)}_{\text{acoustic model}} \cdot \underbrace{P(y)}_{\text{language model}}$$

Hai thành phần **tách rời**: AM học từ audio có nhãn, LM học từ **văn bản thuần** (rẻ, vô hạn). Muốn thêm từ mới vào hệ thống? Chỉ cần sửa LM và từ điển phát âm — **không đụng vào AM**.

Hệ end-to-end mô hình hoá thẳng $P(y|x)$, gộp AM và LM làm một. Được: đơn giản, mạnh hơn khi đủ dữ liệu. Mất: **không còn chỗ nào để "cắm" kiến thức về từ mới**. Phần LM giờ nằm ẩn bên trong trọng số — gọi là **internal LM (ILM)**.

> **Câu này là gốc rễ của toàn bộ contextual ASR:**
> *Contextual ASR = tìm lại cái cửa mà end-to-end đã đóng lại.*
>
> Mỗi họ phương pháp trong báo cáo `01` là một cách mở lại cửa đó ở một vị trí khác nhau.

---

## 3.2 Hybrid HMM–DNN (đủ để đọc paper cũ và hiểu WFST)

Không còn là SOTA, nhưng **phải biết** vì hai lý do: biasing bằng WFST/class-LM sinh ra từ đây, và nhiều khái niệm (lexicon, phone, alignment) vẫn dùng.

Chuỗi xử lý: $P(x|y)$ được phân rã qua **HMM trạng thái âm vị**, mỗi trạng thái có một phân phối phát xạ (GMM, sau này là DNN). Ghép mọi thứ bằng **WFST**:

$$H \circ C \circ L \circ G$$

- $H$: HMM (trạng thái → phone)
- $C$: context-dependency (triphone → phone)
- $L$: **lexicon** (phone → từ) ← *từ điển phát âm nằm ở đây*
- $G$: **grammar/LM** (từ → chuỗi từ) ← *mô hình ngôn ngữ nằm ở đây*

**Vì sao quan trọng với bạn:** thêm một từ mới trong hệ hybrid = thêm một dòng vào $L$ + cập nhật $G$. Không cần dữ liệu audio, không cần train. Đó là "tiêu chuẩn vàng" về khả năng tuỳ biến mà **mọi hệ E2E đến nay vẫn chưa lấy lại được trọn vẹn**. Class-based LM (thay `$CONTACT` bằng danh sách động lúc decode) là kỹ thuật biasing cổ điển nhất, và vẫn được dùng.

---

## 3.3 CTC — căn chỉnh bằng cách tổng trên mọi đường đi

### Ý tưởng

Encoder cho ra $T'$ vector, mỗi vector qua softmax thành phân phối trên $\mathcal{V} \cup \{\varnothing\}$ ($\varnothing$ = **blank**). Ta có một "đường đi" (path) $\pi = (\pi_1,\dots,\pi_{T'})$. Hàm gộp $\mathcal{B}$: **bỏ token lặp liên tiếp, rồi bỏ blank**.

$$\mathcal{B}(a\,a\,\varnothing\,a\,b\,\varnothing\,b) = a\,a\,b\,b \;\;? \quad\text{— không: } \to a\,a\,b\,b$$

Cụ thể: `a a ∅ a b ∅ b` → bỏ lặp: `a ∅ a b ∅ b` → bỏ blank: `a a b b`. **Blank chính là thứ cho phép biểu diễn ký tự lặp** (`ll` trong `hello`).

Xác suất của nhãn $y$ = tổng trên **mọi** đường đi gộp về $y$:

$$P_{\text{CTC}}(y\mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T'} p_t(\pi_t \mid x)$$

Số đường đi là hàm mũ, nhưng tính được bằng **quy hoạch động** (forward–backward), giống HMM. Gọi $l'$ là $y$ có chèn blank xen kẽ ($l' = \varnothing\, y_1 \varnothing\, y_2 \dots \varnothing$, độ dài $2U+1$):

$$\alpha_t(s) = p_t(l'_s)\cdot\Big[\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \underbrace{\alpha_{t-1}(s-2)}_{\text{chỉ khi } l'_s \neq \varnothing,\, l'_s \neq l'_{s-2}}\Big]$$

$$P(y|x) = \alpha_{T'}(2U+1) + \alpha_{T'}(2U)$$

Loss = $-\log P(y|x)$, đạo hàm được, train bằng SGD. **Không cần alignment gán sẵn** — đó là toàn bộ đóng góp của CTC.

### Ba hệ quả phải nhớ

**(1) Giả định độc lập có điều kiện.** $p_t$ chỉ phụ thuộc $x$, **không** phụ thuộc token đã sinh trước đó. Nên:
$$P_{\text{CTC}}(y|x) \text{ không chứa mô hình ngôn ngữ.}$$
→ CTC **bắt buộc** phải ghép LM ngoài để mạnh. → **Tin tốt cho biasing**: không có ILM mạnh chống lại, nên shallow fusion/boosting trên CTC hiệu quả và sạch. Đây là lý do các phương pháp biasing "không cần train lại" (CTC-WS, WCTC-Biasing) đều làm trên CTC.

**(2) Hành vi "peaky".** CTC hội tụ về trạng thái phát blank ở gần như mọi frame, và bùng một **spike** nhọn tại đúng thời điểm token xuất hiện. Hệ quả: timestamp gần như miễn phí; và có thể dùng spike làm **tín hiệu kích hoạt biasing** (Spike-Triggered Contextual Biasing) hoặc để dò từ khoá (CTC word spotter).

**(3) Frame-synchronous.** Mỗi bước decode = một frame → streaming tự nhiên, latency thấp, chi phí decode $O(T')$.

### Giải mã CTC

- **Greedy**: lấy argmax mỗi frame rồi gộp. Rất nhanh, đủ tốt khi có encoder mạnh.
- **Prefix beam search**: gộp các đường đi cùng tiền tố (phải cộng dồn xác suất — vì nhiều path cho cùng một prefix), cho phép ghép LM n-gram và **hotword boosting** (thư viện `pyctcdecode`, `torchaudio` CTC decoder). Xem chương 5 §5.3 để có pseudo-code.

---

## 3.4 RNN-T / Transducer — chuẩn công nghiệp cho streaming

CTC thiếu mô hình ngôn ngữ. RNN-T (Graves 2012) vá đúng chỗ đó.

### Ba khối

```
        x ──► Encoder ────────► f_t ∈ R^d          (giống encoder CTC)
   y_<u ──► Prediction net ──► g_u ∈ R^d          (chỉ nhìn TEXT đã sinh — đây là internal LM)
                    │
                    ▼
            Joint:  z_{t,u} = W · tanh(W_f f_t + W_g g_u + b)
                    ▼
            P(k | t,u) = softmax(z_{t,u}),  k ∈ V ∪ {∅}
```

### Lưới alignment $T \times U$

Mỗi ô $(t,u)$ = "đã xử lý $t$ frame, đã sinh $u$ token". Từ mỗi ô có hai nước đi:

```
        u
        ↑
   U    ·  ·  ·  ·  ●   ← đích
        ·  ·  ·  ↑  ·
        ·  ·  ●→ ●  ·     ↑ (dọc) = phát ra một token, u+1, KHÔNG tiêu frame
        ·  ●→ ↑  ·  ·     → (ngang) = phát blank ∅, t+1, sang frame kế
   0    ●→ ·  ·  ·  ·
        0 ──────────► t   T
```

$$P_{\text{RNNT}}(y|x) = \sum_{\text{mọi đường từ }(0,0)\text{ tới }(T,U)} \prod \text{xác suất các nước đi}$$

Cũng tính bằng forward–backward:
$$\alpha(t,u) = \alpha(t-1,u)\,P(\varnothing\mid t-1,u) + \alpha(t,u-1)\,P(y_u \mid t, u-1)$$

### Vì sao đây là chuẩn công nghiệp

- **Streaming thật**: chỉ cần frame tới đâu decode tới đó (encoder streaming), không chờ hết câu.
- **Có ILM** (prediction net) → chính xác hơn CTC khi không có LM ngoài.
- Có thể phát nhiều token trên cùng một frame (đi dọc nhiều bước).

### Internal LM và ILME — kiến thức bắt buộc cho biasing

Prediction net + joint đóng vai trò một LM ẩn. Khi ghép LM ngoài bằng shallow fusion, ta **cộng chồng hai LM** → lệch. Cách sửa:

$$\text{score} = \log P_{\text{RNNT}}(y|x) + \lambda \log P_{\text{LM}}(y) - \mu \log P_{\text{ILM}}(y)$$

**ILME (Internal LM Estimation)**: ước lượng $P_{\text{ILM}}$ bằng cách **cho đầu vào encoder về 0** (hoặc bỏ nhánh $f_t$ trong joint) → phần còn lại chính là LM nội tại.

> **Liên hệ contextual ASR — điểm ít người nói ra:** ILM là thứ **chống lại** từ hiếm mạnh nhất (nó học prior tần suất từ dữ liệu huấn luyện). Trừ bớt ILM trước rồi mới boost hotword thường cho kết quả tốt hơn hẳn boost thô. Nếu bạn làm shallow-fusion baseline mà bỏ qua ILME, baseline của bạn yếu **một cách không công bằng** — reviewer sẽ bắt lỗi.

### Biến thể đáng biết

- **TDT (Token-and-Duration Transducer, NVIDIA Parakeet-TDT)**: joint dự đoán thêm **thời lượng** để nhảy nhiều frame một lúc → nhanh hơn nhiều lần.
- **Stateless prediction network**: thay LSTM bằng embedding của 1–2 token gần nhất → ILM yếu đi (dễ ghép LM ngoài hơn), nhanh hơn, gần như không mất chất lượng (dùng trong icefall/Zipformer).
- **Pruned RNN-T loss** (k2): giảm bộ nhớ của lưới $T\times U$ từ $O(TUV)$ xuống chấp nhận được — lý do icefall train được nhanh.

---

## 3.5 AED / Attention Encoder–Decoder — Whisper thuộc nhóm này

### Toán

$$P_{\text{AED}}(y|x) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, x)$$

Decoder tự hồi quy; mỗi bước **cross-attention** vào toàn bộ $h^{enc}$:

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V, \quad Q = W_Q s_u,\; K = W_K h^{enc},\; V = W_V h^{enc}$$

Huấn luyện bằng **teacher forcing** (đưa nhãn đúng làm đầu vào bước sau) + cross-entropy, thường có label smoothing.

### Ưu / nhược

**Ưu**: chất lượng cao nhất trong ba loại cổ điển; decoder là một LM đầy đủ nên văn bản trôi chảy; **dễ gắn task bằng token điều khiển**; nhìn được toàn cục.

**Nhược**:
- **Không đơn điệu về mặt cấu trúc** — attention có thể nhảy lung tung → **hallucination**, lặp vô hạn, bỏ đoạn. Đặc biệt trên im lặng/nhiễu.
- **Exposure bias**: lúc train luôn thấy tiền tố đúng, lúc test phải sống với tiền tố do chính nó sinh ra. → Một lỗi sớm kéo theo cả câu hỏng. *(Ghi nhớ khái niệm này: dòng contextual mới nhất dùng đúng thuật ngữ tương tự — "contextual exposure bias" — cho việc train với oracle context nhưng test với context nhiễu.)*
- Khó streaming (cần cả câu).

### Cấu trúc token của Whisper — phải nắm, vì `initial_prompt` sống ở đây

Whisper mã hoá **task bằng token đặc biệt** trong chuỗi decoder:

```
[<|startofprev|> <văn bản ngữ cảnh trước>]  <|startoftranscript|> <|vi|> <|transcribe|> <|notimestamps|>  văn bản...  <|endoftext|>
└──────────── phần "prompt" tuỳ chọn ─────┘ └──────────── phần điều khiển ─────────────┘
```

Cơ chế của `initial_prompt`: văn bản bạn đưa vào được đặt sau `<|startofprev|>`, nằm **trong cùng chuỗi decoder**. Nghĩa là:

- Nó **không** phải một cơ chế biasing riêng. Nó chỉ là **điều kiện hoá LM của decoder** trên một đoạn text đứng trước. Whisper vốn được huấn luyện để dùng đoạn đó như "phần transcript trước đó" của cùng một đoạn audio dài.
- Hệ quả 1: nó có tác dụng thật với thuật ngữ/văn phong — vì self-attention của decoder thấy các từ đó.
- Hệ quả 2: nó **rất dễ bị lạm dụng** — nếu prompt dài, lệch phân bố (một danh sách 500 từ không giống "transcript trước đó" chút nào), Whisper sẽ **bắt chước cả văn phong danh sách**, sinh hallucination, hoặc lặp lại prompt. Đây là lời giải thích cơ chế cho hiện tượng "prompt thuần sập khi danh sách dài" ở báo cáo `01` §1.5.
- Hệ quả 3: prompt chiếm chỗ trong cửa sổ 448 token của decoder.

### Hybrid CTC/Attention (ESPnet)

Train chung hai loss, decode chung điểm:
$$\mathcal{L} = \gamma\,\mathcal{L}_{\text{CTC}} + (1-\gamma)\,\mathcal{L}_{\text{AED}}, \qquad \text{score} = \gamma \log P_{\text{CTC}} + (1-\gamma)\log P_{\text{AED}}$$
CTC ép attention đơn điệu → **chống hallucination**, hội tụ nhanh hơn. Đây là công thức mặc định của ESPnet và là một baseline rất vững.

---

## 3.6 Non-autoregressive: Paraformer và CIF

Sinh **toàn bộ token song song** trong một bước → nhanh gấp nhiều lần.

Vấn đề: phải biết trước **có bao nhiêu token** và mỗi token ứng với vùng audio nào. **CIF (Continuous Integrate-and-Fire)** giải quyết:

```
mỗi frame encoder → dự đoán một "trọng lượng" w_t ∈ [0,1]
cộng dồn w_t theo thời gian; khi tổng vượt ngưỡng 1.0 → "bắn" (fire):
   phát ra một acoustic embedding = tổng có trọng số các frame kể từ lần bắn trước
   phần dư mang sang token kế
```

Kết quả: chuỗi $U$ embedding, mỗi cái ứng với một token, **có căn chỉnh thời gian rõ ràng**. Số lần bắn = số token dự đoán (train bằng loss đếm số token).

**Vì sao liên quan tới bạn:** CIF cho một điểm neo *một-embedding-cho-một-token* — rất tiện để gắn biasing (SeACo-Paraformer, DYNAC, và CLAR dùng CIF để căn chỉnh RAG trong Speech-LLM). Nếu bạn cần độ trễ thấp mà vẫn muốn biasing mạnh, đây là nhánh đáng xem.

---

## 3.7 Speech-LLM — kiến trúc đang định hình hiện tại

```
audio ─► Speech Encoder ─► downsample/pooling ─► Projector ─► [embedding audio]
                                                                    │
prompt text ─► Tokenizer ─► Embedding ────────────────────────►  [ghép chuỗi]
                                                                    ▼
                                                             LLM (decoder-only)
                                                                    ▼
                                                              transcript
```

- **Encoder**: Whisper-encoder hoặc FastConformer, đóng băng hoặc fine-tune nhẹ.
- **Projector**: linear / MLP / Q-Former / CIF. Nhiệm vụ: (a) đổi số chiều, (b) **giảm số frame** (1500 frame Whisper → ~100–200 "token audio", nếu không LLM sẽ ngộp).
- **LLM**: Qwen, Llama... đóng vai decoder. Thường LoRA thay vì full fine-tune.
- Huấn luyện thường 2 giai đoạn: (1) đóng băng cả encoder lẫn LLM, chỉ train projector để "căn chỉnh"; (2) mở LoRA, train theo hướng dẫn (instruction tuning).

**Với contextual ASR, đây là kiến trúc hấp dẫn nhất về nguyên tắc**: chỉ cần viết ngữ cảnh vào prompt bằng tiếng Việt/Anh, không cần cơ chế đặc biệt nào. Nhưng thực nghiệm cho hai kết luận đã ghi trong báo cáo `01` §1.5, và giờ bạn hiểu được **cơ chế** đằng sau:

1. **Sập khi danh sách dài** ($N \gtrsim 500$): attention của LLM trải trên hàng nghìn token prompt; tín hiệu âm học chỉ là một phần nhỏ trong chuỗi; và LLM vốn có xu hướng "copy" từ prompt → **over-biasing** hoặc nhiễu loạn hoàn toàn.
2. **Context-utilization gap**: model *nhận* prompt nhưng không thực sự *dùng*. Cơ chế: trong huấn luyện, phần lớn dữ liệu ASR **không có** ngữ cảnh hữu ích, nên model học được rằng cách an toàn nhất là **phớt lờ prompt** và chỉ dựa vào audio. Đây là một dạng shortcut learning cổ điển. → Đó cũng chính là lý do các nghiên cứu 2026 chuyển sang **DPO/GRPO**: dùng preference optimization để *thưởng* cho việc dùng đúng ngữ cảnh, thay vì hy vọng nó tự học.

---

## 3.8 Encoder: Conformer, Zipformer, và bài toán audio dài

**Conformer block** (macaron):

$$
\tilde{x} = x + \tfrac{1}{2}\text{FFN}(x); \quad
x' = \tilde{x} + \text{MHSA}(\tilde{x}); \quad
x'' = x' + \text{Conv}(x'); \quad
y = \text{LN}\!\left(x'' + \tfrac{1}{2}\text{FFN}(x'')\right)
$$

Trực giác: **self-attention bắt phụ thuộc xa** (ngữ cảnh câu), **convolution bắt cấu trúc cục bộ** (chuyển tiếp âm vị). Ghép hai cái = xương sống của ngành từ 2020.

- **Zipformer** (icefall): encoder nhiều tốc độ khung (U-Net theo thời gian), nhanh và tốt hơn Conformer ở cùng số tham số. Là baseline mạnh nhất trong k2/icefall — cũng là nơi có sẵn context-graph biasing.
- **E-Branchformer**: hai nhánh song song (attention / conv) rồi hợp nhất; mặc định trong nhiều recipe ESPnet mới.

**Streaming**: giới hạn tầm nhìn bằng **chunk + right-context** (chunked attention mask), hoặc attention nhân quả có bộ nhớ. Đánh đổi: right-context càng lớn → chính xác hơn nhưng độ trễ cao hơn.

**Audio dài (>30 s)**: Whisper cắt cửa sổ 30 s → mất/lặp ở biên. ChunkFormer (ICASSP 2025, nhóm Việt Nam) xử lý theo chunk có right-context và quản lý bộ nhớ để decode được audio rất dài trên một GPU — đáng dùng nếu dữ liệu của bạn là ghi âm cuộc họp/khám bệnh dài.

---

## 3.9 ★ Bảng tổng hợp: mỗi kiến trúc mở cửa nào cho ngữ cảnh

Đây là bảng để **quay lại tra** khi đọc chương 7.

| | **CTC** | **RNN-T** | **AED (Whisper)** | **NAR (Paraformer)** | **Speech-LLM** |
|---|---|---|---|---|---|
| Phân rã | $\prod_t p_t$, độc lập | $\sum_{\text{align}}$, có pred-net | $\prod_u P(y_u\|y_{<u},x)$ | song song + CIF | LLM tự hồi quy |
| Internal LM | ~không có | **có** (pred-net) | **mạnh** (decoder) | yếu | **rất mạnh** |
| Streaming | ✅ tự nhiên | ✅ tốt nhất | ❌ khó | ⚠️ chunk | ❌ |
| Timestamp | ✅ miễn phí (spike) | ✅ | ⚠️ phải học riêng | ✅ (CIF) | ❌ |
| Hallucination | rất ít | ít | **nhiều** | ít | **nhiều** |
| **Tiêm ngữ cảnh ở đâu** | biasing lúc decode (boost/WFST/CTC-WS); wildcard-CTC | bias vào **encoder** và/hoặc **prediction net**/joint (CATT, adapter, TCPGen-RNNT) | bias vào **decoder state** (CLAS), **prompt** (`<\|startofprev\|>`), **dynamic vocab** ở output | bias vào **acoustic embedding sau CIF** (SeACo, DYNAC) | **prompt**, logit-space, retrieval + prompt |
| Không train lại được không? | ✅ (họ 1) | ⚠️ một phần | ✅ (prompt/họ 1 nếu có logits) | ⚠️ | ✅ (prompt) |
| Chịu $N$ lớn | tốt (ký hiệu) | trung bình | kém | trung bình | **kém nhất** |

**Đọc bảng này theo một câu:** *càng nhiều internal LM thì càng khó ép, nhưng ép được thì càng mượt.* CTC dễ ép nhất vì nó "trống"; Speech-LLM khó ép nhất vì nó có ý kiến riêng rất mạnh.

---

## 3.10 Tổng kết chương

1. Mọi kiến trúc là một cách xử lý bài toán căn chỉnh: CTC tổng trên path, RNN-T tổng trên lưới, AED để attention tự lo, NAR đếm trước rồi bắn.
2. **Internal LM** là khái niệm trung tâm: nó khiến E2E mạnh, đồng thời khiến việc tiêm từ hiếm vào trở nên khó. ILME là công cụ để "gỡ" nó ra.
3. `initial_prompt` của Whisper **không phải** cơ chế biasing — nó là điều kiện hoá tiền tố. Hiểu điều này giải thích luôn vì sao nó sập với danh sách dài.
4. Bảng §3.9 là bản đồ để đọc mọi paper biasing: câu đầu tiên cần hỏi là *"họ tiêm vào chỗ nào?"*.

---

## Bài tập tự kiểm tra

1. Vì sao CTC cần token blank? Cho ví dụ một từ tiếng Việt mà thiếu blank sẽ không biểu diễn được.
2. Trong lưới RNN-T, đường đi "toàn đi ngang trước rồi mới đi dọc" tương ứng với hành vi gì của hệ thống? Vì sao nó tệ cho streaming?
3. Giải thích bằng lời tại sao **trừ ILM** lại giúp hotword boosting hiệu quả hơn. Viết công thức điểm số cuối.
4. Bạn có Whisper-large đóng băng (không được fine-tune) và một danh sách 3.000 thuật ngữ y khoa. Dựa vào bảng §3.9, liệt kê **3 phương án khả thi** và dự đoán phương án nào hỏng trước, vì sao.

---

*Chương tiếp → [`04-token-va-tu-vung.md`](04-token-va-tu-vung.md): trước khi nói về beam search và biasing, phải làm rõ "một từ hiếm" trông như thế nào dưới mắt model — vì đó là nơi phần lớn khó khăn thật sự nằm.*
