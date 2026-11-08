# Chương 2 — Biểu diễn (embedding) cho audio

> **Mục tiêu chương:** hiểu "embedding cho audio" không phải *một* thứ mà là **một họ** biểu diễn, mỗi loại sinh ra để phục vụ một mục đích khác nhau. Và quan trọng nhất cho đề tài của bạn: hiểu **không gian chung audio–text** (§2.7) — cơ chế cho phép so sánh trực tiếp một đoạn sóng âm với chuỗi ký tự `"paracetamol"`. Đây là nền tảng của toàn bộ dòng retrieval-based contextual biasing (họ 6 trong báo cáo `01`) — dòng mạnh nhất hiện nay.

---

## 2.1 "Embedding" nghĩa là gì

Định nghĩa làm việc:

> **Embedding** là ánh xạ $\phi: \text{đối tượng} \to \mathbb{R}^d$ sao cho **quan hệ hình học trong $\mathbb{R}^d$ phản ánh quan hệ ngữ nghĩa mà ta quan tâm.**

Vế sau mới là phần quan trọng. Log-mel ở chương 1 cũng là một vector $\mathbb{R}^{80}$, nhưng khoảng cách Euclid giữa hai vector log-mel **không** nói lên "hai đoạn này có cùng âm vị không" — nó bị chi phối bởi người nói, micro, độ to. Vậy log-mel là *đặc trưng* (feature) chứ chưa phải *embedding tốt* theo nghĩa trên.

Câu hỏi luôn phải đặt ra khi gặp một embedding bất kỳ: **"gần nhau trong không gian này nghĩa là giống nhau ở điểm gì?"** Câu trả lời được quyết định hoàn toàn bởi **hàm mất mát dùng để huấn luyện nó**.

### Bốn trục để phân loại mọi embedding audio

| Trục | Các giá trị | Ví dụ |
|---|---|---|
| **(1) Độ hạt (granularity)** | frame (10–40 ms) · đoạn/âm vị · từ · toàn câu | log-mel (frame) ↔ x-vector (câu) |
| **(2) Liên tục hay rời rạc** | vector thực · chỉ số codebook | HuBERT feature ↔ HuBERT k-means unit |
| **(3) Cách học** | thủ công · supervised · self-supervised · contrastive · codec (tái tạo) | mel ↔ Whisper-enc ↔ wav2vec2 ↔ CLAP ↔ EnCodec |
| **(4) Căn chỉnh với cái gì** | không gì cả · nhãn âm vị/từ · **văn bản** · danh tính người nói | HuBERT ↔ CTC feature ↔ **CLAP/BR-ASR** ↔ ECAPA |

Trục (4) là trục quyết định cho contextual ASR. Bias list là **văn bản**; muốn dùng nó để lọc/định hướng, ta cần embedding audio **được căn chỉnh với văn bản**.

---

## 2.2 Tầng độ hạt: một câu nói có bao nhiêu loại vector?

```
"cho tôi đơn thuốc paracetamol"   (2.4 giây)
│
├─ mức frame       240 vector ∈ R^80      (log-mel, 10 ms)
├─ mức frame-enc    60 vector ∈ R^512     (encoder Conformer, 40 ms)
├─ mức token        ~8 vector ∈ R^512     (sau CIF/attention pooling, 1 vector/token)
├─ mức từ           5 vector              (pool theo biên từ, cần forced alignment)
└─ mức câu          1 vector ∈ R^192      (x-vector/ECAPA: "ai nói", hoặc CLAP: "nói về cái gì")
```

Ba loại tác vụ dùng ba tầng khác nhau:
- ASR cần tầng **frame-enc** (phải biết cái gì xảy ra lúc nào).
- Speaker verification / LID cần tầng **câu** (thông tin toàn cục).
- **Retrieval hotword cần tầng đoạn/từ hoặc câu** — và đây chính là chỗ khó: ta muốn biết *"trong câu này có xuất hiện cụm `paracetamol` không"*, tức một câu hỏi mức **từ**, nhưng thường chỉ có nhãn mức **câu**. §2.7 nói về cách vượt qua điều này.

---

## 2.3 Ba trụ cột SSL: wav2vec 2.0, HuBERT, WavLM

**Self-supervised learning (SSL)**: học biểu diễn từ audio **không nhãn** bằng cách đặt ra một bài toán tự sinh nhãn. Vì sao quan trọng: audio không nhãn thì vô hạn (YouTube, podcast), audio có nhãn thì đắt. Với ngôn ngữ ít tài nguyên như tiếng Việt, SSL là cách duy nhất khai thác quy mô.

### 2.3.1 wav2vec 2.0 — contrastive trên latent lượng tử hoá

Kiến trúc:

```
waveform → [CNN 7 lớp, stride tổng 320] → z_1..z_T   (latent, 20 ms/frame)
                                            │
                    ┌───────────────────────┴──────────────┐
                    ▼                                      ▼
            mask ~6.5% vị trí                    lượng tử hoá VQ (Gumbel)
                    ▼                                  q_1..q_T
            [Transformer 12/24 lớp] → c_1..c_T          (mục tiêu)
                    └────────── contrastive ─────────────┘
```

Loss contrastive tại các vị trí bị che:

$$\mathcal{L}_m = -\log \frac{\exp\big(\text{sim}(c_t, q_t)/\kappa\big)}{\sum_{\tilde{q} \in Q_t} \exp\big(\text{sim}(c_t, \tilde{q})/\kappa\big)}$$

trong đó $Q_t$ gồm $q_t$ (positive) và $K$ **distractor** lấy từ các vị trí bị che khác **trong cùng câu** (điểm này quan trọng: distractor cùng người nói, cùng kênh → buộc model phân biệt bằng **nội dung ngữ âm**, không phải bằng giọng). $\text{sim}$ = cosine, $\kappa$ = temperature.

Thêm **diversity loss** để mọi entry trong codebook đều được dùng (chống collapse). Codebook: 2 nhóm × 320 entry → $320^2 = 102.400$ tổ hợp.

**Ý nghĩa**: wav2vec2 tạo ra một biểu diễn mà "gần nhau ≈ cùng đơn vị ngữ âm", **không cần biết âm vị là gì**.

### 2.3.2 HuBERT — dự đoán nhãn cụm (ổn định hơn)

Contrastive khá "khó tính" (nhạy với cách lấy distractor). HuBERT đổi sang bài toán **phân loại**:

1. **Vòng 1**: chạy k-means ($k$=100) trên **MFCC** của toàn tập → mỗi frame có một nhãn cụm giả $\hat{z}_t \in \{1..100\}$.
2. Che một số frame, bắt Transformer **dự đoán nhãn cụm** của các frame bị che:
   $$\mathcal{L} = -\sum_{t \in \mathcal{M}} \log p(\hat{z}_t \mid \tilde{x}, t)$$
3. **Vòng 2**: chạy lại k-means ($k$=500) trên **feature tầng giữa (thường tầng 6)** của model vòng 1 → nhãn tốt hơn nhiều → train lại.
4. Lặp.

Trực giác: nhãn ban đầu rất nhiễu, nhưng chỉ cần nó **nhất quán ở mức thống kê** là model học được cấu trúc; rồi model tốt hơn lại sinh nhãn tốt hơn — vòng lặp tự nâng cấp. Đây là ý tưởng đẹp nhất trong speech SSL.

**Hệ quả thực tế cực kỳ hữu ích:** cụm k-means trên HuBERT tầng giữa tương ứng khá tốt với **âm vị**. Nên "HuBERT unit" được dùng làm **token rời rạc cho tiếng nói** (§2.5).

### 2.3.3 WavLM — thêm khử nhiễu và trộn giọng

Giống HuBERT, nhưng khi huấn luyện thì **trộn** thêm câu nói khác/nhiễu vào đầu vào, và vẫn bắt dự đoán nhãn của **câu gốc**. Kết quả: biểu diễn bền với nhiễu và **giữ thông tin người nói tốt hơn** → WavLM-Large là backbone mặc định cho diarization (DiariZen), separation, speaker verification. Nếu đề tài của bạn chạm tới đa người nói (trục A của báo cáo `00`), WavLM là lựa chọn mặc định.

### 2.3.4 So sánh nhanh

| | Mục tiêu | Điểm mạnh | Dùng khi |
|---|---|---|---|
| wav2vec 2.0 | contrastive vs latent lượng tử | kinh điển, nhiều checkpoint đa ngữ (XLS-R) | fine-tune CTC cho ngôn ngữ ít tài nguyên |
| HuBERT | dự đoán nhãn cụm | ổn định, unit rời rạc đẹp | sinh discrete token, textless NLP |
| **WavLM** | + khử nhiễu/trộn giọng | bền nhiễu, giữ speaker | diarization, multi-talker, SV |

---

## 2.4 Phân tích theo tầng: chọn tầng nào để lấy embedding?

Đây là kiến thức thực hành ít được viết ra nhưng dùng hằng ngày.

Một model SSL 24 tầng **không** cho một embedding; nó cho **24 embedding khác nhau**, và chúng mã hoá những thứ khác nhau. Kết quả probing (SUPERB và các nghiên cứu layer-wise) cho bức tranh khá nhất quán:

```
tầng 1–4    : đặc tính tín hiệu, kênh, ▲ THÔNG TIN NGƯỜI NÓI mạnh nhất
tầng 5–12   : ▲ THÔNG TIN NGỮ ÂM (phone) mạnh nhất  ← lấy ở đây cho ASR/alignment
tầng 13–20  : ngữ nghĩa/từ vựng nhiều hơn
tầng cuối   : bị "kéo" về phía mục tiêu pretrain (nhãn cụm) → có thể KÉM hơn tầng giữa
```

> **Cạm bẫy #4 (rất phổ biến):** lấy `last_hidden_state` theo phản xạ. Với model SSL, **tầng cuối thường không phải tầng tốt nhất**. Cách làm chuẩn của SUPERB: học **tổng có trọng số** các tầng, $h = \sum_l w_l h^{(l)}$ với $w$ học được bằng softmax — rồi **nhìn vào $w$** để biết tác vụ của bạn ăn ở tầng nào. Bài này nên làm 1 lần đầu dự án, tốn nửa ngày, tiết kiệm nhiều tuần.

**Whisper encoder** khác: nó được huấn luyện **có giám sát** (audio–text, 680k giờ) nên biểu diễn của nó vốn đã "hướng về văn bản" — rất tốt làm đầu vào cho Speech-LLM (Qwen-Audio, SALMONN, Canary đều dùng encoder kiểu này). Đổi lại: nó bị nén về phía nội dung ngôn ngữ, **mất bớt thông tin người nói/paralinguistic**, và vì huấn luyện trên cửa sổ 30 s cố định nên biểu diễn frame không "sạch" về mặt căn thời gian như CTC.

---

## 2.5 Embedding rời rạc: từ vector sang "token của tiếng nói"

Ý tưởng: biến audio thành **chuỗi số nguyên**, giống như văn bản. Khi đó mọi công cụ của LLM dùng lại được.

### Hai họ token, mã hoá hai thứ khác nhau

| Họ | Cách tạo | Giữ cái gì | Mất cái gì |
|---|---|---|---|
| **Semantic token** | k-means trên HuBERT/w2v-BERT tầng giữa | nội dung ngữ âm/ngôn ngữ | giọng, ngữ điệu, chất âm |
| **Acoustic token** | RVQ trong codec neural (SoundStream, **EnCodec**) | **toàn bộ tín hiệu** (tái tạo được) | không mất gì, nhưng "loãng" ngữ nghĩa |

**RVQ (Residual Vector Quantization)** — cơ chế cốt lõi của codec:

```
x → encoder → v
  q1 = VQ_1(v)            ; r1 = v - q1
  q2 = VQ_2(r1)           ; r2 = r1 - q2      ← lượng tử hoá phần DƯ
  ...
  v ≈ q1 + q2 + ... + qL
```

Mỗi tầng codebook thêm một "lớp chi tiết". Tầng 1 mang thông tin thô nhất (gần nội dung), các tầng sau mang chi tiết âm sắc. **SpeechTokenizer** khai thác trực tiếp điều này: ép tầng RVQ đầu tiên phải khớp với HuBERT unit (semantic distillation), các tầng sau giữ acoustic → được một tokenizer "phân tầng ngữ nghĩa".

**Liên quan gì tới contextual ASR?**
1. Speech-LLM có thể nhận **token rời rạc** thay vì embedding liên tục → ngữ cảnh (bias list) và tiếng nói nằm cùng một loại đầu vào cho LLM.
2. Dòng retrieval **speech→speech** (LA-RAG, báo cáo `01` §1.6) dựng datastore ở **mức token của tiếng nói** — tức là biểu diễn rời rạc/liên tục mức frame được lưu và tra cứu như một kho k-NN.
3. Token rời rạc làm ANN index rẻ hơn nhiều (so vector float 1024 chiều).

---

## 2.6 Embedding mức câu: người nói, ngôn ngữ

Cần cho trục đa người nói và cho target-speaker ASR.

- **x-vector**: TDNN + **statistics pooling** (nối mean và std theo trục thời gian → vector cố định) + softmax phân loại người nói. Lấy tầng ẩn làm embedding.
- **ECAPA-TDNN**: bản nâng cấp — SE block (channel attention), multi-scale (Res2Net), **attentive statistics pooling** (trọng số theo frame). Là chuẩn de-facto hiện nay, 192 chiều.
- Loss: **AAM-softmax** (additive angular margin) — ép các lớp cách nhau một góc biên $m$ trên mặt cầu:
  $$\mathcal{L} = -\log\frac{e^{s\cos(\theta_{y}+m)}}{e^{s\cos(\theta_{y}+m)} + \sum_{j\neq y} e^{s\cos\theta_j}}$$
  Ý nghĩa: **so sánh bằng cosine**, nên embedding phải chuẩn hoá L2 trước khi dùng.

Cùng công thức đó cũng dùng cho **language ID**, **emotion**, v.v. Ý tưởng chung: *pooling theo thời gian + loss phân biệt lớp = embedding mức câu*.

---

## 2.7 ★ Không gian chung audio–text — trái tim của retrieval biasing

Đây là mục quan trọng nhất chương này.

### Vấn đề

Bias list là **văn bản**: `["paracetamol", "amoxicillin", "Nguyễn Thị Hồng", ...]`, có thể 200.000 mục.
Tín hiệu vào là **audio**.
Muốn hỏi: *"trong 200.000 mục này, mục nào có khả năng xuất hiện trong đoạn audio này?"* — nhưng hai thứ ở hai thế giới khác nhau.

### Lời giải: học hai encoder ánh xạ về **cùng một** không gian

```
audio  ─► f_audio(·) ─►  a ∈ R^d ─┐
                                  ├─► so sánh bằng cosine: sim(a,t) = aᵀt / (‖a‖‖t‖)
text   ─► f_text(·)  ─►  t ∈ R^d ─┘
```

Huấn luyện bằng **contrastive loss (InfoNCE)**, dạng đối xứng như CLIP/CLAP. Với một batch $B$ cặp $(a_i, t_i)$ khớp nhau:

$$\mathcal{L} = \frac{1}{2B}\sum_{i=1}^{B}\left[ -\log\frac{e^{\text{sim}(a_i,t_i)/\tau}}{\sum_{j} e^{\text{sim}(a_i,t_j)/\tau}} \; -\log\frac{e^{\text{sim}(a_i,t_i)/\tau}}{\sum_{j} e^{\text{sim}(a_j,t_i)/\tau}} \right]$$

Đọc công thức này thành lời: *"kéo cặp đúng lại gần, đẩy mọi cặp sai trong batch ra xa"*. $\tau$ (temperature) điều khiển độ "gắt": $\tau$ nhỏ → phạt nặng các negative gần.

Sau khi huấn luyện, quy trình suy luận:

```
Offline (1 lần):  200k cụm bias ──f_text──► ma trận T ∈ R^{200000×d} ──► index ANN (FAISS)
Online (mỗi câu): audio ──f_audio──► a ──► top-k cosine ──► 10–50 cụm ──► đưa vào module biasing
```

**Đây chính là lời giải cho bài toán scale** — insight lớn nhất 2025 (báo cáo `01` §1.6). Mọi cơ chế biasing sâu (họ 2–5) đều gãy khi $N$ lớn; retrieval kéo $N$ từ 200.000 xuống 10 trước khi đưa vào.

### Ba khó khăn thật sự của cách này (và đây là chỗ có đề tài)

**(a) Nhãn chỉ có ở mức câu, câu hỏi lại ở mức từ.**
Ta có cặp (audio câu, transcript câu), nhưng muốn học "đoạn audio này ↔ cụm từ này". Cách xử lý: cắt theo forced alignment; hoặc học **global–local** (GLCLAP: căn chỉnh cả mức toàn cục lẫn cục bộ); hoặc dùng attention pooling để model tự tìm đoạn khớp. Khi một câu chứa **nhiều** thực thể, embedding câu bị "trung bình hoá" và mất phân biệt — COALA (Interspeech 2026) đặt đúng vấn đề collapse này.

**(b) Đồng âm.** `bảo` vs `bão` vs `báo`; `Đại` vs `Đài`. Cosine similarity không tự biết phân biệt nếu trong lúc huấn luyện chưa từng bị ép. Cách xử lý: **hard negative mining** — cố tình đưa các cụm phát âm gần giống vào mẫu âm; BR-ASR dùng **dynamic curriculum** tăng dần độ khó của negative. Với tiếng Việt, tập hard negative tự nhiên là **các âm tiết chỉ khác thanh điệu** — một chi tiết thiết kế mà paper tiếng Anh không có lý do gì phải nghĩ tới. *(Đây là một trong những chỗ hở rõ nhất cho đóng góp của bạn.)*

**(c) Chính tả ≠ phát âm.** $f_{text}$ nhận **chữ viết**. Với `vaccine` mà người Việt đọc là *"vắc-xin"*, embedding chữ và embedding audio không có lý do gì gặp nhau. → §2.8.

### Đo chất lượng retriever

Không đo bằng WER mà bằng **Recall@k**: trong top-$k$ trả về, có bao nhiêu % thực thể đúng được bao gồm. Vì retriever là **điểm nghẽn**: nếu Recall@10 = 80%, thì 20% thực thể **không có cơ hội nào** được nhận đúng, dù module biasing phía sau hoàn hảo. Xem chương 8 §8.5 để biết cách thiết kế thí nghiệm tách hai tầng này.

---

## 2.8 Embedding nhận thức phát âm (phonetic-aware) — trục sống còn cho tiếng Việt

Vấn đề gốc, phát biểu gọn: **bias list sống trong không gian chính tả, tín hiệu sống trong không gian phát âm.** Khi hai không gian này lệch nhau, mọi cơ chế biasing đều yếu đi — không phải vì cơ chế dở, mà vì "chìa khoá tra cứu" sai.

Ba mức xử lý, từ rẻ đến đắt:

**Mức 1 — G2P (grapheme-to-phoneme) rồi biasing trên chuỗi âm vị.**
Chuyển `paracetamol` → `/p a r a s e t a m ɔ l/` (hoặc theo cách người Việt đọc: *pa-ra-xê-ta-mon*), dựng trie trên **âm vị** thay vì trên subword. Cần từ điển phát âm hoặc model G2P.

**Mức 2 — sinh nhiều biến thể phát âm cho mỗi cụm.**
Một cụm bias → $M$ cách đọc khả dĩ (chuẩn, theo giọng Nam, đọc kiểu Việt hoá, đọc kiểu tiếng Anh) → tất cả đều nằm trong trie/danh sách. Đây là ý của "zero-shot trie + đa cách phát âm tổng hợp" (báo cáo `01` §1.3). Rẻ, không cần train lại, và **rất hợp bài toán Việt–Anh**.

**Mức 3 — học tương quan phát âm mà không cần từ điển.**
*Automatic Text Pronunciation Correlation* (ICASSP 2025) học ánh xạ chữ↔phát âm trực tiếp từ dữ liệu — hợp với ngôn ngữ thiếu từ điển âm vị. PARCO (ASRU 2025) đưa phoneme encoding + contrastive entity disambiguation vào một hệ thống hoàn chỉnh.

**Về mặt embedding**, cả ba mức đều quy về một việc: thay $f_{text}(\text{chữ})$ bằng $f_{text}(\text{biểu diễn phát âm})$, hoặc nối cả hai:
$$t = [\,f_{\text{graph}}(w)\;;\; f_{\text{phone}}(\text{G2P}(w))\,]$$

> **Ghi chú cho đề tài của bạn.** Trong báo cáo `00` có một dữ kiện đáng suy nghĩ: trên ViMedCSS, Dynamic Vocabulary gần như không cải thiện gì (36.31 → 36.29 WER), trong khi LoRA adapter cải thiện lớn. Giả thuyết tự nhiên nhất — và kiểm chứng được bằng thí nghiệm nhỏ: **bias list ghi bằng chính tả tiếng Anh, còn người nói phát âm Việt hoá**, nên khoá tra cứu không khớp. Cách kiểm chứng: chạy lại đúng thí nghiệm đó nhưng bias list ghi theo cách đọc Việt hoá (hoặc thêm biến thể). Nếu WER cải thiện rõ → bạn có một phát hiện sạch, giải thích được, và mở ra hướng phonetic-aware biasing cho tiếng Việt. Rẻ, nhanh, và là một thí nghiệm "chẩn đoán" đúng nghĩa.

---

## 2.9 Thực hành: lấy embedding thế nào cho đúng

```python
# Ví dụ: lấy embedding mức frame từ một model SSL, chọn tầng có chủ đích
import torch, torchaudio
from transformers import AutoModel, AutoFeatureExtractor

name = "microsoft/wavlm-base-plus"
fe = AutoFeatureExtractor.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_hidden_states=True).eval()

wav, sr = torchaudio.load("a.wav")
assert sr == 16000, "PHẢI resample về 16k trước"          # cạm bẫy #1
wav = wav.mean(0, keepdim=True)                            # về mono

inp = fe(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model(**inp)

hs = out.hidden_states              # tuple 13 (base) hoặc 25 (large) tensor
frame_emb = hs[6][0]                # ← tầng 6: vùng ngữ âm; ĐỪNG mặc định lấy tầng cuối
print(frame_emb.shape)              # (T', 768), T' ≈ 49 frame/giây

# mức câu: mean-pool + chuẩn hoá L2 (bắt buộc nếu so bằng cosine)
utt_emb = torch.nn.functional.normalize(frame_emb.mean(0), dim=-1)
```

Bốn quy tắc thực hành:

1. **Chuẩn hoá L2 trước khi dùng cosine.** Không chuẩn hoá thì "cosine" của bạn thật ra là dot product, và độ dài vector (thường tương quan với độ to/độ dài audio) sẽ lấn át nội dung.
2. **Pooling có trọng số > mean-pool** khi câu dài: mean-pool làm loãng thực thể ngắn nằm trong câu dài — đúng vấn đề (a) ở §2.7.
3. **Không trộn embedding của hai model/hai tầng khác nhau** trong cùng một index. Chúng không cùng không gian, dù cùng số chiều.
4. **Đo trước khi tin.** Dựng một probing task nhỏ (phân loại âm vị / phát hiện từ khoá) rồi so vài tầng. Nửa ngày công, tránh nhiều tuần đi sai.

---

## 2.10 Tổng kết & cạm bẫy chương 2

| Bạn cần gì | Dùng embedding nào |
|---|---|
| Fine-tune ASR ngôn ngữ ít tài nguyên | wav2vec2-XLS-R / WavLM, tầng giữa, + CTC head |
| Diarization / multi-talker | WavLM-Large |
| Speaker verification, target-speaker ASR | ECAPA-TDNN (192-d, cosine) |
| Đầu vào cho Speech-LLM | Whisper encoder / FastConformer + projector |
| **Retrieval hotword từ audio (họ 6)** | **Dual encoder contrastive audio↔text, có hard negative đồng âm** |
| Biasing khi chính tả ≠ phát âm | Embedding có nhánh **âm vị** (§2.8) |
| Token rời rạc cho LLM/datastore | HuBERT k-means unit hoặc RVQ codec |

**Cạm bẫy tổng hợp:**
- Lấy `last_hidden_state` theo phản xạ (§2.4).
- Quên chuẩn hoá L2 rồi so cosine.
- Mean-pool cả câu dài để tìm một thực thể ngắn.
- Huấn luyện contrastive **không có hard negative** → retriever trông rất tốt trên tập dev dễ, sập trên đồng âm thật.
- Giả định embedding văn bản của bias list "tự khắc" khớp với phát âm.

---

## Bài tập tự kiểm tra

1. Giải thích vì sao wav2vec2 lấy distractor **trong cùng một câu** chứ không lấy từ câu khác. Nếu lấy từ câu khác thì model sẽ học phải cái gì?
2. Bạn có 200k cụm bias và một retriever cho Recall@10 = 85%. Giả sử module biasing phía sau là hoàn hảo (bắt đúng 100% cụm được đưa vào). Cận trên của **B-WER** cải thiện được là bao nhiêu? (Trả lời được câu này nghĩa là bạn hiểu vì sao phải đo retriever riêng.)
3. Thiết kế tập hard negative cho retriever tiếng Việt. Liệt kê 4 loại negative và cho ví dụ mỗi loại.
4. Cho `frame_emb` 768 chiều ở 49 fps của một câu 3 giây. Bạn muốn một vector đại diện cho **riêng từ thứ 4**. Nêu 2 cách làm và điều kiện cần của mỗi cách.

---

*Chương tiếp → [`03-kien-truc-asr.md`](03-kien-truc-asr.md): có đặc trưng và embedding rồi, giờ đến cỗ máy biến chúng thành chữ — và điểm mấu chốt là **mỗi kiến trúc mở ra một cửa khác nhau cho ngữ cảnh chui vào**.*
