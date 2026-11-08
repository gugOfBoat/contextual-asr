# Chương 5 — Giải mã và ghép mô hình ngôn ngữ

> **Mục tiêu chương:** hiểu beam search đủ chi tiết để **tự cài được** biasing vào nó, và hiểu ba cách ghép LM ngoài (shallow fusion, density ratio, ILME) — vì họ phương pháp biasing rẻ nhất, phổ biến nhất và bắt buộc-phải-có-làm-baseline chính là biasing lúc giải mã.

---

## 5.1 Bài toán tìm kiếm

Model cho ta $P(y|x)$ cho **một** $y$ cụ thể. Nhưng ta cần $\arg\max_y$ trên $|\mathcal{V}|^U$ chuỗi — không vét cạn được. Nên mọi hệ ASR đều dùng tìm kiếm **gần đúng**.

- **Greedy**: mỗi bước lấy token xác suất cao nhất. $O(U)$, nhưng một lựa chọn sai sớm không sửa được.
- **Beam search**: giữ $B$ giả thuyết tốt nhất ở mỗi bước.

**Vì sao beam search lại là vấn đề trung tâm của contextual ASR:** một thực thể hiếm có xác suất thấp ở **vài bước đầu**, nên nó bị **cắt khỏi beam trước khi kịp chứng minh mình đúng**. Zhao et al. (Interspeech 2019) chỉ đúng điều này: với E2E, proper noun bị loại sớm, nên biasing **ở mức từ, sau khi đã sinh xong** là vô dụng — phải boost **ở mức token, ngay trong lúc mở rộng beam**. Đây là bài học thiết kế quan trọng nhất chương này.

---

## 5.2 Beam search cho AED (label-synchronous)

```python
# Pseudo-code — beam search cho decoder tự hồi quy
beams = [([BOS], 0.0)]                       # (chuỗi token, log-prob tích luỹ)
for step in range(max_len):
    cand = []
    for seq, score in beams:
        if seq[-1] == EOS:
            cand.append((seq, score)); continue
        logp = model.step(seq, enc_out)      # log P(·| seq, x), vector K chiều
        for k in topk(logp, B):              # chỉ mở rộng B nhánh tốt nhất
            cand.append((seq + [k], score + logp[k]))
    beams = topB(cand, key=lambda s: s[1] / lp(len(s[0])))   # cắt còn B
    if all(b[0][-1] == EOS for b in beams): break
```

**Length normalization.** Log-prob luôn âm nên chuỗi dài luôn thiệt → model thích câu ngắn (lỗi deletion). Sửa bằng cách chia cho $\text{lp}(U)$, ví dụ $\text{lp}(U) = U$ hoặc $\left(\frac{5+U}{6}\right)^\alpha$ (kiểu GNMT). Chi tiết nhỏ nhưng ảnh hưởng WER thật.

**Coverage penalty**: phạt nếu attention bỏ sót vùng audio — công cụ chống hallucination/bỏ đoạn của AED.

---

## 5.3 CTC prefix beam search

Điểm khác biệt: nhiều **path** khác nhau gộp về cùng một **prefix**, nên xác suất phải **cộng dồn**, và phải theo dõi riêng hai trường hợp "prefix kết thúc bằng blank" và "kết thúc bằng non-blank".

```python
# Pseudo-code rút gọn (theo Hannun et al.)
beams = {(): (0.0, -inf)}          # prefix -> (log p_blank, log p_nonblank)
for t in range(T):
    new = defaultdict(lambda: (-inf, -inf))
    for prefix, (pb, pnb) in beams.items():
        for c in topk(logp[t], prune):
            p = logp[t][c]
            if c == BLANK:
                # blank: prefix không đổi, cộng vào nhánh p_blank
                new[prefix] = (logaddexp(new[prefix][0], logsumexp(pb, pnb) + p), new[prefix][1])
            elif prefix and c == prefix[-1]:
                # lặp ký tự: nếu trước đó là non-blank -> vẫn prefix cũ
                new[prefix]        = (new[prefix][0], logaddexp(new[prefix][1], pnb + p))
                # nếu trước đó là blank -> ra ký tự mới
                new[prefix + (c,)] = (new[prefix+(c,)][0], logaddexp(new[prefix+(c,)][1], pb + p))
            else:
                ext = prefix + (c,)
                sc  = logsumexp(pb, pnb) + p
                sc += LM_SCORE(prefix, c)         # ← chỗ cắm LM n-gram
                sc += BIAS_SCORE(prefix, c)       # ← ★ chỗ cắm HOTWORD BOOSTING
                new[ext] = (new[ext][0], logaddexp(new[ext][1], sc))
    beams = topB(new, B)
```

Hai dòng đánh dấu chính là **toàn bộ họ 1** (shallow fusion / ký hiệu). `pyctcdecode` cài đúng như vậy với tham số `hotwords` + `hotword_weight`.

---

## 5.4 Beam search cho RNN-T

Phức tạp hơn vì mỗi frame có thể phát 0, 1 hoặc nhiều token.

```
với mỗi frame t:
    lặp:
        cho mỗi giả thuyết trong beam:
            tính P(k | t, u) qua joint
            nếu k = ∅  -> giả thuyết "hoàn tất ở frame này", chuyển sang beam của frame t+1
            nếu k ≠ ∅  -> mở rộng, u += 1, ở LẠI frame t (prediction net chạy lại)
    cắt beam, giới hạn số token tối đa/frame (chống lặp vô hạn)
```

Biến thể chuẩn: **ALSD** (alignment-length synchronous decoding), **TSD**. Chi tiết thực thi hay gặp: gộp các giả thuyết có **cùng chuỗi token** (vì khác alignment nhưng cùng nhãn → phải cộng xác suất).

---

## 5.5 Ghép mô hình ngôn ngữ ngoài

### (a) Shallow fusion

$$\text{score}(y) = \log P_{\text{ASR}}(y\mid x) + \lambda \log P_{\text{LM}}(y) + \beta\,|y|$$

- $\lambda \approx 0.1$–$0.6$; $\beta$ = phần thưởng độ dài (chống deletion do LM kéo về câu ngắn).
- Cộng **ở mỗi bước token**, không phải cuối câu (lý do ở §5.1).
- Đây cũng là dạng của **hotword boosting**: thay $\log P_{\text{LM}}$ bằng điểm thưởng từ context graph.

### (b) Density Ratio

Nếu LM ngoài huấn luyện trên miền đích, còn model ASR đã "mang sẵn" LM của miền nguồn, thì phải **trừ** LM nguồn:

$$\text{score} = \log P_{\text{ASR}} + \lambda \log P_{\text{LM}}^{\text{đích}} - \mu \log P_{\text{LM}}^{\text{nguồn}}$$

### (c) ILME (Internal LM Estimation) — bản tinh tế hơn

Thay vì huấn luyện riêng một LM nguồn, **ước lượng thẳng LM nội tại của chính model**:
- RNN-T: cho đầu vào encoder $f_t \to 0$ trong joint → phần còn lại là ILM.
- AED: bỏ cross-attention (đặt context vector = 0) → decoder trở thành LM thuần.

$$\text{score} = \log P_{\text{ASR}} + \lambda \log P_{\text{LM}} - \mu \log P_{\text{ILM}}$$

> **Vì sao mục này nằm trong một tài liệu về contextual ASR:** ILM là **lực cản chính** đối với từ hiếm. Nó đã học rằng "amoxicillin" gần như không bao giờ xuất hiện. Boosting phải thắng được lực cản đó. Nếu bạn trừ bớt ILM trước, cùng một mức boost $\lambda$ sẽ hiệu quả hơn nhiều và ít gây over-biasing hơn (vì không phải đẩy $\lambda$ lên cao). **Một baseline shallow-fusion nghiêm túc phải có ILME.**

---

## 5.6 WFST, n-gram và context graph

### N-gram và làm trơn

$P(w_i \mid w_{i-n+1..i-1})$ ước lượng bằng đếm, cộng với **backoff** (Katz/Kneser-Ney) để xử lý n-gram chưa gặp. Lưu dưới dạng ARPA. Vẫn cực kỳ hữu dụng: nhanh, ghép được vào CTC/RNN-T ở mức token, và **sửa được ngay lập tức** khi có từ mới — đúng thứ ta cần cho biasing.

### WFST — đủ dùng, không sa đà

- **FSA/FST**: máy trạng thái hữu hạn có trọng số; đường đi = chuỗi; trọng số cộng dồn theo **tropical semiring** (cộng log-prob, lấy min/max).
- **Composition** $A \circ B$: nối đầu ra của $A$ vào đầu vào của $B$ — cách ghép lexicon với LM.
- **Class-based LM**: trong $G$, đặt một ký hiệu lớp `$CONTACT`; lúc chạy mới thay bằng FST liệt kê danh bạ thật của người dùng. **Đây là contextual biasing bản gốc**, có từ thời hybrid, và vẫn là cách sạch nhất về mặt khái niệm.

### Context graph = trie + failure arc (Aho–Corasick)

Cấu trúc dùng trong **k2/icefall** và **WeNet**:

```
        b       o       s       s
root ──► ● ────► ● ────► ● ────► ●(●)   ← nút cuối: thưởng "hoàn tất cụm"
  ▲      │       │       │       │
  └──────┴───────┴───────┴───────┘
        failure arc: quay về, HOÀN TRẢ điểm đã thưởng
```

**Cơ chế boosting + rollback** (chi tiết quyết định, hay bị làm sai):

- Khi giả thuyết đi được một token khớp trong trie: cộng thưởng $+\alpha$ (thường chia đều: $\alpha = \lambda / L$ cho cụm dài $L$ token, hoặc thưởng theo từng cạnh).
- Khi giả thuyết **đi lệch** khỏi trie (không khớp nữa): **trừ lại toàn bộ phần thưởng đã cộng** trên đường đi hiện tại, rồi theo failure arc về nút phù hợp.

Vì sao phải hoàn trả: nếu không, một giả thuyết chỉ khớp *nửa cụm* rồi rẽ sang chỗ khác sẽ được thưởng miễn phí → sinh ra "nửa thực thể", đúng loại lỗi khó chịu nhất của biasing thô. **Rollback biến điểm thưởng từ "quà tặng" thành "khoản vay"**: chỉ ai đi hết cụm mới được giữ.

```python
# Pseudo-code BIAS_SCORE dùng trong beam search (§5.3)
def bias_score(state, token):
    if token in state.children:              # đi tiếp được trong trie
        nxt = state.children[token]
        s = +alpha
        if nxt.is_end: s += alpha_finish     # thưởng thêm khi hoàn tất cụm
        return s, nxt
    else:                                    # trượt
        return -state.accumulated, state.failure_link(token)   # ← HOÀN TRẢ
```

### Vì sao vẫn cần cái này trong thời Speech-LLM

Vì nó **không cần train lại**, **scale tới hàng chục nghìn cụm**, và chạy được trên GPU (TurboBias xử lý 20.000 cụm chỉ chậm thêm 2–5% RTFx — báo cáo `01` §1.1). Không phương pháp neural nào hiện đạt cả ba tính chất đó cùng lúc.

---

## 5.7 Rescoring: cơ hội thứ hai

**N-best rescoring**: sinh $n$ giả thuyết ở lượt 1, chấm lại bằng model mạnh hơn (LM neural, LLM, hoặc chính model theo chiều ngược).

$$\text{score}_{\text{final}} = \log P_{\text{ASR}} + \lambda_1 \log P_{\text{LLM}} + \lambda_2 |y| \;(+\;\lambda_3\,\text{điểm khớp bias list})$$

- **Rẻ, dễ, và là baseline hợp lệ cho contextual ASR**: cho LLM biết bias list rồi bảo nó chọn trong $n$ giả thuyết.
- **Giới hạn cứng**: nếu thực thể đúng **không có trong n-best**, rescoring bó tay. Đây là *oracle n-best WER* — nên đo nó để biết trần trên của mọi phương pháp rescoring/post-hoc. (Chương 8 §8.6.)
- **Lattice rescoring** mạnh hơn n-best (lattice chứa nhiều giả thuyết hơn theo cấp số nhân) nhưng phức tạp hơn.
- **MBR (Minimum Bayes Risk)**: chọn giả thuyết có **rủi ro kỳ vọng** thấp nhất (trung bình khoảng cách sửa lỗi với các giả thuyết khác), thay vì xác suất cao nhất. Thường tốt hơn 0.2–0.5 điểm WER.
- **ROVER**: bỏ phiếu giữa nhiều hệ thống.

---

## 5.8 Streaming và endpointing

- **Latency** = trễ thuật toán (right-context, chunk size) + trễ tính toán.
- **Endpointing**: quyết định khi nào người nói đã dứt câu — dùng xác suất blank/silence liên tiếp.
- Với biasing streaming: context graph vẫn dùng được nguyên vẹn (nó chỉ là điểm cộng theo token). Còn deep biasing thì phải tính lại embedding ngữ cảnh mỗi bước → đó là lý do có **Deferred NAM** (encode ngữ cảnh trễ, chỉ top-K) và các thiết kế "adaptive biasing" chỉ bật khi cần.

---

## 5.9 Bảng các "núm vặn" và tác dụng thật

| Núm | Tăng lên thì | Cạm bẫy |
|---|---|---|
| Beam size $B$ | rare word có cơ hội sống sót hơn; chậm hơn | quá lớn có thể **tệ đi** (LM/length bias khuếch đại) |
| $\lambda_{\text{LM}}$ | văn bản trôi chảy hơn | nuốt từ hiếm, tăng deletion |
| $\beta$ (length bonus) | ít deletion | nhiều insertion |
| $\mu$ (ILME) | LM ngoài/biasing hiệu quả hơn | quá lớn → mất mạch, ra chuỗi vô nghĩa |
| $\alpha$ (hotword boost) | B-WER giảm | **U-WER tăng** — over-biasing; phải quét và vẽ đường cong |
| Ngưỡng cắt (prune) | nhanh | cắt mất nhánh chứa thực thể |

> **Quy tắc vàng cho luận văn:** không bao giờ báo cáo biasing tại **một** giá trị $\alpha$. Luôn quét $\alpha$ và vẽ **đường cong B-WER ↔ U-WER**. Một điểm đơn lẻ có thể được chọn thiên vị; cả đường cong thì không. Chương 8 §8.4.

---

## Bài tập tự kiểm tra

1. Vì sao trong CTC prefix beam search phải theo dõi riêng $p_{blank}$ và $p_{nonblank}$? Cho ví dụ hỏng nếu gộp làm một.
2. Cài `bias_score` với rollback cho một trie 3 cụm, chạy tay trên chuỗi token `["am","ox","ic","ate"]`. Điểm cuối cùng là bao nhiêu?
3. Giải thích vì sao boosting **ở mức từ sau khi decode xong** gần như vô dụng với E2E, còn boosting **mức token trong beam** thì hiệu quả.
4. Bạn đo được oracle n-best WER (n=10) là 4.2%, WER thực tế 8.9%. Điều đó nói gì về tiềm năng của phương pháp rescoring/post-hoc correction?

---

*Chương tiếp → [`06-ban-chat-contextual-asr.md`](06-ban-chat-contextual-asr.md): đã đủ nền. Giờ phát biểu chính xác bài toán contextual ASR và **bốn nguyên nhân gốc** khiến nó khó.*
