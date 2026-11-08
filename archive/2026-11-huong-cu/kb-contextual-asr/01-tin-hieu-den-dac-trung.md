# Chương 1 — Từ sóng âm đến đặc trưng

> **Mục tiêu chương:** sau chương này bạn phải trả lời được, không cần tra: một file wav 16 kHz dài 5 giây, sau tiền xử lý chuẩn, biến thành tensor kích thước bao nhiêu, mỗi chiều nghĩa là gì, và tại sao mọi bước biến đổi lại được chọn như vậy. Đây là tầng thấp nhất — mọi thứ ở các chương sau đều đứng trên nó.

---

## 1.1 Âm thanh là gì, dưới góc nhìn kỹ thuật

Tiếng nói là dao động áp suất không khí. Micro biến dao động đó thành điện áp liên tục $s(t)$. Máy tính không lưu được hàm liên tục, nên phải **số hoá** theo hai trục:

- **Trục thời gian → lấy mẫu (sampling).** Đo giá trị $s(t)$ tại các thời điểm cách đều nhau $1/f_s$ giây. $f_s$ = sampling rate.
- **Trục biên độ → lượng tử hoá (quantization).** Làm tròn giá trị đo được về số nguyên có $B$ bit. $B$ = bit depth.

Kết quả: một mảng số nguyên 1 chiều. Một câu nói 5 giây ở 16 kHz, 16-bit mono = mảng 80.000 số, mỗi số trong khoảng $[-32768, 32767]$.

### Định lý lấy mẫu (Nyquist–Shannon) — vì sao là 16 kHz

Muốn khôi phục được thành phần tần số $f$ thì phải lấy mẫu ở tốc độ $f_s > 2f$. Ngược lại: với $f_s$ cho trước, ta chỉ giữ được thông tin đến **tần số Nyquist** $f_s/2$.

| $f_s$ | Nyquist | Giữ được gì | Dùng ở đâu |
|---|---|---|---|
| 8 kHz | 4 kHz | Đủ nhận ra nguyên âm, **mất nhiều thông tin phụ âm xát** (`s`, `x`, `ʃ` có năng lượng tới 8–10 kHz) | Điện thoại cũ, tổng đài |
| **16 kHz** | 8 kHz | Đủ cho gần như toàn bộ thông tin ngữ âm | **Chuẩn của mọi ASR hiện đại** |
| 44.1/48 kHz | 22 kHz | Thừa cho tiếng nói | Nhạc, video |

> **Cạm bẫy #1 (kinh điển, làm hỏng WER mà không ai biết):** đưa audio 8 kHz vào model huấn luyện ở 16 kHz. Nếu bạn chỉ "upsample" bằng cách nội suy, dải 4–8 kHz sẽ **trống rỗng** — model chưa từng thấy phổ như vậy, WER tăng vô cớ. Ngược lại, đưa audio 44.1 kHz mà quên resample thì mọi thứ bị "dịch tần số", model hỏng hoàn toàn. Luôn kiểm tra `sr` trước khi làm bất cứ điều gì.

### Vài chi tiết thực tế hay bị bỏ qua

- **Mono.** Đa kênh chỉ xuất hiện với mảng micro (phòng họp). Nếu file stereo mà bạn lấy 1 kênh, có thể mất người nói ngồi bên kia.
- **Chuẩn hoá biên độ.** Thường chia cho 32768 để đưa về float $[-1,1]$. Một số pipeline làm peak-normalize hoặc chuẩn hoá theo RMS; **khác nhau giữa train và test là một nguồn lỗi thật**.
- **Dither & pre-emphasis.** Di sản Kaldi: cộng nhiễu cực nhỏ (dither) để tránh log(0), và lọc pre-emphasis $x'[n] = x[n] - 0.97x[n-1]$ để bù độ dốc phổ (tiếng nói suy giảm ~6 dB/octave ở tần số cao). Model neural hiện đại phần lớn **không cần** pre-emphasis nữa, nhưng recipe Kaldi/ESPnet vẫn bật — nên khi so hai hệ thống, phải biết bên nào bật.

---

## 1.2 Framing: vì sao phải cắt thành khung 25 ms

Tiếng nói **không dừng** (non-stationary): trong một giây, bộ máy phát âm thay đổi hình dạng liên tục. Nhưng phân tích tần số (Fourier) chỉ có nghĩa khi tín hiệu **dừng** (tính chất thống kê không đổi).

Giải pháp: giả định **tựa dừng trong khoảng ngắn**. Trong 20–30 ms, lưỡi và môi gần như đứng yên → coi là dừng được.

```
waveform:  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frame 1:   [--- 25 ms ---]
frame 2:        [--- 25 ms ---]        ← trượt 10 ms, chồng 15 ms
frame 3:             [--- 25 ms ---]
...
```

- **Window length** = 25 ms → ở 16 kHz là **400 mẫu**.
- **Hop length (frame shift)** = 10 ms → **160 mẫu**.
- ⇒ **1 giây audio = 100 frame.** Con số này in vào mọi thứ: độ phân giải timestamp, độ dài chuỗi encoder, cách tính DER/tcpWER.

**Vì sao chồng lấn 15 ms?** Nếu cắt không chồng, một sự kiện ngữ âm rơi vào ranh giới hai frame sẽ bị "xé". Chồng lấn làm chuỗi đặc trưng mượt và bất biến hơn với vị trí cắt.

**Hàm cửa sổ (windowing).** Cắt thẳng bằng cửa sổ chữ nhật = nhân với hàm nhảy bậc → trong miền tần số là tích chập với hàm sinc có búp phụ lớn → **rò phổ (spectral leakage)**: năng lượng của một tần số bị "vấy" sang tần số khác. Nên người ta nhân frame với cửa sổ trơn:

$$w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N-1}\right) \quad \text{(Hamming; Hann tương tự với 0.5/0.5)}$$

Cửa sổ trơn ở hai đầu → giảm rò phổ, đổi lại độ phân giải tần số hơi kém đi. Đây là đánh đổi kinh điển của phân tích tín hiệu.

---

## 1.3 Từ frame sang phổ: DFT và spectrogram

Với mỗi frame $\mathbf{x} = (x[0],\dots,x[N-1])$ đã nhân cửa sổ, tính **DFT** (thực tế dùng FFT):

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \qquad k = 0,\dots,N-1$$

- $X[k]$ là số phức. $|X[k]|$ = biên độ tại tần số $f_k = k \cdot f_s / N$; $\arg X[k]$ = pha.
- Vì tín hiệu thực nên phổ đối xứng → chỉ giữ $N/2 + 1$ bin đầu.
- Thường zero-pad frame 400 mẫu lên $N = 512$ (luỹ thừa của 2, FFT nhanh) → **257 bin**, mỗi bin rộng $16000/512 = 31.25$ Hz.
- **Power spectrum**: $P[k] = |X[k]|^2$.

**Ghép các frame lại theo thời gian → spectrogram**: ma trận $257 \times T$ ("ảnh" của âm thanh).

> **Pha bị vứt đi.** ASR hầu như chỉ dùng biên độ. Tai người rất kém nhạy với pha tuyệt đối, và biên độ đủ để phân biệt âm vị. (Ngoại lệ: speech enhancement/separation thì pha quan trọng — đó là lý do các model tách nguồn hiện đại như TF-GridNet làm việc trên phổ phức hoặc trực tiếp trên waveform.)

### Cấu trúc mà bạn phải nhìn thấy trong spectrogram

Với tiếng nói hữu thanh (nguyên âm), spectrogram có **hai tầng cấu trúc** — đây là điều then chốt cho tiếng Việt:

1. **Cấu trúc điều hoà (harmonics)**: các vạch ngang cách đều nhau $F_0, 2F_0, 3F_0,\dots$ — do dây thanh rung. $F_0$ = tần số cơ bản = **cao độ**. Với tiếng Việt, **đường $F_0$ theo thời gian chính là thanh điệu.**
2. **Bao phổ (spectral envelope)**: các vùng năng lượng mạnh gọi là **formant** ($F_1, F_2, F_3$) — do cộng hưởng khoang miệng. **Formant xác định nguyên âm** (`a` vs `i` vs `u`).

Nói cách khác: *nguồn* (dây thanh, mang thanh điệu) và *bộ lọc* (khoang miệng, mang âm sắc) — mô hình **source–filter**. Toàn bộ ngữ âm học nằm ở đây.

---

## 1.4 Mel filterbank: nén phổ theo cách tai người nghe

257 bin tuyến tính là dư thừa và không khớp với tri giác. Tai người phân biệt tần số thấp rất tinh, tần số cao thì thô. Thang **mel** mô hình hoá điều đó:

$$m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Cách làm:

1. Chọn $M$ (thường **80**, Whisper large-v3 dùng 128) điểm cách đều **trên thang mel** giữa $f_{min}$ (thường 0 hoặc 20 Hz) và $f_{max}$ (thường 8000 Hz).
2. Đổi ngược về Hz → được các điểm cách nhau **hẹp ở tần số thấp, thưa ở tần số cao**.
3. Dựng $M$ bộ lọc **tam giác** chồng lấn, mỗi bộ lọc phủ 3 điểm liên tiếp.
4. Năng lượng băng thứ $m$: $E_m = \sum_k P[k]\, H_m[k]$.

```
Đáp ứng bộ lọc mel (minh hoạ):
 |   /\  /\   /\     /\        /\             /\
 |  /  \/  \ /  \   /  \      /  \           /  \
 | /   /\   X    \ /    \    /    \         /    \
 +-------------------------------------------------→ Hz
   0   200 400  800     1600         3200        6400
   ↑ dày đặc ở thấp                    ↑ thưa dần ở cao
```

Kết quả: từ 257 chiều xuống **80 chiều**. Đây chính là câu trả lời cho câu hỏi ở đầu chương: **80 chiều = 80 băng tần mel, chiều thứ $m$ là năng lượng trong băng tần thứ $m$.**

### Lấy log — bước quan trọng hơn vẻ ngoài

$$\text{logmel}_m = \log(E_m + \epsilon)$$

Ba lý do, xếp theo mức quan trọng:

1. **Nén dải động.** Năng lượng âm thanh trải trên nhiều bậc độ lớn ($10^6$ lần). Mạng neural ghét đầu vào như vậy. Log nén lại thành khoảng hợp lý.
2. **Biến nhân thành cộng.** Kênh truyền (micro, phòng) tác động theo kiểu nhân trong miền phổ: $P_{obs} = P_{speech} \cdot H_{channel}$. Lấy log → $\log P_{obs} = \log P_{speech} + \log H$. Nhiễu kênh trở thành **cộng hằng số**, và có thể trừ đi bằng cách chuẩn hoá trung bình (CMVN). Đây là lý do sâu xa của bước log, không phải chuyện thẩm mỹ.
3. **Khớp tri giác.** Cảm nhận độ to của tai người xấp xỉ logarit (định luật Weber–Fechner).

### CMVN (Cepstral/Cepstral Mean & Variance Normalization)

$$\hat{v}_{t,m} = \frac{v_{t,m} - \mu_m}{\sigma_m}$$

trong đó $\mu_m,\sigma_m$ tính trên **một câu** (utterance-level, phổ biến), **một người nói** (speaker-level), hoặc **toàn tập** (global). Tác dụng: xoá đặc tính kênh/micro cố định. Cạm bẫy: CMVN utterance-level cần biết cả câu → **không dùng được cho streaming thuần**; streaming phải dùng CMVN chạy (running) hoặc global.

> **Cạm bẫy #2:** mỗi toolkit tính log-mel hơi khác nhau (Kaldi vs torchaudio vs Whisper: khác ở dither, pre-emphasis, cách chuẩn hoá, `f_max`, cách log). **Feature của toolkit A đưa vào model của toolkit B = WER tăng bí ẩn.** Luôn dùng đúng feature extractor đi kèm checkpoint (`WhisperFeatureExtractor` cho Whisper, v.v.).

---

## 1.5 MFCC — vì sao đã lỗi thời nhưng vẫn phải biết

Thêm một bước **DCT** lên log-mel:

$$c_i = \sum_{m=1}^{M} \text{logmel}_m \cos\!\left[\frac{\pi i (m - 0.5)}{M}\right], \quad i = 1..13$$

Mục đích lịch sử: DCT **khử tương quan** giữa các chiều, để dùng được GMM với ma trận hiệp phương sai **đường chéo** (rẻ hơn nhiều). Thêm delta ($\Delta$) và delta-delta ($\Delta^2$) để bù thông tin động → 39 chiều.

**Vì sao mạng neural không cần MFCC nữa:** mạng neural không cần đầu vào khử tương quan (nó tự học biến đổi tuyến tính), và DCT **vứt bỏ thông tin** (giữ 13/80 hệ số). CNN/Transformer trên log-mel luôn tốt hơn.

**Nhưng phải biết vì:** (a) đọc paper trước 2015; (b) HuBERT dùng MFCC cho vòng k-means đầu tiên (chương 2); (c) nhiều pipeline forced-alignment (MFA) vẫn dùng MFCC.

**Về mặt lý thuyết**: MFCC là "cepstrum" — phổ của log-phổ. Nó **tách source khỏi filter**: hệ số bậc thấp ($c_1..c_{13}$) mô tả bao phổ (formant, tức âm sắc), hệ số bậc cao mô tả cấu trúc điều hoà mịn (tức $F_0$, tức **thanh điệu**). ⇒ **MFCC 13 chiều vứt bỏ gần hết thông tin thanh điệu.** Đó là lý do các hệ ASR tiếng Việt/Trung thời hybrid phải nối thêm **đặc trưng pitch** (Kaldi `compute-kaldi-pitch-feats`: 3 chiều pitch + POV). Với log-mel 80 chiều thì thông tin điều hoà vẫn còn (dù bị nén ở băng thấp), nên model neural hiện đại học được thanh điệu mà không cần pitch feature rời — nhưng đây là điểm đáng kiểm chứng thực nghiệm nếu bạn làm tiếng Việt ⚠️.

---

## 1.6 Subsampling: vì sao encoder rút ngắn chuỗi 4 lần

100 frame/giây là quá dài cho self-attention (chi phí $O(T^2)$). Ngoài ra, một âm vị kéo dài 50–100 ms tức 5–10 frame — độ phân giải 10 ms là **dư thừa**.

Nên encoder bắt đầu bằng 2 lớp CNN stride 2 → giảm 4× → **25 frame/giây, mỗi frame 40 ms**.

Sổ tay kích thước (nhớ để đọc code không lạc):

| Model | Đầu vào | Sau frontend | Ghi chú |
|---|---|---|---|
| Conformer/Zipformer chuẩn | log-mel 80, 100 fps | 25 fps, $d$=256–512 | subsample 4× |
| **Whisper** | log-mel 80 (large-v3: 128), **luôn pad/cắt đúng 30 s** = 3000 frame | **1500 frame** (20 ms/frame), $d$=1280 (large) | subsample 2× |
| wav2vec2 | **waveform thô** | 49 fps (~20 ms/frame), $d$=768/1024 | 7 lớp CNN, tổng stride 320 |

**Ví dụ tính tay (bài tập bắt buộc):** file 5 giây, 16 kHz.
- Waveform: 80.000 mẫu.
- Frame: $\lfloor (80000 - 400)/160 \rfloor + 1 = 498$ frame.
- Log-mel: tensor $498 \times 80$.
- Sau encoder Conformer (4×): $\approx 125 \times 512$.
- Với Whisper: pad lên 30 s → $3000 \times 80$ → encoder → $1500 \times 1280$ (kể cả khi audio chỉ có 5 giây thật — 25 giây còn lại là im lặng được pad, và **đây chính là nguồn gốc của hiện tượng hallucination lặp vô hạn** của Whisper trên audio ngắn/im lặng).

---

## 1.7 SpecAugment và tăng cường dữ liệu ở mức đặc trưng

SpecAugment (Park et al., 2019) — tăng cường **trực tiếp trên log-mel**, rẻ và hiệu quả bất ngờ:

1. **Time masking**: chọn ngẫu nhiên $t_0$, che $[t_0, t_0+t)$ frame (đặt về 0 hoặc về trung bình).
2. **Frequency masking**: che $[f_0, f_0+f)$ băng mel.
3. **Time warping**: kéo giãn nhẹ trục thời gian (ít quan trọng, nhiều recipe bỏ).

Trực giác: buộc model **không được phụ thuộc vào một dải tần hay một khoảnh khắc duy nhất** → phải dùng ngữ cảnh → chính là thứ ta muốn.

Các tăng cường khác ở mức waveform: speed perturb (0.9/1.0/1.1×), volume perturb, thêm nhiễu (MUSAN), tích chập RIR (mô phỏng vọng phòng), codec augmentation (nén mp3/opus rồi giải nén).

> **Liên hệ contextual ASR:** khi huấn luyện module biasing, phần lớn dữ liệu chứa thực thể hiếm là **tổng hợp bằng TTS** (báo cáo `01` §1.8b). Lúc đó augmentation ở mức waveform/feature là bắt buộc, vì audio TTS quá "sạch" so với audio thật → model học biasing trên phổ TTS rồi hỏng trên audio thật. Đây là một failure mode có thật, đáng làm ablation.

---

## 1.8 Bảng tổng kết chương & cạm bẫy

| Bước | Vào → Ra | Tham số chuẩn | Sai thì hậu quả |
|---|---|---|---|
| Sampling | analog → mảng int | 16 kHz, 16-bit, mono | Sai `sr` → hỏng hoàn toàn |
| Framing + window | mảng → $T$ frame | 25 ms / 10 ms, Hann | Ít ảnh hưởng nếu lệch nhẹ |
| FFT | frame → 257 bin phức | n_fft=512 | — |
| Power | phức → thực | $\|X\|^2$ | — |
| Mel filterbank | 257 → 80 | 80 băng, 0–8 kHz | `f_max` sai → lệch feature |
| Log | nén dải động | $\log(E+\epsilon)$ | Quên log → mạng khó hội tụ |
| CMVN | chuẩn hoá | utterance/global | Train/test khác nhau → WER tăng |
| Subsample | $T$ → $T/4$ | CNN stride 2×2 | — |

**Ba cạm bẫy chết người, nhắc lại:**
1. Sample rate không khớp.
2. Feature extractor không khớp checkpoint.
3. Chuẩn hoá (CMVN/biên độ) khác nhau giữa train và inference.

Cả ba đều gây "WER tăng mà không hiểu vì sao", và cả ba đều **không** báo lỗi.

---

## Bài tập tự kiểm tra

1. Một file 3.2 giây, 16 kHz. Tính số frame log-mel, kích thước tensor, số frame sau encoder 4×, và số frame nếu đưa vào Whisper.
2. Vì sao lấy log **trước** khi đưa vào mạng lại giúp chống nhiễu kênh? Viết lại bằng công thức.
3. Nếu bạn muốn model nhạy hơn với **thanh điệu tiếng Việt**, bạn sẽ can thiệp ở bước nào trong chuỗi trên? Nêu 2 phương án và đánh đổi của mỗi phương án.
4. Mở `torchaudio`, tính log-mel của một file bằng `torchaudio.compliance.kaldi.fbank` và bằng `WhisperFeatureExtractor`, so sánh shape và dải giá trị. Chúng khác nhau ở chỗ nào?

---

*Chương tiếp → [`02-embedding-cho-audio.md`](02-embedding-cho-audio.md): log-mel là đặc trưng "thủ công". Chương sau nói về đặc trưng **học được** — tức embedding — và về không gian chung audio–text, nền tảng của toàn bộ contextual biasing hiện đại.*
