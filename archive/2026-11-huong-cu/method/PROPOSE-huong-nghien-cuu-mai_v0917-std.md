# Method: Coi mỗi tên là một "keyword" rồi hỏi thẳng audio
### (bản viết lại — open-vocabulary spoken-term detection)

> Bản `PROPOSE-mai-trinh-thay.md` = câu chuyện dễ hiểu. Bản này = method để bảo vệ.

---

## 1. Một câu đổi giả định

| | Input → Output |
|---|---|
| **Contextual ASR truyền thống** | `audio + bias list (cho sẵn) → transcript` |
| **Method của em** | `audio + domain → TỰ SUY RA bias list → transcript` |

Bias list `B` là **biến ẩn**: `(x, d) → B̂ → ŷ`. Gợi ý của thầy — **dùng domain làm mồi** — chính là bước thu hẹp kho.

---

## 2. Đổi cách nghĩ về bài toán (đây là chỗ đã siết lại)

**Cách nghĩ CŨ (đã bỏ):** "tìm token nào ASR nghe sai / confidence thấp rồi tra". Yếu vì confidence calibration kém — ASR **tự tin cao ngay lúc nghe sai** một tên hiếm thành từ thường.

**Cách nghĩ MỚI:**
> Bài toán **không phải** "tìm token đáng nghi". Mà là: sau khi domain + ngữ cảnh đề xuất ~200 tên hợp lý, **coi mỗi tên là một keyword query và hỏi thẳng audio: "từ này có thực sự được nói ở đâu không?"**

Đây là bài **Open-Vocabulary Spoken-Term Detection (STD)** — một mảng có sẵn lý thuyết, không phải heuristic tự chế.

**Tại sao KHÔNG cosine một state với embedding cả cụm:** một state `h_t` chỉ đại diện **một lát âm rất ngắn**, còn `pembrolizumab` hay `Nguyễn Văn Trỗi` là **cả một cụm**. Ép một frame khớp toàn bộ phrase là khập khiễng. → Phải so **theo chuỗi (sequence)**, không so điểm.

---

## 3. Công thức core (thay cho `cos(h_t, z_e)`)

Coi entity là một **chuỗi** đơn vị âm/token: `e = (u₁, u₂, …, u_L)` (vd `pem|bro|li|zu|mab`).
Audio thành chuỗi hidden states `H = (h₁, …, h_T)`.

Hỏi: **có đoạn audio nào mà chuỗi của `e` align tốt vào không?**

$$
R(e \mid x) = \max_{s}\; A(H_s,\, e)
\qquad\qquad
s^{*}(e) = \arg\max_{s}\; A(H_s,\, e)
$$

- `A(·)` = **điểm align chuỗi** (CTC / wildcard-CTC / monotonic token alignment).
- `R(e|x)` = tên đó có được nói không; `s*(e)` = **được nói ở đoạn nào** — vị trí là **sản phẩm phụ của alignment**, không cần detector riêng.

Nếu ASR có CTC thì rất tự nhiên: `R(e|x) = max_{i,j} log P_CTC(e | x_{i:j})` — **CTC tự lo alignment** (không cần biết tên bắt đầu/kết thúc frame nào, mỗi âm dài bao nhiêu).

**Điểm cuối** trộn thêm ngữ nghĩa:
$$
Score(e) = \alpha\, S_{sem}(e \mid d, c) \;+\; \beta\, R(e \mid x)
$$
Nghĩa nói "hợp lý không", audio nói "có thật sự được nói không". `B̂ = TopK_e Score(e)`.

---

## 4. Toàn pipeline

```
   domain d ─► E_d (KB lĩnh vực 10k–100k)
                     │  ③ semantic prune (domain + ngữ cảnh toàn câu)
                     ▼        100k → 200 ứng viên hợp lý
   Audio x ─► Frozen ASR ─► hidden states H
                     │  ④ với MỖI ứng viên: STD trên toàn audio
                     ▼        R(e|x)=max_s A(H_s,e)  → (điểm + vị trí)
              ⑤ fuse Score(e)=α·S_sem+β·R(e|x)
                     ▼        200 → 20
                    B̂  (top-K tên) + span (latent)
                     │
   Audio x ─► ⑥ Contextual ASR (+B̂) ─► transcript
```

Đọc bằng lời: **domain thu kho → semantic lọc còn 200 tên → mỗi tên đem "dò" trên audio kiểu keyword spotting → tên nào audio thực sự đọc thì nổi lên → làm phao → decode lại.**

---

## 5. Ví dụ tiếng Việt (rõ nhất)

Audio thật: *"Tôi đặt khách sạn ở **Nha Trang**."* → ASR ra: *"…ở **nhà tranh**."* (confidence có thể 0.99 — **không quan trọng**).

Domain = `travel` → semantic prune còn: `Nha Trang, Đà Nẵng, Hà Nội, Phú Quốc, Vũng Tàu…`

Coi `Nha Trang` = chuỗi `(Nha, Trang)`, dò trên toàn audio:
```
0–1.2s   0.06
1.2–2.1s 0.08
2.1–2.8s 0.94  ← "Nha Trang" khớp ở đây
```
`Đà Nẵng` best = .19 · `Phú Quốc` best = .08 → **audio trực tiếp ủng hộ `Nha Trang`**.

→ Không cần nói "'nhà tranh' đáng nghi". Ta hỏi thẳng từng ứng viên xem audio có đọc nó không.

---

## 6. Ba cách hiện thực `A(·)` (chọn cách 2)

| | Cách | Precedent | Vai |
|---|---|---|---|
| 1 | **Token-level rồi ghép chuỗi**: mỗi state khớp 1 token entity, đúng thứ tự `a₁<…<a_L` mới thành span. `R(s,e)=Σ_k sim(h_{a_k}, z_{e_k})` | **RECAST** (decoder-state keyword spotting) | baseline |
| **2** | **CTC sequence matching**: `R(e)=max_{i,j} log P_CTC(e\|x_{i:j})`. CTC marginalize mọi alignment → tự tìm vị trí. **Có thể không cần train** (wildcard-CTC). | **WCTC-Biasing** (retraining-free) | **CORE** |
| 3 | **Learned audio–text segment embedding**: `Pool(H_{i:j}) ↔ Embed(e)`, dùng CTC alignment để lấy `(i,j)` | **CTC-aligned Audio-Text Embedding** | mở rộng |

> Cái `cos(proj(h_t), z_e)` ban đầu **hạ xuống làm baseline yếu**, KHÔNG phải core. Core = **align theo chuỗi**.

**Train (nếu cần, nhẹ):** ASR **đóng băng**; chỉ train projector + entity encoder bằng contrastive với hard-negative cùng domain, gold span lấy từ forced-alignment. Cách 2 (wildcard-CTC) có thể chạy **training-free** làm mốc.

**Vì sao hết "lỏ":** thay giả định yếu (confidence⇒tên riêng) bằng **một thuật toán align chuỗi có nền** (STD/CTC); vị trí là output của alignment chứ không phải ngưỡng tay; điểm âm học là log-likelihood/score align thật, không phải edit-distance bịa.

**Chi phí tính:** sau semantic prune chỉ còn ~200 ứng viên, mỗi cái chạy 1 lần STD trên audio → `200 → 20`. Rõ ràng, không "magic".

---

## 7. Thống kê DOMAIN từng dataset (đã kiểm)

Điều kiện chọn: (1) audio, (2) domain lộ được ở test, (3) entity/slot làm gold ẩn, (4) transcript để chấm.

### 7.1 ContextASR-Bench — **chính** (đã đếm thật từ jsonl)
- **Speech/English: 15.326 câu · 116.167 entity (7,6/câu) · domain_label RẤT mịn = 3.275 nhãn free-form.**
- Không phải "10 domain gọn" — mà là nhãn chi tiết, **gom cụm** thì lệch nặng về vài mảng:

| Cụm domain (top) | Ví dụ nhãn |
|---|---|
| **Y tế** (áp đảo) | Medical (1663), Pediatrics (599), Healthcare (240), Cardiology, Oncology, Neurology, Gastroenterology, Pharmaceuticals… |
| **Quân sự/Quốc phòng** | Military Aviation, Defense Technology, Military History/Training/Defense… |
| **Thể thao/Game** | Sports (315), Esports (106), Video Games, Sports Betting… |
| **Văn hoá/Du lịch** | Cultural Heritage, Cultural Tourism, Performing Arts… |
| **Khác** | Geopolitics, Consumer Electronics, International Relations, Botany… |

> ⚠️ 2 điểm phải nói với thầy: (a) domain **quá mịn** → khi eval nên **gom về coarse** (Medical/Military/Sports/…) mới đúng nghĩa "coarse context"; (b) nhánh **Dialogue KHÔNG có domain_label** (5.273 câu, trường trống) → chỉ dùng nhánh **Speech**.

### 7.2 SLURP — **giọng người thật EN** · 18 scenario (taxonomy cố định)
`alarm, audio, calendar, cooking, datetime, email, general, iot, lists, music, news, play, qa, recommendation, social, takeaway, transport, weather` — 18 scenario / 46 action / entity fillers.

### 7.3 Speech-MASSIVE — **tiếng Việt + đa ngữ** · cùng taxonomy MASSIVE
**18 scenario** (như SLURP) / **60 intent** / **55 slot**. 12 ngôn ngữ, có `vi-VN`.
> Cảnh báo quy mô: chỉ fr/de có train đầy đủ; các thứ tiếng khác (gồm VI) chỉ có **few-shot train = 115 câu** + dev/test. → Dùng VI làm **test generalization / few-shot**, **không** train retriever VI từ đầu.

### 7.4 SpokenWOZ — **hội thoại người thật, khó thật** (secondary) · 8 domain
Kế thừa MultiWOZ: `attraction, hotel, restaurant, taxi, train, hospital, police` (+ `profile`). 249h, 5.7k hội thoại, có slot value + span + ASR timestamp.
> Khó vì: 1 hội thoại đa domain, ngữ cảnh kéo nhiều lượt → để **benchmark khó cuối**, không phải bộ đầu.

**Loại khỏi ưu tiên:** ATIS (chỉ 1 domain airline); LibriSpeech/Common Voice (không có cấu trúc domain+entity tự nhiên).

---

## 8. Xây candidate KB — KHÔNG được lấy entity của chính câu (cheating)

`E_d` = **hợp các entity của TRAIN theo domain**. Test chỉ thấy `E_d`, không biết gold nào trong đó.
Hai chế độ khó: **Seen** (`e*∈E_d^train`) vs **Unseen** (`e*∉E_d^train` → cần KB ngoài / open-vocab, khó hơn).

---

## 9. Đánh giá — 3 stage

**Stage A · Retrieval + Localization** (ContextASR + SLURP-real): `EntityRecall@K, MRR`, và nếu có alignment: `LocalizationRecall`. So `semantic-only / STD-only / semantic+STD`.

**Stage B · End-to-end** (bảng chính):

| Config | Ý nghĩa |
|---|---|
| No bias | baseline |
| Full domain KB bias | quá nhiều distractor |
| Semantic top-K | không có spoken-term detection |
| **Semantic + STD (method)** | — |
| Oracle entities | cận trên |

Report: `WER, NE-WER, EntityRecall`. Chỉ số quyết định = **NE-WER↓** trong khi WER từ thường giữ nguyên.

**Stage C · Generalization**: Speech-MASSIVE-VI (đa ngữ/Việt) + SpokenWOZ (hội thoại thật).

**Thứ tự chốt: ContextASR-Bench → SLURP → Speech-MASSIVE(VI) → SpokenWOZ** (controlled → realistic).

---

> ### Một câu nói với thầy
> *"Em làm đúng ý thầy — dùng **domain làm mồi** để còn ~200 tên hợp lý. Nhưng em **bỏ cách đoán tên riêng bằng độ tự tin**. Thay vào đó em **coi mỗi tên là một keyword và dò thẳng trên audio** (open-vocabulary spoken-term detection kiểu CTC/wildcard-CTC): tên nào audio thực sự đọc thì điểm align cao và tự lộ vị trí — không cần detector, và confidence của ASR có cao 0.99 cũng không sao. Em chỉ cần ASR đóng băng, thậm chí wildcard-CTC chạy không-train được. Lấy top tên làm phao rồi decode lại. Benchmark chính ContextASR-Bench, sau đó SLURP, Speech-MASSIVE tiếng Việt, SpokenWOZ."*
