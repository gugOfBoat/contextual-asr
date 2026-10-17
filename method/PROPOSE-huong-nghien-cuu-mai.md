# Method: Tự khám phá bias list từ audio + domain, rồi decode lại
### (bản gọn để trình bày — CHỐT 1 hướng sâu)

> Bản `PROPOSE-mai-trinh-thay.md` = câu chuyện dễ hiểu kể kèm. Bản này = method để bảo vệ.

---

## 1. Một câu đổi giả định

| | Input → Output |
|---|---|
| **Contextual ASR truyền thống** | `audio + bias list (cho sẵn) → transcript` |
| **Method của em** | `audio + domain → TỰ SUY RA bias list → transcript` |

Bias list `B` không được cho sẵn → trở thành **biến ẩn** phải tự khám phá:

```
(x, d)  →  B̂  →  ŷ
```
`x`=audio, `d`=domain đã biết (vd `oncology`), `B̂`=bias list model tự suy, `ŷ`=transcript.

---

## 2. Nghịch lý cốt lõi (điểm bán hàng)

> **Từ cần gợi ý = đúng những từ ASR vừa nghe SAI** → không tra được bằng CHỮ (chữ đã hỏng), phải tra bằng **ÂM**.

---

## 3. Nền tảng lý thuyết — 1 công thức Bayes (đủ cho cả method)

$$
P(e \mid x, h_0, d)\ \propto\ \underbrace{P(x \mid e)}_{\text{âm học}}\ \cdot\ \underbrace{P(e \mid h_0, d)}_{\text{ngữ nghĩa + domain}}
$$

Nói bằng lời — **hai loại bằng chứng bù nhau**:
- **Ngữ nghĩa `P(e|h₀,d)`**: từ nào *hợp lý* trong ngữ cảnh này? (context xung quanh + domain prior)
- **Âm học `P(x|e)`**: trong số đó, từ nào *thực sự nghe giống* audio nhất?

→ Toàn bộ method chỉ là ước lượng 2 số hạng này rồi kết hợp. Không cần gì thêm.

---

## 4. METHOD — pipeline 5 bước (đây là cái em chọn implement)

```
Audio ─► ① First-pass ASR ─► transcript thô + confidence + N-best/posterior
                                        │
                            ② Uncertainty Detection
                            (tách span nghi ngờ vs context xung quanh)
                                        │
                    ┌───────────────────┴───────────────────┐
              context xung quanh                    span nghi ngờ (audio/âm)
                    │                                        │
   domain ─► ③ Semantic Retrieval                            │
              (KB domain 100k → top-200)                     │
                    └───────────────────┬───────────────────┘
                                        ▼
                            ④ Acoustic/Phonetic Reranking
                                  (top-200 → top-20)
                                        │
                                     Bias list B̂
                                        │
   Audio ─────────────────────► ⑤ Contextual ASR ─► transcript cuối
```

**① First-pass ASR = "context sensor", KHÔNG phải output.**
Giữ nhiều hơn 1-best: N-best + token posterior. Dù entity sai chữ (`pembra lizuma`), acoustic structure vẫn gần `pembrolizumab`.

**② Uncertainty Detection — bước then chốt, ít ai làm.**
Token nào `P(token|x) < τ` = chỗ nghi tên riêng. Tách câu làm 2 vai:
```
treated with  [pembra lizuma]  for metastatic melanoma
└─context trái─┘ └─span âm học─┘ └────context phải────┘
```
→ **Ngữ nghĩa lấy từ context xung quanh, âm học lấy từ đúng span nghi ngờ** (không đem cả câu đi so với 1 entity ngắn).

**③ Semantic Retrieval — thu hẹp không gian tìm.**
Query `= f(domain + context xung quanh)`, bi-encoder cosine trên KB domain `E_d`. Trả lời: *"từ nào plausible ở đây?"* — **không cần biết audio phát âm gì**. `100k → 200`.

**④ Acoustic/Phonetic Reranking — phần novelty.**
Score 200 candidate bằng độ khớp âm với span nghi ngờ. Trả lời: *"từ nào thực sự nghe giống?"* `200 → 20`.
Điểm kết hợp: `S(e) = α·logP(e|d) + β·logP(e|h₀,d) + γ·logP(x|e)`.

**⑤ Contextual ASR** decode lần 2 với `B̂` → transcript cuối.

**Vì sao semantic TRƯỚC, acoustic SAU:** so âm trên cả 100k–1M từ là bất khả thi và dễ khớp bậy; semantic + domain thu về 200 rồi mới so âm mới sạch. Mỗi tầng giải một loại bất định khác nhau → bổ sung tự nhiên.

---

## 5. Riêng bước ④ — âm học tính thế nào (3 mức, làm từ dễ)

| Mức | Cách | Công thức |
|---|---|---|
| **A. dễ nhất** | phonetic edit distance | `S = −EditDist(φ(span), φ(e))` |
| **B. mạnh hơn** | trên N-best (robust) | `S = max_i sim(φ(hᵢ), φ(e))` |
| **C. đẹp nhất** | trực tiếp trên CTC/RNNT posterior | `S = log P(e|x)` — *không cần decode entity ra 1-best* |

Prototype đầu chạy mức **A** (không train gì), rồi nâng lên **C**.

---

## 6. Đóng góp riêng tiếng Việt

- **Thanh điệu/dấu**: `"nhà tranh"` vs `"Nha Trang"` khác dấu → khác phát âm → so âm không khớp bậy. Đây là chỗ tiếng Anh không có → **đóng góp riêng** cần đưa vào `φ(·)`.

---

## 7. Đánh giá: tách Retrieval khỏi ASR (nếu không reviewer bác)

| Tầng | Đo gì | Metric |
|---|---|---|
| ① Bias discovery | retrieve đúng entity? | Recall@K, MRR |
| ② Context exploitation | cho **oracle bias** → ASR tận dụng nổi? | nếu oracle vẫn tệ → decoder là bottleneck |
| ③ End-to-end | kết quả cuối | **NE-WER↓**; WER từ thường phải GIỮ NGUYÊN |

**Metric đề xuất — Context Utility:** `CU(B) = WER(x) − WER(x,B)`. Cái ta cần không phải recall cao mà là bias có **thực sự giúp ASR** không.

**Ablation = cách chứng minh method** (không phải "menu" — mà để tách đóng góp từng thành phần):

| Cấu hình | Bỏ đi cái gì | Trả lời câu hỏi |
|---|---|---|
| No-bias | — | mức nền |
| Domain-only `R(d)` | bỏ audio + semantic | domain prior đóng góp bao nhiêu? |
| Semantic-only `R(h₀,d)` | bỏ ④ acoustic | **âm học đáng giá bao nhiêu?** ← thí nghiệm then chốt |
| **Full (semantic+acoustic)** | — | method chính |
| Oracle bias | — | cận trên; khoảng cách = "context gap" |

**Phép thử quyết định:** tách riêng nhóm câu **lần 1 nghe SAI tên**. Trên nhóm này kỳ vọng **semantic+acoustic ≫ semantic-only** (vì chữ đã hỏng, chỉ âm còn dùng được). Ra đúng vậy = bằng chứng method có giá trị.

---

## 8. Dataset (đã kiểm tra thật)

| Vai trò | Bộ | Vì sao |
|---|---|---|
| **Chính** | **ContextASR-Bench (2025)** | có sẵn `<audio, text, coarse context, fine entities>`. Đổi protocol: chỉ lộ `audio + coarse`, giấu entity làm ground-truth cho discovery. |
| **Người thật** | **SLURP** | 18 scenario, slot → convert thành bias target; giọng người thật. |
| **Đa ngữ / Việt** | **Speech-MASSIVE** | có tiếng Việt, cùng taxonomy 18 domain/55 slot. |

---

## 9. Positioning & novelty

```
Truyền thống      → người đưa bias list sẵn
Large-scale       → retrieve từ candidate pool cho sẵn
METHOD CỦA EM     → chỉ có domain thô → tự dựng candidate → chọn theo ngữ nghĩa
                    → rerank theo ÂM HỌC → contextual decode
```
- Bước "chọn vài tên từ kho lớn" là bước work gần nhất *cố tình né* — em làm đúng bước đó.
- Chưa ai làm cho **tiếng Việt** (dấu/thanh điệu).
- **Không train model khổng lồ** → hợp tầm luận văn.

---

## 10. Lộ trình

| Phase | Làm gì | Chốt |
|---|---|---|
| **P1** | So `ASR(x)` vs `+domain` vs `+oracle` | oracle ≫ domain → có "context gap" đáng giải |
| **P2** | Pipeline bước ①②③ + acoustic mức A (no train) | baseline chạy được |
| **P3** | Nâng ④ lên CTC posterior (mức C) | phần lõi novelty |
| **P4** | (mở rộng) train contrastive audio↔entity nếu cần đẩy số | |
| **P5** | Validate SLURP + Speech-MASSIVE (VI) | generalize |

*Iterative refinement (`y₀→B₁→y₁→B₂…`) để dành làm ablation, KHÔNG phải core.*

---

> ### Một câu nói với thầy
> *"Method của em: chỉ từ audio + domain thô, em (1) chạy ASR lần 1 làm cảm biến, (2) tìm span nghi ngờ, (3) retrieve candidate theo **ngữ nghĩa + domain** để thu hẹp kho, (4) rerank theo **âm học** vì từ cần bias chính là từ ASR nghe sai, (5) decode lại. Về lý thuyết chỉ là Bayes `P(e|x,h₀,d) ∝ P(x|e)·P(e|h₀,d)` — ngữ nghĩa nói từ nào hợp lý, âm học nói từ nào thực sự được nói. Không train model lớn, chạy trên ContextASR-Bench và kiểm tiếng Việt bằng Speech-MASSIVE."*

---

## 11. Phụ lục — Ma trận dataset (audio + domain)

**Điều kiện chọn dataset** (bám đúng `(x, d) → B̂ ẩn → ŷ`): mỗi mẫu phải có đủ 3 thứ —
1. **audio** `x`
2. **nhãn domain/lĩnh vực** `d` để điều kiện hóa (KHÔNG phải bias list cho sẵn)
3. **entity/slot ground-truth** để *giấu đi làm đáp án*, dùng chấm bias-discovery — không đưa cho model.

| Dataset | Audio `x` | Domain `d` | Entity GT (để giấu) | Speech thật? | Ngôn ngữ | Khớp | Vai |
|---|:---:|---|---|---|---|:---:|---|
| **ContextASR-Bench** (2025) | ✅ | ✅ coarse context | ✅ named entities (fine) | ✗ TTS synthetic | EN (+ZH) | ★★★★★ | **Chính** — có sẵn đúng bộ 4 `<audio, text, coarse, fine>`; chỉ lộ audio+coarse, giấu fine |
| **Speech-MASSIVE** | ✅ | ✅ 18 domain | ✅ 55 slot type | ✅ người thật | 12 thứ, **có tiếng Việt** | ★★★★☆ | **Việt/đa ngữ** — convert slot value → bias target |
| **SLURP** | ✅ | ✅ 18 scenario | ✅ slot value | ✅ người thật | EN | ★★★★☆ | **Người thật** — kiểm khỏi lệ thuộc TTS |
| **VietMed** | ✅ | ✅ (y tế) | ✅ thuật ngữ y khoa | ✅ | VI | ★★★★☆ | **Việt chuyên sâu 1 domain** (bộ phụ) |
| **AISHELL-NER** | ✅ | ✗ không có taxonomy domain | ✅ NER | ✅ | ZH | ★★★☆☆ | thiếu `d` → tự gán domain / test generalize |
| **MASSIVE** (gốc) | ✗ chỉ text | ✅ 18 domain | ✅ slot | — | 51 thứ | — | thiếu `x` → chỉ làm **KB candidate**, không test được |
| custom entity-rich corpus | ✅ | ✅ | ✅ | tùy | tùy | ★★★★★ nếu build | dự phòng |

> AISHELL-NER **thiếu cột `d`** và MASSIVE gốc **thiếu `x`** → cả hai rớt khỏi "test chính", đúng theo yêu cầu.

**Trục thứ 2 — candidate universe** (mới thật sự quyết định độ khó): cùng 1 câu, kho ứng viên càng to càng khó.

| Setting | S | M | L | XL | Open |
|---|---|---|---|---|---|
| Kho ứng viên | 100 | 1.000 | 10.000 | 100.000 | vô hạn / sinh ra |

→ Ma trận cuối = **dataset × candidate-universe**, không chỉ chọn dataset.

**Số đã kiểm thật trên Speech-MASSIVE tiếng Việt:**
- **2.033 câu** validation (~6.3h), trong đó **1.385 câu có tên riêng** → tập test discovery.
- Kho train nội bộ **11.514 câu** để dựng `E_d`.
- ⚠️ Kho nội bộ **chỉ phủ 56%** số tên trong câu test → 44% là tên đời thật → **phải bổ sung KB ngoài** (vd MusicBrainz). *Chính con số này biện minh vì sao cần retrieval "cao cấp" chứ không chỉ tra kho có sẵn.*
