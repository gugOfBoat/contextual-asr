# Báo cáo duyệt hướng — Chuẩn bị buổi propose với thầy (16/09/2026)

> Mục đích: (1) chốt hướng đi & lý do; (2) ánh xạ gợi ý của thầy vào bức tranh nghiên cứu; (3) kế hoạch triển khai; (4) nội dung trình bày buổi mai + câu hỏi cho thầy.
> Nguồn tổng hợp: `research-proposal.md` (khảo sát gap chi tiết) + gợi ý miệng của thầy.

---

## 0. TL;DR (đọc 30 giây)

- **Bài toán:** Contextual ASR **không có oracle bias list** cho từng câu. Hệ thống chỉ nhận **audio (+ domain thô, tuỳ chọn)** rồi phải **tự tìm ra tập từ khoá (bias list)** để nhận dạng tốt hơn.
- **Gợi ý của thầy = 2 setting lồng nhau**, và cả hai đều **khớp** với 2 setting đã được phân tích trong `research-proposal.md`:
  - **Mở rộng 1** = Setting A: `(audio, domain) → đoán bias list → ASR`.
  - **Mở rộng 2** = Setting B: `audio → nghe lần 1 (đoán domain + context) → nghe lần 2 ASR có context`.
- **Cảnh báo novelty (quan trọng):** Bản thân "đoán bias list từ domain" và "nghe 2 lần" **đã có người làm** (Zhang 2026, CTC-Assisted 2024, Siskos 2025). Nếu propose y nguyên gợi ý → dễ bị phản biện "đã có rồi".
- **Novelty phòng thủ được** (khe hở còn mở): **bất định về domain** (soft `P(d|x)` vs hard argmax), **độ mịn của domain** (coarse→fine, có điểm tối ưu `g*`), và **giao thức benchmark không rò rỉ oracle**.
- **Đề xuất chốt:** đi theo **B + D + E + F** — *domain-conditioned retrieval + two-pass same-utterance + soft/hierarchical domain + benchmark oracle-free*. Tên đề xuất:
  > **Uncertainty-Aware Domain-Conditioned Retrieval for Contextual ASR Without Oracle Bias Lists**
- **Thí nghiệm khởi động (MVP):** tái hiện Zhang trên **CATSLU**, thêm retrieval đơn giản (BM25/dense/acoustic), so **oracle-domain vs hard-domain vs soft-domain**. **KHÔNG train SpeechLLM mới ngay từ đầu.**

---

## 1. Chốt hướng đi & lý do

### 1.1 Hướng chọn
Giữ đúng khung 2 setting của thầy, nhưng **định vị lại đóng góp** để tránh trùng lặp:

| | Gợi ý của thầy | Định vị nghiên cứu (đã sắc bén hoá) |
|---|---|---|
| Mở rộng 1 | audio + domain, chưa có hot-words → đoán bias list | `(x, d, V_d) → C_x → ASR(x, C_x)`: retrieval top-K từ **kho từ vựng domain lớn `V_d` dựng chỉ từ train** |
| Mở rộng 2 | audio-only, nghe 2 lần (lần 1 lấy context, lần 2 ASR) | `(x, V) → h₁, P(d\|x) → C_x → ASR₂(x, C_x)`: dùng chính câu nói làm truy vấn, domain **bất định** |

### 1.2 Vì sao chọn hướng này (chứ không phải "đoán bias list" thuần)
`research-proposal.md` cho thấy 3 mảnh đã bị "chiếm chỗ":

1. **audio → domain → keyword → ASR**: **Zhang et al. 2026 (IEEE Access)** đã làm trên tiếng Trung/CATSLU (giảm KER ~41%). → *Không được* claim đây là mới.
2. **audio/hypothesis → retrieve top-K từ kho lớn**: Huang 2024 (100k), BR-ASR 2025 (200k), RECAST 2025, Hou 2025, GLCLAP, CLAR đã giải quyết. → *Không được* claim "retriever mới".
3. **nghe 2 lần (first-pass → lọc → decode lại)**: **CTC-Assisted LLM-ASR 2024** đã làm same-utterance; **Siskos 2025 (Findings EMNLP)** "RAG-based context discovery for ASR" đã tự tìm context từ kho 466k từ. → *Không được* claim "two-pass" hay "context discovery" là mới.

### 1.3 Khe hở còn mở (novelty phòng thủ được)
Điểm mấu chốt: **Zhang et al. tự nói rằng "lọc tập con từ backend cực lớn nằm NGOÀI phạm vi của họ"** → đây chính là cửa cho mình. Cộng thêm 3 câu hỏi **chưa ai làm có hệ thống** trong nhóm bài gần nhất:

- **G4 — Bất định/sai domain:** chưa có phân tích WER khi domain đúng vs domain sai vs corrupt; chưa có cơ chế abstention (bỏ biasing khi không chắc).
- **G5 — Độ mịn domain (granularity):** `medical → cardiology → electrophysiology` — mịn hơn thì thu hẹp không gian tìm nhưng tăng lỗi phân loại. Giả thuyết có **điểm tối ưu `g*`** (non-monotonic) → kết quả thú vị kể cả khi WER cải thiện khiêm tốn.
- **G6 — Hard vs Soft domain:** `V_hard = V_{argmax P(d|x)}` vs `S_dom(w) = log Σ_d P(w|d)P(d|x)`. Soft giữ được ứng viên từ domain phụ khi phân loại lưỡng lự → tránh "định tuyến thảm hoạ" vào 1 domain sai.
- **F — Benchmark oracle-free:** không nhét từ hiếm của câu test vào kho ứng viên (chống rò rỉ). Đây là lỗ hổng thật giữa "benchmark retrieval kiểu cũ" và "triển khai thực".

> **Thông điệp định vị 1 câu:** *"Domain có đóng góp gì ĐO ĐƯỢC vượt trên retrieval âm học/ngữ nghĩa hiện đại — đặc biệt khi bằng chứng về domain là bất định?"* → biến Zhang/BR-ASR/Huang/RECAST thành **baseline**, không phải đối thủ trùng lặp.

---

## 2. Điều PHẢI tránh claim (để không bị thầy/reviewer bắt lỗi)

Không được nói "lần đầu tiên" với bất kỳ điều nào sau:
- ❌ "first automatic hotword retrieval" (Huang/BR-ASR)
- ❌ "first retrieval-augmented ASR / context discovery for ASR" (Siskos, EMNLP 2025)
- ❌ "first two-pass contextual ASR" (CTC-Assisted 2024)
- ❌ "first audio→domain→keywords" (Zhang 2026)
- ❌ "first acoustic+semantic context" (Fu et al. 2023)

Thay vào đó dùng **claim dạng "formulation + study"**:
- ✅ "Ta *hình thức hoá* contextual ASR **không có oracle list** thành bài toán retrieval + recognition từ KB đóng băng."
- ✅ "Ta *nghiên cứu* retrieval **nhận biết bất định**, tích hợp posterior domain đã hiệu chỉnh với bằng chứng âm học + ngữ nghĩa pass-1; so **hard vs soft** dưới lỗi phân loại có kiểm soát."
- ✅ "Ta *phân tích hệ thống* độ mịn domain, quy mô kho ứng viên, loại distractor; đo đánh đổi B-WER↓ vs U-WER≈0."
- ✅ "Ta *đề xuất giao thức benchmark oracle-free* (từ tham chiếu chỉ dùng để chấm, không đưa vào kho suy luận)."

---

## 3. Kế hoạch triển khai (theo mục 4 CLAUDE.md — mục tiêu kiểm chứng được)

### Giai đoạn 0 — MVP "giết ý tưởng nhanh" (2–3 tuần)
Mục tiêu: xác minh domain-conditioned retrieval có thắng baseline rẻ không, **trước khi** đầu tư lớn.

- **Dataset:** CATSLU (nhanh, tái hiện Zhang, 3 domain music/weather/video).
- **Quy tắc chống rò rỉ:** `V_d` dựng **chỉ từ train**; oracle terms `O_x` trích từ tham chiếu test **chỉ để chấm điểm**.
- **Recognizer:** dùng **contextual recognizer có sẵn** (KHÔNG train SpeechLLM mới) để cô lập phần *discovery*.
- **Ladder baseline:** B0 no-context · B1 oracle (ceiling) · B2 toàn bộ `V_d` · B4 frequency top-K · B5 BM25(h₁) · B6 dense · B7 acoustic · B8 domain+semantic · B9 domain+semantic+acoustic.
- **Sweep K** ∈ {5,10,20,50,100}; đo **Recall@K, B-WER/KER, U-WER**.
- **Thí nghiệm novelty quyết định:** `oracle domain → hard predicted domain → soft P(d|x)`.

**Tiêu chí thất bại (falsification):** ý tưởng yếu nếu
- `R@K_retrieval ≈ R@K_frequency/random`, HOẶC
- `ΔB-WER ≥ 0`, HOẶC
- B-WER giảm nhưng **U-WER tăng đáng kể** (biến context xấu thành lỗi trên câu thường).

### Giai đoạn 1 — Paper mạnh (nếu MVP pass)
- Thêm **SLURP** (18 domain) cho nghiên cứu domain-confusion & soft-domain.
- Thêm **Contextual Earnings-22 / Earnings-21** cho đánh giá thực tế, entity dày.
- Mở rộng: kho `V` từ vài nghìn → 10⁵⁺; distractor 3 tầng (random / near-neighbor ngữ nghĩa / near-neighbor ngữ âm); domain hierarchy coarse→fine; abstention theo ngưỡng confidence; OOD test; đo efficiency (RTF, index size).
- Tiêu chí chấp nhận cốt lõi: **ΔB-WER < 0 trong khi ΔU-WER ≈ 0.**

### Về dataset (đúng câu hỏi thầy: "survey cách tạo dataset")
- Không có benchmark nào có sẵn đủ tổ hợp: *audio + domain phân cấp + KB domain lớn non-oracle + oracle terms chỉ để chấm + distractor thực + giao thức lỗi domain*.
- → Đây là **cơ hội đóng góp F** (benchmark). Cách dựng: `O_i = {w ∈ y_i : hiếm ∨ entity ∨ đặc-thù-domain}`; `V_d` từ transcript train/entity train/KB đóng băng; **tuyệt đối không** `V_d ← y_i`. Báo cáo **KB Coverage** tách khỏi Recall@K.

---

## 4. Nội dung trình bày buổi mai (đề xuất 6–7 slide)

1. **Slide 1 — Bài toán 1 câu:** Contextual ASR khi **không được phát oracle bias list**; input = audio (+ domain thô). Nhấn: chỉ tập trung contextual ASR, **không** ôm multi-speaker (đúng ý thầy).
2. **Slide 2 — Hai setting của thầy, vẽ rõ:**
   - Mở rộng 1: `(audio, domain) → đoán bias list → ASR`.
   - Mở rộng 2: `audio → nghe lần 1 (domain+context) → nghe lần 2 ASR`.
3. **Slide 3 — Trung thực về prior art:** bảng 5 dòng (Zhang / Siskos / CTC-Assisted / BR-ASR / Huang) + 1 câu "cái gì đã bị làm". → cho thầy thấy mình đã survey kỹ.
4. **Slide 4 — Khe hở còn mở (đóng góp của ta):** G4 bất định domain, G5 độ mịn, G6 hard-vs-soft, F benchmark oracle-free. Nhấn câu: *"Zhang tự loại phần lọc backend ra khỏi scope → đây là cửa của mình."*
5. **Slide 5 — Hệ thống đề xuất:** sơ đồ `x → h₁, P(d|x) → S(w)=αS_ac+βS_sem+γS_dom → TopK C_x → (confidence?) → ASR₂`. Nhấn: các khối là prior-art, **đóng góp = cách hình thức hoá + đánh giá bất định**.
6. **Slide 6 — Kế hoạch & MVP:** CATSLU trước, ladder baseline, thí nghiệm `oracle→hard→soft`, tiêu chí B-WER↓ & U-WER≈0. Nói rõ **không train model to ngay**.
7. **Slide 7 — Câu hỏi cần thầy quyết** (mục 5 dưới).

> Mẹo trình bày: mở đầu bằng **đồng thuận với thầy** ("em thấy 2 mở rộng thầy gợi ý rất đúng hướng"), rồi **nâng cấp** bằng phần novelty ("nhưng để tránh trùng Zhang/Siskos, em đề xuất trục bất-định-domain"). Vừa tôn trọng gợi ý, vừa thể hiện đã đọc sâu.

---

## 5. Năm câu hỏi chốt với thầy (để buổi họp ra quyết định, không lan man)

1. **Đóng góp chính là "retriever tốt hơn" hay "nghiên cứu bất định/cấu trúc domain"?** → hướng 2 an toàn hơn nhiều trước Huang/BR-ASR/RECAST.
2. **Cái gì được xem là "hợp lệ có sẵn" lúc triển khai?** global KB / domain KB / lịch sử người dùng / chỉ audio? → quyết định bài toán có thực sự khác retrieval kho-lớn hay không.
3. **Có bắt buộc generalize sang entity chưa thấy khi train không?** Nếu có → `V_d` phải lấy một phần từ **KB ngoài đóng băng**, không chỉ transcript train.
4. **Bất định domain có đủ làm novelty lõi không?** Nếu đủ → dùng component có sẵn, dồn công vào G4–G6.
5. **Deliverable có gồm benchmark oracle-free không?** → tăng đáng kể sức nặng, chống "lối tắt" nhét từ hiếm test vào kho ứng viên.

---

## 6. Rủi ro & lưu ý (phản biện của mình)

- **Rủi ro trùng lặp cao nhất = Siskos 2025 (EMNLP).** Phải trích dẫn nổi bật; điểm khác của mình: Siskos dùng **các đoạn nói TRƯỚC ĐÓ** làm truy vấn (hợp long-form), còn mình dùng **chính câu hiện tại** + **bất định domain**. Cần nói rõ ranh giới này.
- **Rủi ro "đúng nhưng nhỏ":** cải thiện WER có thể khiêm tốn. → Bán bài theo **đánh đổi granularity/uncertainty** (kết quả `g*` non-monotonic + soft cứu hard) vẫn publishable dù WER không nhảy vọt.
- **Kỷ luật U-WER:** luôn báo B-WER và U-WER tách nhau. Hệ thống "tăng recall từ hiếm bằng cách chèn bừa vào câu thường" là **thất bại**, không phải thành công.
- **LLM-generated context chỉ là baseline**, không phải trung tâm: Siskos cho thấy embedding-retrieval ngang/tốt hơn về WER và rẻ hơn nhiều; entity ảo do LLM bịa có thể bị biasing biến thành lỗi chèn.

---

## 7. Việc cần làm ngay sau khi thầy duyệt (checklist)

- [ ] Chốt setting lõi với thầy (A? B? cả hai lồng nhau?).
- [ ] Chốt "cái gì có sẵn lúc deploy" (câu hỏi 2).
- [ ] Tải & khảo sát CATSLU; dựng `V_d` từ train; script trích `O_x` từ test (chỉ để chấm).
- [ ] Chọn contextual recognizer có sẵn (không train mới).
- [ ] Dựng ladder B0–B9 + sweep K; chạy `oracle→hard→soft`.
- [ ] Viết novelty-check ngắn đối chiếu 5 bài đe doạ (Zhang/Siskos/CTC/BR-ASR/Huang).
