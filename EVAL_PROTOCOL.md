# Protocol đánh giá scorer (bản thống nhất, 17/10/2026)

> Câu hỏi khoa học: **cho audio + bias list, scorer có xếp từ thật sự được nói lên trên các từ nhiễu không — và việc đó giúp ASR được bao nhiêu?**
> Mọi thí nghiệm scorer từ nay theo file này. Nhãn nguồn: **[A]** theo nguồn (IS21/Hou25/PTC-Bias/BR-ASR), **[B]** thông lệ, **[C]** nhóm tự thiết kế.

## 0. Các paper cùng nhánh thiết kế thực nghiệm thế nào (đã đọc, 17/10)
| Paper | Đo riêng bước chọn từ? | Nhiễu khó (đồng âm)? | Đo ASR | Học được gì |
|---|---|---|---|---|
| **Hou25** (Interspeech 2025, §3.4) | **không** — chỉ WER, B-WER | không | Whisper-turbo, N 100…2000 | — (thiếu sót của họ) |
| **CTC-Filter** (Yang et al., arXiv 2411.06437, SLT 2024) | **không** | không | SLAM-ASR; có hàng "GT Hotwords" = trần; ablation 3 cách lọc F1/F2/F3 (Bảng 3) | baseline lọc **không train**: hyp CTC thô + so edit distance với list |
| **BR-ASR** (Gong et al., arXiv 2505.19179, Interspeech 2025) | **có** — Bảng 2: Recall_B@99, Recall_B#50, Recall_H#50, đặt cạnh WER/B-WER, test-other N2000 | **có** — đồng âm = khoảng cách Levenshtein âm vị ≤ 2 (dùng khi train; Recall_H đo trên test) | nhiều ASR; kho tới 200k từ (Hình 2) | metric chuẩn cho bước chọn từ; đo riêng "đồng âm lọt top" |
| **PTC-Bias** (arXiv 2609.28727) | **có** — Bảng 3, cùng 3 metric của BR-ASR | **có** — Recall_H#50 | Bảng 1: N 100…2000, tách hàng "chỉ chọn" và "chọn + sửa"; **Qwen3-ASR-0.6B** (không phải 1.7B), posterior WavLM (†) hoặc encoder của Qwen (AuT) | tách đóng góp từng bước; báo độ trễ (Hình 3) |
| Novitasari (arXiv 2604.12398) | không | không | list NER 10 / 200 từ; hàng "gợi ý bằng âm vị" = trần | cách *dùng* từ đã chọn (gợi ý phát âm) |

Rút ra: (1) đo riêng bước chọn từ là **thông lệ của 2 paper gần nhất** (BR-ASR, PTC-Bias) — dùng đúng tên metric của họ để so được; (2) "nhiễu nghe giống" đã có tiền lệ (Recall_H); (3) baseline đơn giản = CTC-Filter; (4) luôn có hàng trần (oracle / GT hotwords).

**Sửa số liệu cũ:** PTC-Bias 4.10 / 7.76 và CTC-Filter 4.81 / 9.45 là trên **Qwen3-ASR-0.6B** + posterior WavLM, không phải 1.7B → không đặt cạnh trực tiếp số 1.7B của nhóm.

## 1. Bốn nguyên tắc
1. **Chọn trên dev, báo trên test, test chạy một lần.** Checkpoint, ngưỡng θ, K đều chọn trên dev. [B]
   Số hiện có (p ≥ 0.5 → 4.78) dùng θ chọn khi đã nhìn test → **sơ bộ**, chạy lại theo protocol này.
2. **Đo tách 3 tầng:** (a) scorer xếp hạng đúng không (không cần ASR); (b) ASR tốt lên bao nhiêu; (c) cách *dùng* output scorer (mục 4).
3. **Mọi con số đặt giữa sàn và trần**, và có baseline không train. Scorer không thắng baseline đơn giản thì không có ý nghĩa.
4. **Cùng ASR mới so số tuyệt đối.** Khác ASR thì chỉ so G (mục 3).

## 2. Dữ liệu
Đơn vị: mỗi **lượt từ hiếm trong reference** (cột 3 `ref.tsv`, giống B-WER); từ hiếm = ngoài `common_words_5k`. [A]

| Mã | List mỗi câu | Đo điều gì | Nguồn |
|---|---|---|---|
| **L1** `N{100,500,1000,2000}` | từ đúng + N từ hiếm **ngẫu nhiên** (IS21 gốc) | so với Hou25, PTC-Bias, BR-ASR | [A] |
| **L2** `hard_N2000` | L1-N2000, thay một phần nhiễu bằng từ **gần đồng âm** với từ đúng (Levenshtein âm vị ≤ 2, theo BR-ASR) | phân biệt từ dễ nhầm (danh bạ có *Ann / Anne / Anna*) | định nghĩa đồng âm [A]; cách chèn vào list test [C] — paper không mô tả |
| **L3** `missing_N{100,2000}` | chỉ có nhiễu | scorer có biết "không chọn gì" | [C] |
| **L4** kho 209k | không có list | bài toán mở; chẩn đoán | [C]; BR-ASR có thử tới 200k |

- L2: nhiễu trùng hệt âm vị với từ đúng (đồng âm hoàn toàn) audio không phân biệt được → đếm riêng, không tính là lỗi scorer.
- **Dev:** IS21 chỉ có list cho test → tự dựng cho dev-clean + dev-other đúng quy tắc IS21 (seed cố định), đủ L1–L3. Dev cũ (500 câu, N100) quá dễ → bỏ.

## 3. Metric

### Tầng (a) — scorer xếp hạng (không cần ASR) — mở rộng `bench/eval_retrieval.py`
| Metric | Định nghĩa | Nguồn |
|---|---|---|
| **Recall_B#K**, K ∈ {10, 50} | % lượt từ đúng nằm trong top-K | [A] BR-ASR, PTC |
| **Recall_B@99** | số từ trung bình phải giữ để đạt 99% recall (thấp = tốt) | [A] |
| **Recall_H#50** | % từ gần đồng âm lọt top-50 (thấp = tốt). **Theo BR-ASR (2505.19179, §2.2.3 Eq. 4 + chú thích 1–2 §3.1.1):** H = mọi từ **trong list** có Levenshtein âm vị ≤ 2 với một từ đúng (kể cả đồng âm hoàn toàn), tính trên **L1 N2000**. Bản cũ của nhóm (`near_N2000`, chỉ 20 từ gần nhất trong kho, bỏ khoảng cách 0) **không khớp** → phải sửa trước khi so. L2 giữ làm phân tích phụ | [A] |
| Tại θ*: Recall, Precision, độ dài list | θ* chọn trên dev; lý tưởng: độ dài = số từ đúng (~2,2 / câu) | [C] |
| Giữ nhầm trên L3 | số từ giữ / câu, % câu rỗng | [C] |
| Recall trên **từ khó** | chỉ từ Qwen lần 1 chép sai (`pass1_*.jsonl`) | [C] |
| ms / câu | | [A] PTC báo độ trễ |

Báo chính ở **test-other N2000** (giống Bảng 2 BR-ASR / Bảng 3 PTC), đủ bảng thì cả clean.

### Tầng (b) — ASR — `bench/score.py` (scorer IS21 chính thức)
- **B-WER** (chính), **U-WER** (không được tăng), WER. Báo wnorm (số chính) + plain (đặt cạnh paper). [A]/[B]
- **G = % khoảng trống lấp được** = `(B_cả_list − B_x) / (B_cả_list − B_oracle)`; 0% = như đưa cả list, 100% = bằng oracle, > 100% chỉ khi sửa output (như PTC). [C]
- **Phân rã lỗi theo lượt từ hiếm** — trả lời "lỗi của ai": [C]
  - *scorer bỏ sót*: từ đúng không nằm trong list đưa vào ASR;
  - *ASR không dùng được*: có trong list nhưng ASR vẫn chép sai;
  - *đúng*.
- Paired bootstrap theo câu (1.000 lần), khoảng tin cậy 95% của ΔB-WER; chỉ viết "tốt hơn" khi không chứa 0. [B]

## 4. Tầng (c) — cách dùng output của scorer
Scorer cho ra *điểm từng từ* (và có thể *vị trí* trong audio). Dùng thế nào là một trục riêng — đo bằng **list oracle** trước để biết trần của từng cách, sau đó mới thay bằng list của scorer:

| Cách dùng | Nguồn |
|---|---|
| top-K (K = 50 Hou25; ≤ 10 PTC, BR-ASR cho SLAM) | [A] |
| ngưỡng θ (độ dài thay đổi, có thể rỗng) | [A] PTC (η = 0.05) / [C] |
| thứ tự trong prompt (tăng dần) | [A] Hou25 Bảng 3 |
| thêm gợi ý phát âm cho từ đã chọn | [A] Novitasari (họ fine-tune; Qwen đóng băng → phải thử) |
| sửa sau decode dùng vị trí từ trong audio | [A] PTC Correction |

Đọc: nếu oracle + cách X vẫn sai nhiều → giới hạn ở phía ASR, cải thiện scorer không giúp thêm.

## 5. Các hệ được so
| Hệ | Vai | Train? |
|---|---|---|
| Ngẫu nhiên | sàn | không |
| **CTC-Filter kiểu Yang 2024** (hyp lần 1 của Qwen + edit distance âm vị với list) | baseline đơn giản | không |
| PTC Retrieval (code của họ) | SOTA chọn từ | có sẵn |
| Hou25 tái hiện (TTS + CNN) | đối thủ trực tiếp | có |
| pxattn, pmaxsim (+ bản cải tiến) | của nhóm | có |
| Oracle | trần | — |

ASR: **Qwen3-ASR-1.7B** (mục tiêu), **Whisper-large-v3-turbo** (đặt cạnh Hou25), **Qwen3-ASR-0.6B** (đặt cạnh PTC-Bias Bảng 1, tuỳ chọn).

## 6. Đọc kết quả → quyết định
| Nếu thấy | Nghĩa là | Làm tiếp |
|---|---|---|
| Mọi hệ Recall_B#50 ≈ 99% trên L1 (PTC, BR-ASR đã vậy) | L1 bão hoà | phát triển scorer theo Recall_H#50, Recall_B@99, từ khó |
| CTC-Filter ≈ scorer | audio không thêm gì so với hyp | xem lại kiến trúc |
| Phân rã: phần lớn lỗi là "ASR không dùng được" | scorer đủ tốt | chuyển sang tầng (c): gợi ý phát âm, sửa sau decode |
| Recall_H#50 cao | scorer không phân biệt từ nghe giống | negative gần hơn (curriculum như BR-ASR), khớp theo thứ tự âm vị |
| Giữ nhầm nhiều trên L3 | list sai sẽ gây hại | hiệu chỉnh xác suất / ngưỡng |
| pxattn ≥ Hou25 tái hiện, có ý nghĩa thống kê | "âm vị thay TTS + ngưỡng" đứng được | viết thành đóng góp |

## 7. CHỐT (18/10) — thực nghiệm của SCORER (tầng a)
Phần ASR (tầng b, c: đưa list vào Qwen / Whisper, cách dùng output) **chưa bàn** — chốt sau.

Báo số **chỉ trên test**. Dev-other chỉ dùng để chọn checkpoint / biến thể scorer — giống COALA validate trên dev-clean [B].

| Bảng | Nội dung | Split / list | Hàng |
|---|---|---|---|
| **Chọn từ** | Recall_B#10, #50, Recall_B@99, Recall_H#50, ms/câu | test-clean + test-other, N2000 (L1, định nghĩa BR-ASR) | ngẫu nhiên · hypmatch · Hou25 tái hiện · PTC Retrieval (code của họ) · BR-ASR (số paper, chỉ other) · **ours** |
| **Ablation scorer** | Recall_B#10, K99 | test-other N2000 | mỗi thay đổi trong `scorer/DEEPDIVE.md` §4 đã giữ lại |
| **Phân tích** | recall theo từ Qwen chép sai / độ dài âm vị / độ dài câu; `missing_N2000` (số từ giữ nhầm) | test | — |

**Quy tắc chọn (cố định trước khi chạy test):** checkpoint / biến thể = Recall_B#10 cao nhất trên **dev-other N2000**.

## 8. Thứ tự làm
1. ✅ (17/10) `bench/build_lists.py` (dev + L2), `eval_retrieval.py`, `compare.py` (bootstrap + G), `score.py --ctx` (phân rã). Số: `scorer/results.md` §6.
2. ✅ (17/10) `bench/baseline_hypmatch.py` trên hyp Qwen lần 1 → tầng (a) xong; chưa đưa vào ASR.
3. Tầng (a) cho pxattn / pmaxsim trên dev → mốc dev cho vòng tối ưu. — GPU ngắn.
4. Tối ưu scorer theo `scorer/DEEPDIVE.md` §4 (đo trên dev-other).
5. Hou25 tái hiện + PTC Retrieval → tầng (a).
6. Chạy test một lần cho bảng ở §7.
7. Phần ASR (tầng b, c) — bàn sau.
