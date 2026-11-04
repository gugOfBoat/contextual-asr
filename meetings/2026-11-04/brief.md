# Báo cáo thầy — chốt scorer, dùng vị trí thế nào, kế hoạch tới Interspeech 2027

> Soạn 04/11/2026. Đổi tên thư mục theo ngày họp thật.
> Nguồn số:
> - `scorer/results.md` §7–11
> - `scorer/runs/2026-11-02/REPORT.md`, `s1_breakdown.md`, `gender.md`
> - `scorer/PLAN-chot-scorer.md`
> - `bench/results.md` §5
> - repo Hoàng `itshoang2024/hou25_interspeech` (NOTES F11–F16, `table1.tex`, `brasr_tl.tex`)
>
> Quy ước: **paper nói** / **ta đo** / **ta diễn giải**.
> Whisper = Whisper-large-v3-turbo, chuẩn hoá plain. Qwen = Qwen3-ASR-1.7B, chuẩn hoá wnorm.
> Số viết dạng clean / other.

Mạch nói (khoảng 15 phút): **1** đang làm gì (1 phút) → **2** scorer: số chính (3 phút) → **3** các thí nghiệm và insight (5 phút) → **4** chốt scorer: còn thiếu gì (2 phút) → **5** bước tiếp: dùng vị trí (2 phút) → **6** chọn SpeechLLM (1 phút) → **7** lịch tới 09/02 (1 phút) → **8** câu cần thầy quyết.

---

## 1. Đang nghiên cứu gì (1 câu + 1 hình)

> Bias list càng dài và càng nhiều nhiễu thì SpeechLLM càng (1) bỏ sót từ đúng vì bị loãng, và (2) tự chèn từ trong list dù audio không có.
> Nhóm xây một **scorer nghe audio**: nhận audio + list, trả cho mỗi từ **có được nói không (p)** và **nói ở đâu (t)**. Scorer độc lập với mô hình ASR. Sau đó dùng (p, t) để can thiệp vào SpeechLLM.

```
audio ─┬──────────────────────────────► SpeechLLM (Whisper / Qwen3-ASR / ...) ──► text
       │                                      ▲
       └► scorer(audio, list 2000 từ) ──► top-k / p≥θ   (đã làm: đưa vào prompt)
                                     └──► (từ, p, t_đầu, t_cuối)  (bước tiếp: cổng lúc decode)
```

Đã xong:
- ① chọn từ (scorer);
- ② đưa vào prompt.

Đang mở:
- ③ dùng **vị trí** để can thiệp lúc decode mà không chèn sai.

---

## 2. Scorer pxattn: kiến trúc và số chính

**Kiến trúc (30 giây).** Khác Hou25 đúng một chỗ: từ trong list được biểu diễn bằng **âm vị (g2p)** thay vì **audio TTS**. Phía audio giữ encoder Whisper-large-v3 đóng băng, giống Hou25.
- Mỗi âm vị là một query, dùng cross-attention (2 lớp) "tìm" mình trong audio.
- Gộp [mean; max] → MLP → sigmoid ra p.
- 4,96M tham số; train trên train-clean-100, đúng dữ liệu của Hou25.

**Bảng A. B-WER trên Whisper theo N.** Cùng ASR, cùng prompt naive, top-50 xếp tăng dần; chỉ đổi scorer.

| | N100 | N500 | N1000 | N2000 |
|---|---|---|---|---|
| không prompt (ta chạy) | 9,43 / 18,92 | ← | ← | ← |
| nguyên list (ta chạy) | 8,04 / 13,98 | 8,87 / 18,22 | 9,20 / 18,32 | 9,15 / 18,69 |
| Hou25 (**số paper**) | 5,4 / 10,0 | 5,4 / 10,7 | 5,6 / 11,3 | 5,8 / 11,7 |
| Hou25 tái hiện (**Hoàng**, mlx-whisper) | **4,4 / 8,7** | **4,5 / 8,8** | **4,6 / 9,5** | **4,9 / 9,2** |
| **pxattn top-50** (nhóm) | 4,57 / 8,92 | 4,89 / 9,21 | 5,16 / 9,79 | 5,29 / 10,00 |
| pmaxsim top-50 (nhóm) | 4,60 / 10,30 | 4,95 / 9,53 | 5,17 / 9,83 | 5,47 / 10,37 |

**Bảng B. Qwen3-ASR, list N2000.**

| | WER | B-WER |
|---|---|---|
| không context | 1,63 / 3,41 | 7,40 / 16,08 |
| nguyên list N2000 | 1,69 / 3,55 | 7,40 / 16,45 |
| pxattn top-50 | 1,47 / 2,95 | 5,71 / 11,57 |
| **pxattn lọc p ≥ 0,5** | **1,32 / 2,68** | **4,78 / 9,71** |
| oracle (chỉ từ đúng) | 1,22 / 2,43 | 4,01 / 7,73 |
| list chỉ có nhiễu (`missing_N2000`) → pxattn p ≥ 0,5 | 1,65 / 3,42 | 7,39 / 16,18 |

*Tham chiếu: PTC-Bias (paper, Qwen3-ASR-**0.6B**, khác model nên không so trực tiếp): CTC-Filter 4,81 / 9,45; PTC 4,10 / 7,76.*

**Bảng C. Chọn từ, test-other N2000** (`brasr_tl.tex`).

| | R_B#50 ↑ (từ đúng lọt top-50) | R_B@99 ↓ (cần bao nhiêu từ để giữ 99% từ đúng) | R_H#50 ↓ (từ nghe giống lọt top-50) |
|---|---|---|---|
| Hou25 (Hoàng) | 97,43 | 256,4 | **17,26** |
| pxattn 100 h | 97,71 | 106,7 | 21,58 |
| pmaxsim 100 h | 98,44 | 99,8 | 25,52 |
| pxattn 960 h | **98,59** | **63,9** | 21,47 |
| PTC-Bias (paper) | 99,30 | 16,9 | 22,00 |

**Bảng D. Khoảng tin cậy ghép cặp (bootstrap theo câu, 1.000 lần, Whisper, plain).** Mới chạy 04/11.

| A − B | clean | other |
|---|---|---|
| pxattn N100 − nguyên list N100 | **−3,47** [−6,16; −1,90] | **−5,07** [−5,78; −4,42] |
| pxattn N2000 − nguyên list N2000 | **−3,85** [−4,46; −3,26] | **−8,69** [−9,63; −7,78] |
| pxattn N100 − pmaxsim N100 | −0,03 [−0,28; +0,21] | −1,38 [−5,57; +0,39] |
| pxattn N2000 − pmaxsim N2000 | −0,17 [−0,53; +0,17] | −0,37 [−0,89; +0,14] |
| pxattn − Hou25 (Hoàng) | **chưa có**: cần file hyp từng câu của Hoàng | |

**Câu nói về kết quả (đã sửa so với buổi trước):**
- Buổi trước nói "thắng Hou25 cả 8 ô". Câu đó **chỉ đúng khi so với số trong paper**.
- Trong cùng pipeline với bản tái hiện của Hoàng, pxattn **kém 0,2–0,9 B-WER** (1 seed, chưa có CI).
- Claim đúng là: **chính xác gần bằng Hou25** (recall bằng hoặc hơn), đổi lại:
  - không cần TTS;
  - nhanh hơn khoảng 3 lần (332 so với 922 ms/câu);
  - p có hiệu chỉnh nên lọc theo ngưỡng được (@99: 107 so với 256);
  - cho **vị trí tới mức âm vị**.

---

## 3. Các thí nghiệm quan trọng và insight

| # | Thí nghiệm | Số chính | Insight |
|---|---|---|---|
| E1 | Bảng A: list dài trên Whisper | nguyên list N ≥ 500 ≈ không prompt; scorer top-50 giảm B-WER khoảng một nửa | Với N lớn, list **không vừa prompt** (Whisper chỉ giữ 223 token cuối): ở N1000–2000 chỉ 2–6% từ đúng còn lại trong prompt. Nút thắt là chọn từ, không chỉ là "loãng". |
| E2 | Bảng 2 của Hou25: kiểu prompt | prompt chỉ có từ đúng: B-WER 5,12 nhưng WER **20,08** (clean) | Prompt 0–1 từ làm Whisper **lặp chữ** (khoảng 30 câu, mỗi câu thừa ~430 từ). List hoàn hảo vẫn làm hỏng → **cách mô hình dùng list là nút thắt riêng**. Bọc câu spoken chữa được. Qwen không bị lặp. |
| E3 | Qwen + lọc ngưỡng | p ≥ 0,5: 4,78 / 9,71, gần oracle 4,01 / 7,73 | Sigmoid/BCE cho p dùng được làm ngưỡng → list được phép rỗng. Trên `missing_N2000`, scorer đưa Qwen về đúng mức không context (7,39 so với 7,40): không bị list nhiễu kéo. |
| E4 | Bảng C: chọn từ | R_B#50 ≥ Hou25; R_H#50 21–26 so với 17 | Âm vị **tìm từ đúng** tốt hơn, nhưng **dễ bị lừa hơn bởi từ nghe giống**. Đây là điểm yếu duy nhất rõ ràng. |
| E5 | Cùng pipeline với Hoàng | pxattn kém 0,2–0,9 | Rất có thể do E4. Đã kiểm gián tiếp ở E6; kiểm trực tiếp cần file của Hoàng. |
| E6 | **S1: lỗi từ hiếm đến từ đâu** (`s1_breakdown.md`) | Lỗi "nghe giống" (cách ≤ 2 âm vị) chiếm phần lớn: none 4,96 / 10,24 điểm. Ở N2000, pxattn: từ đúng bị rớt khỏi prompt 1,68 điểm (other); từ nhiễu xuất hiện trong hyp 13 / 54 lần, pmaxsim 70 / 106 lần; số từ nhiễu nghe giống trong mỗi prompt là 1,47–1,53 với pxattn, 1,74–1,81 với pmaxsim | Lỗi còn lại không do thiếu từ trong prompt (97,7% từ đúng đã có ở other) mà do **ASR viết chính tả quen hơn**, cộng với **từ nhiễu nghe giống kéo sai**. Scorer càng để lọt từ nghe giống thì Whisper càng chèn nhiễu. Khớp giả thuyết ở E4. |
| E7 | **Scorer có biết từ ở đâu không** (REPORT 02/11; mốc MFA; 400 câu/tập) | attention lớp 1 của pxattn: sai số 0,03 s, 99% trúng span, 65–67% âm vị có đỉnh đúng đoạn của mình; pmaxsim 0,03–0,04 s (chỉ cả từ); Hou25 S1 0,06 s; cosine thô ngang ngẫu nhiên | Vị trí **tự xuất hiện** dù không giám sát. "Cosine matrix" chỉ mang vị trí khi đó chính là phép so mà model được train. Hai phía khác loại (vector audio / id âm vị) gặp nhau trong phép chiếu q/k của attention; nhánh audio **không** bị "âm vị hoá" (probe ≈ chiếu ngẫu nhiên). |
| E8 | Đỉnh theo thời gian có tách đúng/nhiễu tốt hơn p không | AUC của p 0,995–1,000; của đỉnh 0,97 | Thời gian cho biết **ở đâu**, không thêm thông tin **có hay không**. |
| E9 | Giới tính giọng (biên bản 24/10) | 44 nữ / 29 nam (ước từ F0); mọi CI chứa 0, kể cả Hou25 dùng giọng TTS nữ | Không thấy ảnh hưởng. Scorer âm vị miễn nhiễm theo thiết kế vì phía từ không có "giọng". |
| E10 | Vị trí qua prompt (Hoàng F13–F16, scorer Hou25) | prompt theo thứ tự nói (oracle) −0,69 / −1,36; xếp cả 50 từ theo thời gian **+0,42 / +0,58** (hại); hybrid (chỉ từ p ≥ 0,3) −0,17 / −0,41 | Qua prompt, vị trí giúp **ít**. Phần còn lại (lỗi nghe giống, E6) chỉ chạm tới được khi **can thiệp lúc decode**, và lúc đó vị trí trở thành **cổng để không chèn sai**. |
| E11 | hxattn: cho scorer nhìn thêm hyp lần 1 của Qwen | 6,58 / 14,07, kém hẳn | Nghi model học tắt theo hyp thay vì nghe audio → bỏ. |

**Một câu tổng kết phần scorer:**
> Chọn từ gần chạm trần: 97,7% từ đúng đã có trong prompt, Whisper chép ngang oracle. Lỗi còn lại nằm ở **mô hình ASR viết sai dù từ đã có trong prompt**, và ở **từ nhiễu nghe giống**. Cả hai đều cần thông tin **vị trí**, và scorer đã có sẵn thông tin đó.

---

## 4. "Dứt điểm" scorer: còn thiếu gì

| # | Việc | Trả lời câu nào | Chi phí | Tiêu chí xong | Trạng thái |
|---|---|---|---|---|---|
| S1 | So ghép cặp với Hou25 (Hoàng) + phân rã lỗi | Khoảng 0,2–0,9 có thật không, do đâu | CPU, nửa ngày | có CI; biết phần chênh đến từ đâu | công cụ xong (`bench/error_breakdown.py`, `compare.py`); **chờ Hoàng gửi file** |
| S2 | **Temporal NMS**: hai ứng viên chiếm cùng một khoảng thời gian thì chỉ giữ ứng viên p cao hơn | Vị trí có sửa được điểm yếu từ nghe giống không | GPU khoảng 2 h | R_H#50 giảm ≥ 3 điểm, R_B#50 giảm ≤ 0,1 | chưa |
| S3 | Decode Whisper với top-50 sau NMS + prompt hybrid theo thời gian của pxattn | Có đóng được khoảng cách với Hou25 không | GPU khoảng 3 h | B-WER ≤ Hou25 (Hoàng) trong CI | chưa |
| S4 | 3 seed pxattn + bản 1 lớp | Phương sai do train; kiến trúc gọn hơn có đủ không | GPU 4 × khoảng 5 h | — | chưa |
| S5 | (chỉ khi S2 không đủ) negative khó theo khoảng cách âm vị | Sửa từ nghe giống bằng train | GPU khoảng 5 h | như S2 | dự phòng |

File cần xin Hoàng (thư mục `exp/` không có trong git):
- `exp/scorer_n{100,500,1000,2000}_k50_asc/<subset>.tsv`
- `exp/scorer/is21/scores/<subset>.json`
- `exp/localize/is21/*.json`

Sau S1–S4: **chốt scorer = pxattn**, dùng pmaxsim làm tầng lọc rẻ, đầu ra có vị trí. Không mở thêm biến thể kiến trúc nào nữa.

---

## 5. Bước tiếp: dùng vị trí thế nào (phase decode, theo biên bản 1/11)

Đi từ rẻ đến đắt:

1. **Đầu ra có thời gian** (chỉ thêm phần đọc ra, không train lại): `(từ, p, t_đầu, t_cuối, [thời điểm từng âm vị])`, lấy từ attention lớp 1.
2. **D0 — mổ quá trình decode** (biên bản 24/10). Với các lỗi "nghe giống" còn lại, ghi logit tại bước sinh từ đó:
   - token đúng đứng hạng mấy;
   - cách token được chọn bao xa.

   Trả lời "**cần đẩy xác suất mạnh cỡ nào**". Rẻ: chỉ decode lại các câu lỗi.
3. **D1 — boost có cổng** trên Whisper. Cộng điểm cho các token của bias word (trie, kiểu shallow fusion của Le et al. 2021), **chỉ khi**:
   - (i) p(từ) ≥ τ, **và**
   - (ii) decoder đang "nhìn" vào đoạn audio nằm trong [t_đầu − δ, t_cuối + δ] của từ. Lấy từ cross-attention của Whisper, cùng cơ chế word-timestamp.

   Các arm: không cổng / chỉ p / chỉ thời gian / cả hai.
   Đo B-WER, U-WER và **số lần chèn trên `missing_N2000`**: list không có từ đúng nên mọi lần chèn đều sai. Đây chính là câu hỏi "không chèn khi audio không có" của biên bản 1/11.
4. **D2 — tổng quát sang ASR khác** (mục 6). Nếu không đọc được "đang ở frame nào" thì dùng cổng 2 lượt: hyp lần 1 có timestamp → chỉ boost trong cửa sổ.
5. **D3 — khảo sát tài liệu** về điều chỉnh logit: shallow fusion / trie, TCPGen, CB-Whisper, BPB, PTC-Bias (sửa sau decode). *Mới biết tên và vị trí trong bảng của Hou25 / Hoàng, chưa đọc toàn văn; sẽ đọc trước khi code D1.*

**Ta diễn giải về đóng góp:** scorer + **cổng (có / không) × (ở đâu)** dùng được cho nhiều kiểu ASR, không cần train lại ASR. Vị trí làm hai việc:
- trong scorer: NMS sửa lỗi từ nghe giống;
- trong decode: cổng chống chèn sai.

---

## 6. Chọn SpeechLLM để thực nghiệm

Tiêu chí:
- (1) open weight, chạy bằng HF để **can thiệp được logit** từng bước;
- (2) đại diện **các họ kiến trúc khác nhau**, để trả lời câu "cách nào tổng quát cho các model ASR khác" (biên bản 1/11);
- (3) vừa 1 card 4090 24 GB.

| Họ | Model | Vai trò | Biết "đang ở đâu" bằng cách nào |
|---|---|---|---|
| encoder–decoder | **Whisper-large-v3-turbo** (đang dùng) | Tái hiện Hou25; làm D0/D1 đầu tiên | cross-attention head căn chỉnh (word timestamp) |
| audio encoder + LLM | **Qwen3-ASR-1.7B** (đang dùng) | Backbone SpeechLLM chính; so được với PTC-Bias | *chưa kiểm*: attention từ token sinh ra vào token audio, hoặc cổng 2 lượt |
| audio encoder + LLM, cỡ lớn hơn | ứng viên thứ 3 (vd Qwen2-Audio-7B, Phi-4-multimodal, Voxtral) | Kiểm tính tổng quát | *chưa kiểm*: cần xem model nào chạy được trên 24 GB và có hỗ trợ context |
| CTC / transducer (tuỳ chọn) | ứng viên (vd NeMo Parakeet) | Cho thấy cổng dùng được cả khi không có LLM | thời gian có sẵn theo frame |

Đề xuất: **Whisper + Qwen3-ASR là bắt buộc**, thêm 1 model thứ 3 nếu kịp. *Các ứng viên thứ 3 chưa kiểm license, bộ nhớ và cách đưa context; sẽ kiểm trước khi chọn.*

---

## 7. Lịch tới hạn nộp 09/02/2027 (còn khoảng 14 tuần)

| Tuần | Đến khoảng | Việc | Kết quả mong đợi |
|---|---|---|---|
| 1–2 | 17/11 | S1–S4: chốt scorer | Bảng A–D đủ CI, có NMS; chương "scorer" viết xong bản nháp |
| 3–4 | 01/12 | D0 trên Whisper + Qwen; D3 đọc tài liệu | Biết token đúng ở hạng mấy, cần boost bao nhiêu; chốt thiết kế D1 |
| 5–7 | 22/12 | D1 trên Whisper: 4 arm × N100/N2000 × `missing_N2000` | Bảng chính của paper: B-WER giảm mà số lần chèn không tăng |
| 8–9 | 05/01 | D2: Qwen3-ASR (+ model thứ 3 nếu kịp) | Cùng bảng trên SpeechLLM |
| 10 | 12/01 | Tập thứ hai ngoài LibriSpeech (ConEC / Earnings) | Kiểm domain thật |
| 11–14 | 09/02 | Ablation, viết paper, nộp | — |

Rủi ro:
- (1) D1 không thắng prompt hybrid → paper dừng ở scorer + NMS + prompt; vẫn đủ một bài tập trung vào scorer.
- (2) GPU dùng chung, hiện 4 card đều bận → cần xin lịch card cố định.
- (3) Hoàng chưa gửi file → S1 trễ, nhưng không chặn S2–S4.

---

## 8. Câu cần thầy quyết

1. Đồng ý **định vị đóng góp** là "gần bằng Hou25 nhưng không cần TTS, có ngưỡng, có vị trí", thay cho "thắng Hou25"?
2. Đồng ý **dừng mở rộng kiến trúc scorer** sau S1–S4 và chuyển sang phase decode (D0–D2)?
3. Paper chính là **scorer + cổng thời gian lúc decode**, hay chỉ scorer (an toàn hơn, ít mới hơn)?
4. Model thứ 3: cần không, và nếu cần thì ưu tiên model nào?
5. Lịch GPU: xin card cố định cho tuần 1–2 (khoảng 15 h GPU).
