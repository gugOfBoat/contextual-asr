# Chốt phần scorer: pxattn là gì, hơn/kém gì, còn thiếu gì, đưa vị trí vào thế nào (02/11/2026)

Nguồn số:
- `results.md` §7–§11
- `runs/2026-11-02/REPORT.md`
- repo Hoàng `itshoang2024/hou25_interspeech`: NOTES F9–F16, `docs/latex/tables/table1.tex`, `brasr_tl.tex`, `ranking_tl.tex`

Quy ước: **paper nói** / **ta đo** / **ta diễn giải** được tách rõ.

---

## 1. pxattn khác scorer gốc (Hou25) ở đâu, và vì sao chọn như vậy

| Bộ phận | Hou25 (paper §2.2) | pxattn (nhóm) | Vì sao chọn |
|---|---|---|---|
| Biểu diễn bias word | audio TTS → encoder Whisper (cùng không gian với câu nói) | chuỗi âm vị g2p → embedding + 2 lớp Transformer | Không cần TTS: không tốn 80 phút TTS + 21 GB feature cho kho 209k từ. Từ mới (LLM sinh, ngoài kho) chấm được ngay; Hou25 phải TTS bổ sung (F8 của Hoàng: 2,7% từ test không có TTS). Không có "giọng" ở phía từ nên vấn đề giới tính / giọng TTS biến mất (§4). |
| Phía câu nói | Linear 1280→368 dùng chung | Linear 1280→256 + 1 lớp Transformer | Hai phía khác loại nên không dùng chung phép chiếu. Thêm 1 lớp để audio có ngữ cảnh. Ta đo: nhánh này không "âm vị hoá" audio (probe ≈ chiếu ngẫu nhiên, REPORT Q4), nó chỉ là adapter. |
| So khớp | cross-attention 2 chiều → 4 ma trận cosine (Ts × Tw) → CNN 4 lớp | âm vị là query, attend vào frame audio qua 2 `TransformerDecoderLayer` | Phía từ là âm vị (không có thời lượng), nên không có vệt chéo dốc 1 để CNN bắt. Attention tự học căn chỉnh âm vị → frame. Ta đo: lớp 1 tạo "bậc thang" âm vị đúng thứ tự (REPORT Q2). |
| Gộp | average pool toàn ảnh → linear → softmax | [mean; max] trên âm vị → MLP | max giữ được một khớp ngắn trong câu dài. Hoàng (F14) thấy average pool làm loãng vệt ngắn; với scorer S1-only, đỉnh CAM tách từ tốt hơn chính đầu ra của model. |
| Đầu ra / loss | "softmax ra điểm từng từ" (không nói rõ trên gì) | sigmoid, BCE theo từng cặp | Ra xác suất độc lập, nên **lọc được theo ngưỡng và list được phép rỗng**. Trên Qwen, p ≥ 0,5 là cấu hình tốt nhất: 4,78 / 9,71 so với top-50 5,71 / 11,57 (N2000). |
| Chọn từ | top-50, tăng dần | top-50 (như Hou25) hoặc p ≥ θ | Như trên. |
| Biến thể rẻ | – | pmaxsim (kiểu ColBERT: max theo thời gian, rồi trung bình) | Phía từ tính trước được cho cả kho, nên chấm 209k từ chỉ cần một phép nhân ma trận. Dùng làm tầng lọc / chế độ không có list. |

## 2. Kết quả: hơn gì, kém gì (nói thẳng)

**So với Hou25 tái hiện tốt, cùng pipeline** (bảng `table1.tex` của Hoàng; mlx-whisper, cùng prompt; B-WER clean / other; 1 seed, chưa có CI):

| | N100 | N500 | N1000 | N2000 |
|---|---|---|---|---|
| Hou25 full scorer (Hoàng, TTS) | **4,4 / 8,7** | **4,5 / 8,8** | **4,6 / 9,5** | **4,9 / 9,2** |
| pxattn (nhóm) | 4,6 / 9,0 | 4,8 / 9,3 | 5,1 / 9,9 | 5,1 / 10,1 |
| pmaxsim (nhóm) | 4,5 / 8,8 | – | – | 5,3 / 10,2 |

- **Câu "pxattn thắng Hou25 cả 8 ô" (§10) chỉ đúng khi so với số trong paper.** Trong cùng pipeline, so với bản tái hiện 3 epoch của Hoàng, pxattn **kém 0,2–0,9 điểm B-WER**, lệch nhiều nhất ở N lớn trên test-other.
- Hou25 bản của nhóm (2 epoch) kém bản của Hoàng: R_B#50 97,77 / 95,39 so với 98,81 / 97,43.

**Chọn từ** (test-other N2000, `brasr_tl.tex`):

| | R_B#50 ↑ | R_B@99 ↓ | R_H#50 ↓ (từ nghe giống lọt top-50) |
|---|---|---|---|
| Hou25 full (Hoàng) | 97,43 | 256,4 | **17,26** |
| pxattn 100h | 97,71 | 106,7 | 21,58 |
| pmaxsim 100h | 98,44 | 99,8 | 25,52 |
| pxattn 960h | **98,59** | **63,9** | 21,47 |
| PTC-Bias (paper) | 99,30 | 16,9 | 22,00 |

**Ta diễn giải điểm mạnh / yếu:**
- **Mạnh:**
  - Tìm từ đúng ngang hoặc hơn: R@50 ≥ Hou25 ở mọi N.
  - Ngưỡng hiệu chỉnh tốt hơn nhiều: @99 107 so với 256.
  - Không cần TTS, nhanh hơn khoảng 3 lần (332 so với 922 ms/câu).
  - Định vị mịn hơn: 0,03 s, tới từng âm vị, so với 0,05–0,06 s.
- **Yếu:** **từ nghe giống** (H#50 21–26 so với 17). Đây rất có thể là nguyên nhân của khoảng 0,2–0,9 B-WER.
  - `results.md` §10b: với top-50 của pxattn N2000, Whisper chèn từ nhiễu 13 / 51 lần (clean / other); với list ngẫu nhiên chỉ 1–3 lần.
  - Tức từ nghe giống lọt vào prompt và kéo Whisper viết sai. *(Chưa kiểm trực tiếp: cần đếm số lần chèn trong pipeline của Hoàng cho cả hai scorer.)*

**Cách định vị đóng góp (đề xuất):**
- Không claim "chính xác hơn Hou25".
- Claim: **chính xác tương đương (B-WER trong khoảng 0,2–0,9; recall bằng hoặc hơn) nhưng không cần TTS, rẻ hơn, có xác suất dùng được để lọc theo ngưỡng, và cho vị trí tới mức âm vị.**
- Sau đó sửa điểm yếu từ nghe giống bằng chính thông tin vị trí (§5b).

## 3. Đối chiếu biên bản họp với những gì đã có

| Biên bản | Câu hỏi | Trạng thái | Câu trả lời / nguồn |
|---|---|---|---|
| 24/10 | Thêm thông tin vị trí thì performance đổi thế nào? | ✅ với thời gian của Hou25 (Hoàng); ❌ với thời gian của pxattn | F13: thứ tự nói (oracle) −0,69 / −1,36 B-WER; trần thực tế so với thứ tự theo điểm là −0,20 / −0,71. F15: xếp cả 50 từ theo thời gian thì **hại** (+0,42 / +0,58) vì từ nhiễu không có thời gian thật. F16 hybrid (chỉ từ p ≥ 0,3 xếp theo thời gian, đặt cuối): −0,17 / −0,41 |
| 24/10 | Benchmark câu có nhiều bias word | ✅ | F11 (Hoàng) + `results.md` §10: câu ≥ 2 từ hưởng lợi bằng hoặc hơn câu 1 từ |
| 24/10 | Thống kê WER nhóm 1 từ / nhiều từ | ✅ | như trên; oracle 1 từ làm Whisper lặp chữ |
| 24/10 | Quá trình decode của Whisper tại vị trí bias word | ❌ chưa làm | đề xuất thí nghiệm D0 (§6) |
| 24/10 | Giới tính giọng: audio câu nói và audio của từ | ✅ (mới, 02/11) | Không thấy chênh lệch nữ / nam ở cả 3 scorer, kể cả hou25 dùng giọng TTS nữ (CI chứa 0, `runs/2026-11-02/gender.md`). Scorer âm vị miễn nhiễm theo thiết kế. Giới tính ước lượng từ F0. |
| 1/11 | Whisper có cần thông tin vị trí? | ✅ một phần | Qua prompt: có nhưng nhỏ (≤ 0,2 / 0,7). Phần lớn còn lại là lỗi ngữ âm, 3,1 / 5,6 điểm (F12). Phần này chỉ với tới được bằng can thiệp lúc decode, và **vị trí trở thành điều kiện để không chèn sai** |
| 1/11 | Điều chỉnh logit, cách nào tổng quát cho ASR khác | ❌ chưa khảo sát | §6 D1–D3 |
| 1/11 | Tránh viết bias word khi audio không có | ❌ | gate = xác suất (có / không) × cửa sổ thời gian (ở đâu). Đo trên điều kiện `missing_N2000` đã có trong benchmark |

## 4. Vị trí: đã chứng minh scorer của ta có (REPORT 02/11)

| Bản đồ | Định vị từ (clean · other) | Mức âm vị |
|---|---|---|
| pxattn attention lớp 1 | 0,03 s, 99,5% · 0,03 s, 99,2% | có: 65–67% âm vị có đỉnh trong đúng đoạn của nó; ma trận 39×39 có đường chéo |
| pmaxsim cosine | 0,04 s · 0,03 s | không: chỉ "dải dọc" theo cả từ |
| hou25 (nhóm) S1 | 0,06 s · 0,06 s | (frame TTS) |
| pxattn attention lớp 2 / cosine thô | ~50% / ngẫu nhiên | – |

## 5. Đưa vị trí vào scorer như thế nào (đề xuất, chưa code)

Thứ tự từ rẻ đến đắt. Bước a, b, c **không cần train lại**.

**a. Đầu ra mới của scorer** (chỉ thêm phần đọc ra).
- Mỗi ứng viên trả về `(từ, p, t_đầu, t_cuối, [thời điểm từng âm vị])`.
- Thời gian lấy từ attention lớp 1 của pxattn: đường chéo tốt nhất cho cả từ, đỉnh từng hàng cho từng âm vị.

**b. Loại trùng theo thời gian (temporal NMS).** Đây là đề xuất chính để sửa điểm yếu từ nghe giống.
- *Ý tưởng:* từ nhiễu nghe giống từ đúng sẽ được scorer đặt **vào đúng chỗ của từ đúng** (cùng âm thì cùng chỗ). Hai ứng viên chiếm cùng một khoảng thời gian thì chỉ giữ ứng viên p cao nhất, hoặc trừ điểm các ứng viên còn lại.
- *Đo trên mức chọn từ (rẻ):* R_H#50 phải giảm, R_B#50 không được giảm. Sau đó đo B-WER và số lần chèn của Whisper.
- *Ta diễn giải:* đây là cách dùng vị trí **bên trong** scorer để sửa chính điểm yếu đang khiến nó kém Hou25. Cách này không cần train lại và không đụng ASR.
- *Rủi ro:*
  - từ ghép ("grave digger" / "gravedigger") chồng thời gian hợp lệ;
  - hai từ đúng sát nhau.
  → Ngưỡng chồng lấp (IoU) chọn trên dev.

**c. Prompt hybrid với thời gian của pxattn.**
- Làm lại F16 của Hoàng bằng thời gian của scorer nhóm: từ p ≥ τ xếp theo thời gian, đặt cuối prompt.
- Kỳ vọng khoảng −0,2 / −0,4 B-WER. Chỉ cần decode Whisper.

**d. Biasing lúc decode có cổng thời gian.** Đây là phase sau và là đóng góp chính tiềm năng (§6).

**e. (Tuỳ chọn) Huấn luyện có giám sát vị trí.**
- Thêm loss căn chỉnh bằng MFA của train-clean-100 (Lugosch có sẵn cho cả 960 h).
- Chưa cần: vị trí đã tự xuất hiện mà không cần giám sát. Chỉ làm nếu b hoặc d cần thời gian chính xác hơn.

**Đơn giản hoá kiến trúc (tuỳ chọn):** lớp 2 không định vị. Thử pxattn **1 lớp**: nếu chọn từ ngang thì kiến trúc gọn hơn, và vị trí đọc ra từ chính lớp quyết định.

## 6. Kế hoạch "dứt điểm" scorer

| # | Việc | Trả lời câu hỏi | Chi phí | Tiêu chí dừng |
|---|---|---|---|---|
| S1 | So **ghép cặp** pxattn với Hou25 (Hoàng) trong cùng pipeline: CI bootstrap theo câu cho B-WER; đếm số lần chèn từ nhiễu và số lỗi từ nghe giống của từng hệ | Khoảng 0,2–0,9 điểm có thật không, có phải do từ nghe giống không | CPU, ~½ ngày (cần file hyp của Hoàng) | Có CI; biết phần chênh đến từ đâu |
| S2 | **Temporal NMS** (5b) trên ranking N100 / N2000, đo R_B#50, R_H#50 | Vị trí có sửa được điểm yếu từ nghe giống không | Cần rank đầy đủ kèm thời gian: GPU ~2 h (hoặc CPU trên tập con) | H#50 giảm ≥ 3 điểm, #50 giảm ≤ 0,1 |
| S3 | Decode Whisper: top-50 sau NMS; hybrid theo thời gian của pxattn (5c) | Chênh lệch với Hou25 có đóng lại không | GPU ~3 h | B-WER ≤ Hou25 (Hoàng) trong CI |
| S4 | 3 seed pxattn (+ biến thể 1 lớp) | Phương sai do train; kiến trúc gọn hơn có đủ không | GPU 4 × ~5 h | — |
| S5 | (nếu S2 không đủ) negative khó theo **khoảng cách âm vị** thay vì chính tả | Sửa H#50 bằng train | GPU ~5 h | như S2 |

Sau S1–S4: **chốt** scorer = pxattn (+ pmaxsim làm tầng rẻ), với đầu ra có vị trí. Viết chương "scorer":
- Hou25 tái hiện (Hoàng)
- scorer âm vị
- chọn từ + B-WER trên Whisper và Qwen
- vị trí (REPORT)
- NMS

**Phase sau (can thiệp lúc decode, theo biên bản 1/11):**
- **D0 — mổ quá trình decode** (biên bản 24/10).
  - Với ~700 lỗi còn lại (F12: Whisper nghe ra nhưng viết chính tả quen hơn), ghi logit tại bước sinh từ đó: token đúng đứng hạng mấy, cách token được chọn bao xa.
  - Từ đó biết "đẩy xác suất lên" cần mạnh cỡ nào, và nên đẩy bao nhiêu mới không chèn sai. Rẻ: chỉ decode lại các câu lỗi bằng HF.
- **D1 — boost có cổng:** cộng điểm cho các token của bias word (theo trie, kiểu shallow fusion của IS21 / Le et al. 2021), **chỉ khi** (i) p(từ) ≥ τ và (ii) vị trí audio hiện tại của decoder nằm trong [t_đầu − δ, t_cuối + δ] của từ.
  - Vị trí audio hiện tại lấy từ cross-attention của Whisper (cơ chế word-timestamp dùng các head căn chỉnh).
  - Arm: không cổng / chỉ p / chỉ thời gian / cả hai.
  - Metric: B-WER, U-WER, số lần chèn trên `missing_N2000` (list không có từ đúng nên mọi lần chèn đều sai).
- **D2 — tổng quát hoá:** cùng cổng cho Qwen3-ASR (decoder là LLM, nhìn token audio). Cần khảo sát cách biết "đang ở frame nào". Nếu không có thì dùng cổng 2 lượt: hyp lần 1 có timestamp → chỉ boost trong cửa sổ.
- **D3 — khảo sát tài liệu** điều chỉnh logit:
  - shallow fusion / trie (Le 2021)
  - BPB (Sudo 2024)
  - TCPGen trên Whisper (Sun 2023)
  - CB-Whisper (Li 2024)
  - PTC-Bias (sửa sau decode)

  *Mới biết tên và vị trí trong bảng của Hou25 / Hoàng, chưa đọc toàn văn.*
