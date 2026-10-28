# Đề xuất (nháp, 18/10/2026): scorer vừa XẾP HẠNG vừa ĐỊNH VỊ bias word

> Ý tưởng thô của user: scorer xếp hạng chuẩn + "biết" từ bias nằm ở đâu trong audio (và có phải từ khó không) → module phía sau (SpeechLLM) có cả 2 thông tin: **từ gì** và **ở đâu**.
> Trạng thái: **chưa duyệt**. Chưa code.

## 1. Đầu ra mong muốn của scorer
Với mỗi câu + list N từ, trả về các bộ:
```
(từ, điểm, [t_bắt_đầu, t_kết_thúc], cờ "khó")
vd: ("fauchelevent", 0.93, [1.84 s, 2.61 s], khó = hyp lần 1 ở đoạn này là "fauchelevant")
```
- **điểm** → xếp hạng / lọc (đã có).
- **vị trí** → module sau biết *đoạn nào* cần sửa / nhắc. Mới.
- **khó** = đoạn đó trong hyp lần 1 (timestamp từ Qwen3-ForcedAligner) **khác** từ bias → chỉ những chỗ này mới cần can thiệp. Cách này thay cho hxattn, vốn đã hỏng vì học tắt theo hyp.

## 2. Ai đã làm gì (đã kiểm nguồn 18/10)
| Paper | Có vị trí? | Dùng vị trí để làm gì | SpeechLLM | Đo độ chính xác vị trí? |
|---|---|---|---|---|
| **PTC-Bias** (2609.28727, §2.4–2.5) | có — khoảng [a_k, b_k] = đường CTC tốt nhất | **sửa sau decode**: so điểm CTC của từ bias với đoạn hyp trùng khoảng (ngưỡng 2.0) | **đóng băng** (§3.1) | **không** (§3.3 chỉ có Recall / WER) |
| **Novitasari** (2604.12398, §2.3, Hình 1) | có — tagger CTC gán nhãn bias / non-bias / khoảng trắng theo frame | chỉ là **loss phụ khi train** (multi-task); **bỏ đi khi suy luận** | fine-tune LoRA (Granite-Speech) | không |
| Hou25, BR-ASR, COALA, CTC-Filter | không | — | — | — |

→ "**Có vị trí**" thì **không mới** (PTC có rồi). Chỗ còn trống:
1. **Đo** độ chính xác định vị — chưa ai báo.
2. Vị trí do một **scorer học được** (audio + âm vị) sinh ra, thay vì trie + CTC Viterbi.
3. Dùng vị trí theo cách khác *sửa sau decode*: cho SpeechLLM đóng băng **decode lại có nhắc từ, chỉ ở đúng đoạn đó** (mục 4b), và dùng cờ "khó" để chỉ can thiệp chỗ cần.

## 3. Scorer định vị — kiến trúc (mổ xẻ ở `scorer/DEEPDIVE.md`)
Định vị đi cùng thành phần **② căn chỉnh** trong DEEPDIVE: muốn biết từ ở đâu thì phải khớp chuỗi âm vị theo thứ tự lên trục thời gian.

| Phương án | Vị trí lấy từ đâu | Ưu | Nhược |
|---|---|---|---|
| A. pxattn + **giám sát attention** bằng timestamp thật | trọng tâm của attention | sửa ít code | attention không bảo đảm liền và đúng thứ tự |
| B. **head CTC âm vị** trên encoder Whisper + Viterbi trong trie (như PTC) | đường căn chỉnh | chuẩn, có sẵn khoảng | gần PTC → phần mới ít |
| C. **bản đồ "từ × thời gian"**: mỗi ứng viên cho một đường điểm theo frame, điểm = max theo đoạn, vị trí = đoạn max (kiểu phát hiện keyword) | đoạn có điểm cao nhất | xếp hạng và định vị ra **cùng lúc**, học được | cần nhãn timestamp, tốn bộ nhớ khi N = 2000 → dùng cascade (pmaxsim lọc 200 → C) |

Nhãn timestamp cho train / đo: alignment MFA công khai của LibriSpeech (**cần kiểm** nguồn / độ phủ), hoặc chạy Qwen3-ForcedAligner trên transcript gốc.

**Metric định vị** (thêm vào tầng (a)): % từ đúng có khoảng dự đoán chồng khoảng thật (IoU ≥ 0.5), sai lệch tâm (ms). Tham khảo ATWV của spoken term detection [B].

## 4. Module phía sau (SpeechLLM) dùng vị trí thế nào — xếp theo độ khả thi với Qwen đóng băng
| # | Cách dùng | Đóng băng? | Mới? |
|---|---|---|---|
| a | **Gating**: chỉ đưa vào prompt từ có điểm cao **và** cờ "khó" → list ngắn, ít distractor, ít chèn từ | ✅ | vừa (bản không vị trí đã thử = "đáng bias") |
| b | **Decode lại cục bộ**: cắt đoạn [t₀ − δ, t₁ + δ], decode lại với prompt chỉ có 1–2 từ đó, ghép vào hyp lần 1 (timestamp từ ForcedAligner) | ✅ | **cao** — chưa thấy ai làm |
| c | Sửa sau decode bằng điểm âm vị (như PTC) | ✅ | thấp |
| d | Ghi vị trí vào prompt text ("khoảng 1.8 s: fauchelevent") | ✅ | Qwen không được train như vậy → nhiều khả năng bị bỏ qua |
| e | Gating trong mô hình / căn chỉnh prompt theo âm vị / **RL–DPO–GRPO phạt chèn từ không có bằng chứng âm thanh** | ❌ phải fine-tune | RL cho biasing đã có: RLBR (2601.13409), Kong et al. (2512.21828, GRPO + distractor + giảm hallucination) → chỉ mới nếu reward dùng **vị trí** |

**Mâu thuẫn cần chốt:** đã chốt Qwen3-ASR-1.7B **đóng băng** (STATE §1). Nhóm (e) đòi fine-tune → hỏi thầy, hoặc để làm phần mở rộng.

## 5. Kiểm tra khả thi trước (go / no-go), làm trước khi code scorer định vị
**Thí nghiệm oracle:** từ đúng + **vị trí thật** → cách (b) trên Qwen đóng băng, so với oracle list thường (B-WER 4.01 / 7.73).
- Nếu (b) với oracle **thấp hơn rõ 4.01 / 7.73** → vị trí có giá trị, đáng làm scorer định vị.
- Nếu **không** → vị trí không giúp được Qwen đóng băng, quay về tối ưu xếp hạng (E0–E5) + gating (a).

Chi phí: lấy timestamp + ~2 lượt Qwen trên test-other (GPU vài giờ).

## 6. Mở rộng (b) của user: list hàng nghìn từ, chống chèn từ
- Scaling: pmaxsim đã chấm kho 209k trong 75 ms/câu (`scorer/results.md` §1) → làm tầng lọc rẻ, phương án C chạy trên 200 từ còn lại.
- Chèn từ / hallucination: đo bằng `missing_N2000` (list không có từ đúng). Thêm metric **số từ trong list bị chèn sai / câu**. Vị trí + cờ "khó" là cơ chế chặn: từ không có đoạn âm khớp thì không được đưa vào.
