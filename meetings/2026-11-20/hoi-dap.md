# Hỏi – đáp chuẩn bị gặp thầy (20/11)

Mọi số: test-other, list IS21 N2000 (mỗi câu = từ hiếm thật + 2000 nhiễu), 5248 lượt từ hiếm. Nguồn: `scorer/analysis/REPORT-1115.md`, `SMALL_R.md`, `MO-XE-RECALL1.md`. Trong ngoặc vuông là khoảng tin cậy 95% (bootstrap theo câu).

## Câu trả lời 30 giây

> Hou25 lọt top-50 gần như đủ (95.4%), nhưng ở hạng 1 chỉ 72.6%: cứ 4 từ đúng thì 1 từ có nhiễu đứng trên. Nhiễu đó không phải từ nghe giống (96% là từ khác hẳn), mà do Hou25 **lấy trung bình cả bảng** nên câu càng dài càng nhiều nhiễu điểm cao (1.4 → 12.4/câu). Em đề xuất scorer **âm vị, hai nhánh (tự do + đúng thứ tự), dạy vị trí lúc train**: Recall@1 89.3, biết từ nằm ở giây nào (lệch 0.02 s), lọc list còn 1.9 từ/câu. Đưa vào Whisper bằng nhắc đúng lúc: B-WER 9.69 → 7.65; WER tổng chưa có ý nghĩa.

## A. Finding: Hou25 đưa nhiễu lên

| Thầy hỏi | Trả lời | Trang |
|---|---|---|
| Hou25 kém ở đâu? | Không phải sót từ đúng, mà **để nhiễu đứng trên**. Top-50 95.4, @10 90.9, @5 87.1, @1 72.6. Rơi 22.8 điểm từ top-50 xuống hạng 1 (âm vị chỉ rơi 10.9). | 8 |
| Recall@1, @5 là gì? | @K = % từ đúng có ít hơn K từ **nhiễu** đứng trên (từ đúng khác không tính, "lọc" như Bordes 2013). Cần @1 vì cuối cùng chỉ đưa ~2 từ/câu cho ASR. | 6 |
| Sao không dùng top-50 như paper? | Có báo (95.4 / 98.4) nhưng ở mốc 50 mọi hệ 95–99%, che mất khác biệt. | 6, 8 |
| Nhiễu nào vượt lên? | Xét từ nhiễu cao nhất mỗi lần trượt (1429 lượt): **96% khác hẳn** (> 2 âm vị), 3% gần âm, 1% đồng âm. Vd `jeered` bị `appropriately` vượt. | 10 |
| Nó có giống gì trong câu không? | 76% nhiễu điểm cao (p ≥ 0.5) **không giống đoạn nào** trong câu (âm vị scorer: 40%). | 10 |
| Vì sao lại thế? | Cuối mạng là trung bình mọi ô. Linear cuối tuyến tính nên điểm = trung bình điểm từng ô; CNN chỉ nhìn 9×9 ô (0.18 s), nên mảnh khớp ở giây 1 và giây 7 cộng như nhau, **mất thứ tự**. Câu càng dài càng nhiều mảnh: nhiễu p ≥ 0.5 tăng 1.4 (câu < 5 s) → 12.4 (≥ 20 s). | 10–12 |
| Có bằng chứng nhân quả không? | Có: cùng lượt train, **chỉ đổi cách gộp** sang vệt chéo: 73.3 → 82.9, +9.5 [+7.9; +11.2]. Gộp "chỗ sáng nhất" chỉ 73.8 → cái giúp là thứ tự. | 13 |
| Vệt chéo là ý mới? | Không claim (gần dò từ khoá bằng audio mẫu / DTW; ý của anh Hoàng). Cái mới: chỉ ra cách gộp là nguồn lỗi, và dùng vị trí cho ASR. | 12 |
| Từ nghe giống thì sao? | List IS21 ít từ nghe giống. Trên list `hard_N2000` (toàn nhiễu nghe giống): Hou25 51.4, âm vị 61.2 (+9.7 [+7.9; +11.4]) → TTS không giúp. | — |

## B. Tái hiện Hou25 có đúng không

| Thầy hỏi | Trả lời | Trang |
|---|---|---|
| Paper dùng 4 feature, mình có đủ không? | Đủ. §2.2 Eqn 1–2: cross-attention 2 chiều → S1…S4 = cos của (speech, bias), (speech, bias→speech), (speech→bias, bias), (speech→bias, bias→speech), xếp 4 kênh → CNN 32/64/128/256 k3 → pool → Linear. Code `models.Hou25` đúng y vậy, 1.95M tham số. Linear 368, 8 head, dropout 0.1, lr 5e-5, nhiễu 40% edit distance + 60% ngẫu nhiên, edge-tts, encoder Whisper-large-v3: theo §3.3. | 11 |
| Chỗ nào tự chọn? | Paper không nói: logit + BCE (thay softmax), encoder đóng băng, pool chỉ trên vùng thật. | — |
| Tái hiện có fail không? | Theo thước đo của paper thì **không**: top-50 → Whisper-turbo, N2000, B-WER paper 5.8 / 11.7 (clean / other); bản 3 epoch của Hoàng 4.9 / 9.2; bản của em 5.15 / 9.69. Paper không báo Recall@1 nên 72.6 không so với paper được. | — |
| So có công bằng không? | **Chưa.** Hou25 của em chỉ 2 epoch (hết giờ GPU), chọn bản theo Recall@10 dev-clean; scorer âm vị 6 epoch, chọn theo Recall@1 dev-other. Chưa chạy: Hou25 6 epoch, và Hou25 chỉ S1 (bỏ cross-attention) để biết 4 feature đóng góp bao nhiêu. | 9 |
| Hoàng có Hou25 mạnh hơn? | Trong pipeline của Hoàng, Hou25 3 epoch → prompt: 9.2 (other), còn pxattn của nhóm (bản cũ) 10.1 → **chưa có ckpt của Hoàng để chấm Recall@1**. Cần so lại trước khi viết paper. | — |

## C. Phương pháp đề xuất

| Thầy hỏi | Trả lời | Trang |
|---|---|---|
| Ý chính? | Bỏ TTS. Từ → chuỗi âm vị (G2P), audio → 1 vector / 20 ms, bảng M = cos(âm vị, khung). Hai nhánh đọc bảng: **tự do** (mỗi âm vị lấy khung sáng nhất, ít sót) + **đúng thứ tự** (DP, âm sau cách âm trước 1–15 khung, loại nhiễu rải rác). logit = a·thứ tự + b·tự do + c → sigmoid. | 16–22 |
| Vị trí ra từ đâu? | Không học riêng: lần ngược đường DP → khung đầu–cuối → × 0.02 s. | 26 |
| Dạy vị trí là gì? | Lúc train, với từ đúng, xoá phần bảng ngoài đoạn MFA ± 0.1 s → muốn điểm cao phải khớp đúng chỗ. Không thêm tham số. | 24 |
| Train thế nào? | 28.539 câu train-clean-100; mỗi từ đúng + 4 nhiễu; BCE; 6 lượt; chọn bản theo Recall@1 500 câu dev-other. Recall@1 không vào loss. | 23 |
| Vì sao nhanh? | Phía từ không phụ thuộc câu → tính sẵn 209k từ; chấm 2000 từ = 1 phép nhân ma trận, 4–7 ms/câu (Hou25 ~1200 ms). | 17 |
| G = 15 khung, ngưỡng 0.5 lấy đâu? | Nhóm tự đặt (loại C): G = 0.3 s giữa hai âm vị; ngưỡng/λ chọn trên 400 câu dev-other. | 20 |

## D. Kết quả

| Thầy hỏi | Trả lời | Trang |
|---|---|---|
| Recall@1/5/10? | Hou25 72.6 / 87.1 / 90.9 · vệt chéo 82.9 / 95.3 / 96.7 · tự do 87.5 / 96.2 / 97.4 · **đề xuất 89.3 / 97.5 / 98.6**. | 8 |
| Mỗi thành phần góp bao nhiêu? | Dạy vị trí (trên tự do) 87.5 → 88.3; thêm nhánh thứ tự chung bảng: @1 đứng yên, nhóm nghe sai +2.9; tách hai bảng +1.0 → 89.3. So tự do: +1.8 [+0.9; +2.6]. | 28 |
| Nhóm khó nhất? | Từ Whisper nghe sai (986 lượt): Hou25 45.9, tự do 64.5, đề xuất **70.9**. Câu ≥ 20 s: nhiễu điểm cao 12.4 → 0.9/câu. | 28, 29 |
| Vị trí đúng tới đâu? | Tâm lệch trung vị 0.02 s so với MFA, 99.1% tâm rơi trong đoạn thật (chỉ đo trên từ đúng). Hou25 gốc 44% (số này lệch với lần train lại 79%, chưa kiểm). | 27 |
| Lọc list còn bao nhiêu? | p ≥ 0.5 + bỏ từ trùng chỗ: 1.92 từ/câu = 1.65 đúng + 0.27 nhiễu, precision 85.9% (câu thật có 1.79 từ hiếm). Hou25: 1.30 đúng + 1.06 nhiễu. | 30 |
| Vào ASR được gì? | Whisper-turbo, nhắc đúng lúc (λ = 5 khi decoder đang nghe đúng giây của từ), không prompt: B-WER 9.69 → **7.65**, −2.04 [−2.76; −1.37]; test-clean 5.15 → 3.83. Thêm prompt: 6.42. | 33 |
| WER tổng thắng không? | **Chưa**: −0.17 [−0.39; +0.12]. Cần vị trí: nhắc mọi lúc WER 3.66, nhắc đúng lúc 3.22 (−0.44 [−0.55; −0.35]). | 33, 34 |
| Audio không có từ nào trong list? | Chưa ổn: WER 4.51 so với 4.25 không context. | — |
| Đã thử Qwen3-ASR chưa? | Chưa với scorer mới; mới Whisper. Bản cũ (pxattn lọc p ≥ 0.5 vào prompt Qwen): B-WER 9.71 other. | — |

## E. Chỗ yếu – nói trước khi bị hỏi

1. Hou25 2 epoch vs âm vị 6 epoch; mỗi hệ train 1 lần; dev 500 câu dao động 2–3 điểm.
2. Bản một model (89.3) còn kém bản cộng hai model riêng (90.8).
3. Nhiễu nghe như từ thường (`flauve` ≈ "fall") vẫn lọt, vì không có từ hiếm nào tranh chỗ.
4. WER tổng chưa có ý nghĩa; list toàn nhiễu làm WER tăng.
5. Mới một ASR (Whisper); hướng nhóm là cắm vào nhiều SpeechLLM.

## Câu nên hỏi thầy

1. Đóng góp chính đặt ở scorer, ở cách nhắc, hay cả hai?
2. Có cần so với PTC-Bias không, hay Hou25 là đủ?
3. Có nên chạy Hou25 6 epoch + bản chỉ S1 trước khi viết?
