# Giải thích nhanh cho người mới (đi kèm slide trang 9, 18, 23, 24, 26)

Bài toán: một câu nói, kèm một danh sách 2000 từ hiếm. Chỉ 1–2 từ trong đó thật sự được nói. Máy phải chấm điểm từng từ để từ thật đứng đầu.

**Từ điển nhỏ**
- **Từ hiếm:** từ không nằm trong 5000 từ tiếng Anh thông dụng nhất (vd tên riêng `bartley`).
- **Scorer:** bộ chấm điểm. Vào: audio câu nói + một từ. Ra: một số 0–1, càng gần 1 càng tin từ đó được nói.
- **Hou25:** scorer của một bài Interspeech 2025 (Hou và cộng sự), là mốc để so. Paper không có code, đây là bản nhóm tái hiện; paper dùng điểm này để chọn top-50 từ đưa vào prompt. **Đề xuất** = scorer của nhóm mình.
- **TTS:** máy đọc chữ thành tiếng (text-to-speech).
- **Âm:** đơn vị phát âm. `bartley` đọc là 6 âm B · AA · R · T · L · IY (B như "b", AA như "a" dài, IY như "i"). Lấy từ từ điển phát âm.
- **MFA:** công cụ có sẵn, cho câu chữ + audio thì chỉ ra mỗi từ bắt đầu, kết thúc ở giây nào. Dùng làm đáp án vị trí.
- **Tập test:** LibriSpeech test-other (2939 câu sách đọc, khó nghe hơn); mọi số trong file đều trên tập này. **Tập riêng để chọn bản:** 500 câu khác (dev), không trùng test.
- Số **thật** (đo trên test): 72.6 / 82.9 / 87.5 / 89.3 / 88.3, 0.02 s, 99.1%. Số **giả để minh hoạ** (bịa ra cho dễ thấy): điểm 0.87 / 0.79 / 0.48; xác suất 90% / 40% / … và phạt tính từ chúng; ví dụ vị trí cột 6–16, 0.12–0.34 s, MFA 0.10–0.36 s.

## 1. Con số Recall@1 nghĩa là gì

Câu: "... met **bartley** at the gate ...". Máy chấm 2000 từ rồi xếp hạng.
- Nếu `bartley` đứng trên mọi từ sai → **trúng**.
- Nếu có một từ sai (vd `bentley`) đứng trên → **trượt**.

Làm vậy cho cả 5248 lần từ hiếm xuất hiện trong tập test. Recall@1 = tỉ lệ trúng. 89.3 nghĩa là trúng khoảng 4687 / 5248 lần.

## 2. Bốn hệ trong bảng (trang 9): mỗi hệ hơn hệ trước ở đâu

| Hệ | Cách chấm 1 từ | Recall@1 |
|---|---|---|
| Hou25 | Cho máy đọc to từ đó, so đoạn audio máy đọc với câu, rồi **lấy trung bình** độ giống trên cả câu | 72.6 |
| Hou25 vệt chéo | Như trên, nhưng thay trung bình bằng: chỉ lấy trung bình dọc **vệt chéo sáng nhất** (= đoạn liền trong câu khớp với từ, xem dưới) | 82.9 |
| Khớp tự do | Không cần máy đọc. Đổi từ thành **âm**: bartley → B AA R T L IY. Mỗi âm tìm chỗ giống nhất trong câu, lấy trung bình | 87.5 |
| Đề xuất | Như khớp tự do, cộng thêm điểm **"các âm phải nối nhau đúng thứ tự"**, và lúc train **dạy đúng chỗ** | 89.3 |

"Vệt chéo" là gì: vẽ bảng, hàng = từng 20 ms của audio máy đọc, cột = từng 20 ms của câu. Nếu từ thật nằm trong câu thì đoạn đầu khớp đoạn đầu, đoạn sau khớp đoạn sau → các ô sáng xếp thành một đường chéo. Vệt chéo = trung bình độ giống dọc đường chéo sáng nhất, thay vì trung bình cả bảng.

Vì sao trung bình hỏng: câu dài 20 giây, từ thật chỉ chiếm 0.3 giây. Lấy trung bình cả câu thì phần đúng bị pha loãng, từ sai có vài chỗ hơi giống rải rác lại được cộng dồn và vượt lên.

## 3. Train (trang 23)

Mỗi từ thật trong câu train sinh ra 5 câu hỏi "từ này có được nói không?" (1 đúng + 4 sai). Câu có 2 từ thật → 10 câu hỏi. Ví dụ câu chỉ có bartley:

| Từ | Đáp án | Máy đoán | Bị phạt |
|---|---|---|---|
| bartley | có | 90% có | ít (0.11) |
| bentley (viết giống) | không | 40% có | nhiều nhất (0.51) |
| bartlett (viết giống) | không | 30% có | 0.36 |
| kirkland (bốc bừa) | không | 5% có | 0.05 |
| ghetto (bốc bừa) | không | 2% có | 0.02 |

- **Từ viết giống:** một trong 20 từ gần chính tả nhất với từ thật (ít phép thêm/xoá/đổi chữ cái nhất). **Bốc bừa:** lấy ngẫu nhiên từ kho 209.000 từ hiếm (kho lấy từ dữ liệu train). Mỗi từ thật đi kèm 4 từ sai; mỗi từ sai bốc thăm: 40% khả năng là viết giống, 60% là bốc bừa. Bảng trên là một lần bốc ra 2 + 2.
- **Phạt** = −ln(xác suất máy gán cho đáp án đúng), ln = log tự nhiên. Vd bentley đáp án "không", máy cho "không" 60% → phạt −ln 0.60 = 0.51. Đoán đúng hẳn (100%) → phạt 0. Phạt của câu = trung bình mọi cặp (ở đây 5 cặp) = 0.21.
- **Xác suất ra từ đâu:** máy tính một điểm thô (xem mục 4) rồi đưa qua hàm sigmoid, hàm ép mọi số về khoảng 0–1.
- Dữ liệu: 28.539 câu train-clean-100 của LibriSpeech; 500 câu dev lấy từ dev-other. Từ thật = từ hiếm có trong câu (tối đa 8 từ mỗi câu).

Máy chỉnh trọng số của hai phần nhỏ học được: phần đổi chuỗi âm thành vector, và một lớp đọc lại audio đặt sau bộ mã hoá Whisper (bộ mã hoá Whisper giữ nguyên, không train; scorer chỉ mượn nó để đọc audio, độc lập với mô hình nhận dạng dùng sau), cùng a, b, c ở mục 4, tất cả chỉnh cùng lúc để phạt trung bình nhỏ đi. Lặp qua 28.539 câu là một lượt, chạy 6 lượt. Sau mỗi lượt đo Recall@1 trên 500 câu dev, giữ bản tốt nhất.

## 4. Âm và vị trí (trang 18, 26)

*Ví dụ bartley/ghetto dưới đây dùng số minh hoạ; số đo thật chỉ có ở đoạn "Đo trên toàn bộ test".*

Vẽ một bảng: **mỗi hàng là một âm** (B, AA, R, T, L, IY), **mỗi cột là 20 ms audio**. Ô càng đậm = âm đó càng giống đoạn audio đó.

- **Khớp tự do:** mỗi hàng chọn ô đậm nhất, ở đâu cũng được (hai âm chọn trùng cột cũng được). Từ sai `ghetto` có 4 ô đậm nằm lung tung vẫn được 0.79, gần bằng từ thật 0.87 → dễ bị lừa.
- **Khớp đúng thứ tự:** phải đi một đường từ âm đầu tới âm cuối, đi dần sang phải (âm sau nói sau âm trước). `bartley` đi chéo đẹp, các ô đậm nhất vốn đã nằm đúng thứ tự, nên cách này cũng ra 0.87 như khớp tự do. `ghetto` không nối được thành đường đẹp, đường tốt nhất chỉ 0.48 → rớt xuống dưới `bartley` (không có ngưỡng, chỉ là xếp hạng).

Độ sáng một ô: máy đổi mỗi âm thành một dãy số (vector), mỗi 20 ms audio cũng thành một dãy số; độ sáng = cosine của hai dãy, 1 = cùng hướng (rất giống), 0 = không liên quan. Điểm một cách khớp = trung bình độ sáng các ô được chọn (6 âm → 6 ô). Đề xuất: điểm thô = a × (điểm đúng thứ tự) + b × (điểm tự do) + c, với a, b, c học được, rồi qua sigmoid ra xác suất.

Cách đi đúng thứ tự, nói chính xác: âm thứ 2 phải ở cột sau âm thứ 1 từ 1 tới 15 cột (không được cùng cột); âm thứ 3 sau âm thứ 2 cũng vậy; … Máy không thử từng đường một mà dùng quy hoạch động: đi từ âm đầu, mỗi ô ghi lại "đường tốt nhất tới được đây", nên chỉ cần quét bảng một lần là ra đường có tổng sáng lớn nhất. Trên hình, đường bartley đi qua cột 6, 8, 10, 12, 14, 16.

**Vị trí** = chính con đường đó (lúc test không có MFA, vị trí chỉ lấy từ con đường). Cột đánh số từ 0, cột t phủ đoạn t×0.02 → (t+1)×0.02 s. Đường của `bartley` đi từ cột 6 (bắt đầu 0.12 s) tới cột 16 (kết thúc 0.34 s) → từ nằm ở **0.12–0.34 s**. MFA nói 0.10–0.36 s. **Tâm** = điểm giữa đoạn: cả hai đều 0.23 s → trúng.

Đo trên toàn bộ test: một nửa số từ có tâm lệch không quá 0.02 s (trung vị), và 99.1% tâm rơi trong đoạn MFA.

Vị trí dùng để làm gì: (1) bỏ bớt từ trong list nếu "chỗ" của nó trùng chỗ một từ khác điểm cao hơn (slide trang 30); (2) nhắc mô hình nhận dạng đúng đoạn audio có từ đó (slide trang 34).

## 5. Dạy vị trí (trang 24)

Lúc train, với từ thật, xoá hết phần bảng nằm ngoài chỗ nó được nói (biết nhờ công cụ căn chữ MFA). Máy không còn "ăn gian" bằng cách mượn âm t của từ bên cạnh; muốn điểm cao phải khớp đúng chỗ. Đo riêng trên khớp tự do (chưa có thứ tự): Recall@1 87.5 → 88.3. Đề xuất 89.3 = tự do + thứ tự + dạy vị trí; chưa có số đo riêng cho "chỉ thêm thứ tự".

---
Hình ở trang 18, 24, 26 dùng số giả để minh hoạ. Số thật lấy từ log train (`scorer/runs/2026-11-14/logs/`) và các file kết quả trong `scorer/analysis/`.
