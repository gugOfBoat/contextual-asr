# Kịch bản NGẮN — gặp thầy 06/11 (~10 phút)

Slide `hop-thay.pdf`: trang 1–15 là phần nói, trang 16–23 là phụ lục (chỉ mở khi thầy hỏi).
Bản đầy đủ, có câu hỏi khó: `kich-ban-chi-tiet.md`.

**Cả buổi chỉ cần nhớ 3 câu:**
1. Em làm **bộ lọc nghe audio**: từ danh sách 2000 từ, chọn ra vài từ thật sự được nói rồi đưa cho máy ASR. Lỗi từ hiếm **giảm gần một nửa**.
2. Bộ lọc **so phiên âm của từ với audio**, không cần đọc từ thành tiếng như Hou25, nên **nhanh gấp 3**. Độ chính xác **gần bằng** Hou25, chưa hơn.
3. Không ai dạy nhưng nó **tự biết từ nằm ở đâu** trong câu. Việc còn lại để chốt là so với cách của PTC (**S0**).

---

**Trang 2 — Tóm tắt.**
"Em trình bày bộ lọc: nó làm gì, vì sao em làm như vậy, kết quả, và còn thiếu gì."

**Trang 3 — Bài toán.**
"Đưa cho máy ASR danh sách 2000 từ thì nó rối: với Qwen, kết quả y như không đưa. Nên em làm bộ lọc đứng trước, chỉ giữ vài từ."

**Trang 4 — Hình kiến trúc.**
"Hàng trên là audio, qua Whisper (không train lại). Hàng dưới là từ, đổi thành phiên âm, ví dụ 'cabinet' thành K AE B AH N AH T. Cột giữa (pxattn) là cách em so hai bên."

**Trang 5 — Phiên âm thay cho đọc thành tiếng.**
"Hou25 phải cho máy đọc mọi từ thành tiếng: tốn 21 GB, từ mới phải đọc thêm. Em dùng phiên âm nên có ngay, nhanh gấp 3. Đổi lại em dễ nhầm từ nghe giống hơn một chút: 21,6 so với 17,3."

**Trang 6 — Sao không đổi audio thành phiên âm luôn (cách của PTC)?**
"PTC đổi audio thành phiên âm rồi so theo luật, có 5 tham số chỉnh tay. Em không ép audio, mà cho model học thẳng 'từ có được nói không', chỉ một ngưỡng. Hiện PTC tìm từ tốt hơn trên giấy, nhưng khác điều kiện, nên em sẽ chạy S0 để so công bằng."

**Trang 7 — Attention là gì.**
"Mỗi âm của từ có 100 điểm chú ý, rải lên các ô audio; ô nào giống mình thì dồn nhiều điểm. Em chỉ dạy model 'có hay không', nhưng muốn đúng thì nó phải rải điểm đúng chỗ."

**Trang 8 — Đọc hình.**
"Hàng là các âm của từ 'bastard', cột là thời gian, màu vàng là nhiều điểm. Cả câu tối, chỉ sáng đúng chỗ từ được nói, và sáng theo **bậc thang**: âm đầu bên trái, âm cuối bên phải."

**Trang 9 — Đối chứng.**
"Cùng model, nếu em so thẳng mà không qua attention thì ra nhiễu, ngang đoán bừa. Vậy bậc thang là thứ model học ra, không phải tự có."

**Trang 10 — Số.**
"Trên khoảng 1.000 từ mỗi tập: đặt đúng chỗ 99%, lệch 0,03 giây. Cách so thẳng chỉ khoảng 10%, bằng đoán bừa."

**Trang 11 — Lọc theo ngưỡng.**
"Mỗi từ có một xác suất riêng, nên câu nào không có từ hiếm thì danh sách rỗng được. Với Qwen, lỗi từ hiếm từ 16,5 xuống 9,7, gần mức lý tưởng 7,7. Danh sách toàn từ sai thì bộ lọc loại hết, máy không bị kéo viết bừa."

**Trang 12 — Kết quả chính.**
"Với Whisper, lỗi giảm khoảng một nửa so với đưa nguyên danh sách. Em nói thẳng: buổi trước em nói thắng Hou25, nhưng chạy cùng điều kiện thì em kém 0,2–0,9."

**Trang 13 — Chốt.**
"Nên em xin chốt: **gần bằng Hou25**, nhưng không cần đọc thành tiếng, nhanh gấp 3, lọc theo ngưỡng được, và biết vị trí."

**Trang 14 — Còn thiếu.**
"Quan trọng nhất là S0: chạy cách của PTC trong cùng điều kiện, thắng thì em đổi. S1 là so từng câu với Hoàng, đang chờ file."

**Trang 15 — Xin thầy góp ý.** Đọc 4 câu, **ghi lại câu trả lời**.

---

## Nếu thầy hỏi mà chưa biết trả lời
Nói thật: *"Phần này em chưa đo / chưa chắc, em sẽ kiểm và báo lại thầy."* Rồi ghi câu hỏi vào sổ. Như vậy tốt hơn đoán.

## Phụ lục để mở khi cần
| Thầy hỏi | Mở trang |
|---|---|
| Từng bộ phận khác Hou25 thế nào | 16 |
| Có phải chỉ chọn hình đẹp không | 17 (thêm 3 từ) |
| Lớp attention thứ 2 thì sao | 18 |
| Âm có nhìn đúng âm không | 19 (ma trận) |
| Đã thử cách nào khác | 20 |
| Cải thiện bộ lọc thêm có giảm lỗi không | 21 |
| Có ý nghĩa thống kê không | 22 |
