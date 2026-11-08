# Đề tài: Dạy máy TỰ ĐOÁN "danh sách gợi ý" để nghe đúng tên riêng
### (bản dễ hiểu — đọc để nói lại với thầy)

---

## 1. Câu chuyện vấn đề

Tưởng tượng **cái loa thông minh**. Bạn nói:

> 🎤 *"phát bài **Jai Ho** của **Hương Ly**"*

Loa nghe **mấy từ thường thì đúng**, nhưng **tên riêng thì sai**:

> 📝 *"phát bài ~~chai hô~~ của ~~hương li~~"* ❌

👉 **Tên riêng (tên bài, ca sĩ, địa danh...) là chỗ máy nghe sai nhiều nhất.** Đây là vấn đề kinh điển của nhận dạng giọng nói.

---

## 2. "Cái phao" — và vì sao ngoài đời không có

Cách người ta hay chữa: **đưa cho máy một "phao"** = danh sách các tên cần chú ý.
*("Này máy, để ý mấy tên này nhé: Jai Ho, Hương Ly, Sơn Tùng...")* → có phao thì máy nghe đúng liền.

**Nhưng ngoài đời không ai đưa phao sẵn.** Loa chỉ nhận được **âm thanh**, chứ không có ai gõ sẵn tên bài hát.

> 👉 **Đề tài = dạy máy TỰ CHẾ RA cái phao đó, chỉ từ âm thanh.**

Thầy gợi ý thêm: cho máy biết trước **lĩnh vực** (ví dụ "đây là nhạc"). Vậy máy biết tìm phao trong **kho nhạc**. Nhưng kho nhạc có **hàng triệu bài** — câu này nói tới bài nào? → **vẫn phải đoán.** Đó là phần khó đề tài giải quyết.

---

## 3. Ý tưởng chính (phần mới) — đoán phao bằng ÂM THANH, không bằng CHỮ

Nhận xét mấu chốt:

> **Những tên cần đoán = đúng những từ máy vừa nghe SAI.**
> Nên **không thể tra bằng chữ** (chữ đã hỏng). Phải tra bằng **ÂM THANH**.

### Chạy tay 1 ví dụ từ đầu đến cuối

**Bước 0 — Chuẩn bị "kho tên" (làm sẵn 1 lần):** với mỗi tên, ghi kèm *cách nó phát âm*.

| Tên trong kho | Cách phát âm |
|---|---|
| Jai Ho | `chai ho` |
| Hương Ly | `hương ly` |
| Sơn Tùng | `sơn tùng` |
| ... (605 tên nhạc) | ... |

**Bước 1 — Nghe lần 1:** Whisper cho ra bản ghi thô → *"phát bài **chai hô** của **hương li**"* (sai tên).

**Bước 2 — Đổi cái vừa nghe sang âm:** "chai hô" → `chai ho`; "hương li" → `hương ly`.

**Bước 3 — So âm với kho** (đo xem 2 cách phát âm khác nhau bao nhiêu, càng giống càng tốt):

| Máy nghe (âm) | Trong kho (âm) | Giống? |
|---|---|---|
| `chai ho` | `chai ho` (Jai Ho) | ✅ khớp → chọn **Jai Ho** |
| `hương ly` | `hương ly` (Hương Ly) | ✅ khớp → chọn **Hương Ly** |
| `chai ho` | `sơn tùng` | ✗ khác xa → bỏ |

→ **Phao tự đoán được = { Jai Ho, Hương Ly }** — chỉ từ âm thanh, không ai đưa.

**Bước 4 — Nghe lần 2** (Whisper + phao vừa đoán):
> 📝 *"phát bài **Jai Ho** của **Hương Ly**"* ✅ đúng hết.

### Vì sao chắc chắn chạy được?
Vì **khi máy sai tên, nó sai CHỮ nhưng đúng gần ÂM** ("chai hô" ≈ "Jai Ho"). Chữ hỏng nhưng **âm vẫn còn** → tra ngược ra tên đúng. Đó là lý do "so âm" work còn "so chữ" thì bó tay.

---

## 4. "Làm sao biết chỗ nào có tên riêng?" (câu hay hỏi)

Thật ra **không cần biết chính xác chỗ nào.** Có 3 cách, từ dễ nhất:

- **Cách 1 (dễ nhất):** *quét cả câu.* Lấy từng tên trong kho, xem cách phát âm của nó có xuất hiện **ở bất kỳ đâu** trong câu vừa nghe không. Ví dụ: âm `chai ho` có nằm trong `phát bài chai ho của hương ly` không? → **CÓ** → nghi là "Jai Ho". (Giống tìm 1 từ trong đoạn văn.) → *khỏi cần biết chỗ.*
- **Cách 2:** *dùng điểm tự tin.* Whisper khi viết mỗi chữ có kèm **xác suất** (nó chắc bao nhiêu). Chữ "phát" = 0.98 (chắc), chữ "chai" = 0.42 (lưỡng lự). Chỗ điểm thấp = chỗ nghi tên riêng → chỉ tra ở đó cho nhanh.
- **Cách 3:** *dùng mẫu câu.* Đã biết là nhạc → biết "sau *phát/nghe* là tên bài, sau *của/bởi* là ca sĩ" → nhắm thẳng 2 chỗ đó.

*(Thử cách nào tốt nhất cũng chính là một phần nội dung luận văn.)*

---

## 5. Vì sao KHÔNG làm hỏng từ thường?

Nếu ai nói **"nhà tranh"** (từ thường): máy nghe **đúng, tự tin cao** → không rơi vào "chỗ nghi ngờ" → **không tra, không thay bậy** thành địa danh "Nha Trang".

Thêm nữa: **"nhà tranh" và "nha trang" khác DẤU** → cách phát âm khác nhau → có tra cũng không khớp.
👉 Đây là chỗ **thanh điệu tiếng Việt** quan trọng — và là **đóng góp riêng** cho tiếng Việt (tiếng Anh không có chuyện dấu).

---

## 6. Sơ đồ toàn bộ

```
   🎤 "phát bài Jai Ho của Hương Ly"
            │
            ▼
   ┌──────────────────────┐
   │ ① NGHE LẦN 1 (Whisper)│  → "phát bài chai hô của hương li"  ❌ sai tên
   └──────────────────────┘
            │
            ▼
   ┌──────────────────────┐   so ÂM cái vừa nghe với KHO TÊN nhạc
   │ ② ĐOÁN PHAO (so âm)   │   "chai hô"≈Jai Ho , "hương li"≈Hương Ly
   └──────────────────────┘   → phao = { Jai Ho, Hương Ly }
            │
            ▼
   ┌──────────────────────┐
   │ ③ NGHE LẦN 2 (+phao)  │  → "phát bài Jai Ho của Hương Ly"   ✅ đúng
   └──────────────────────┘
```
*(Chỉ dùng Whisper có sẵn — không phải huấn luyện mô hình mới.)*

---

## 7. Làm trên dữ liệu nào? (đã kiểm tra thật)

Cần bộ dữ liệu có: **audio + nhãn lĩnh vực + tên riêng** (nhưng không cho sẵn phao).

**Bộ chính: Speech-MASSIVE (có bản tiếng Việt).** Mình đã tự tải về đếm:
- **2.033 câu** để đánh giá (~6.3 giờ), trong đó **1.385 câu có tên riêng** để thử.
- Có sẵn **kho 11.514 câu** để dựng "kho tên".
- **Phát hiện quan trọng:** kho nội bộ chỉ chứa **56%** số tên trong câu test → 44% còn lại là tên đời thật ("Hương Ly", "Jai Ho"...). → **Cần bổ sung kho ngoài** (ví dụ MusicBrainz — kho nhạc miễn phí khổng lồ). *Con số này giải thích vì sao method cần "cao cấp" hơn là chỉ tra kho có sẵn.*

**Bộ phụ (nếu muốn 1 lĩnh vực chuyên sâu):** VietMed — giọng nói y tế tiếng Việt, tên riêng = thuật ngữ y khoa.

---

## 8. Chứng minh method tốt bằng cách nào?

So nhiều cách đoán phao trên cùng bộ test, xem cách nào giúp nghe đúng tên nhất:

| Cách đoán phao | Ý nghĩa |
|---|---|
| Không phao | mức nền (để biết sai bao nhiêu) |
| Phao = đáp án đúng | mức trần lý tưởng (tối đa cải thiện được bao nhiêu) |
| Đoán bằng **chữ** | cách cũ |
| Đoán bằng **âm** (của mình) | cách mới |

**Phép thử then chốt:** tách riêng **những câu mà lần 1 nghe SAI tên**. Trên nhóm này:
> Kỳ vọng: **đoán bằng ÂM thắng đoán bằng CHỮ** (vì chữ đã hỏng, chỉ âm còn dùng được).
> Ra đúng vậy → **đó là bằng chứng đề tài có giá trị.**

**Đo bằng 2 con số:**
- **Lỗi trên tên riêng** → phải **GIẢM** (mục tiêu chính).
- **Lỗi trên từ thường** → phải **GIỮ NGUYÊN** (không được gợi ý bừa làm hỏng).

---

## 9. Vì sao đề tài này mới & đáng làm?

1. **Chưa ai làm cho tiếng Việt** (các nghiên cứu cũ toàn tiếng Anh/Trung).
2. **Giải đúng chỗ khó người khác né:** nghiên cứu gần nhất (Zhang 2026, tiếng Trung) *cố tình bỏ qua* bước "chọn vài tên từ kho lớn" — mình làm đúng bước đó.
3. **Điểm mới kỹ thuật:** đoán bằng **âm** thay vì chữ + xử lý **dấu tiếng Việt**.
4. **Dùng đồ có sẵn**, không cần huấn luyện mô hình khổng lồ → hợp tầm luận văn, chạy được trên máy thường.

---

## 10. Câu hỏi để hỏi thầy

1. Thầy đồng ý trọng tâm là **"đoán phao bằng ÂM THANH"** không?
2. Có nên **bổ sung kho tên từ ngoài** (vì kho nội bộ chỉ phủ 56%) không?
3. Nên làm **tiếng Việt là chính** hay mở thêm vài ngôn ngữ để so sánh?
4. Bộ Speech-MASSIVE tiếng Việt không có tập "test" riêng — dùng tập "validation" (2.033 câu) để đánh giá được không?

---

> ### Một câu để nói với thầy
> *"Em dạy máy TỰ ĐOÁN danh sách tên riêng cần chú ý — chỉ từ audio và lĩnh vực — bằng cách **so ÂM THANH** cái nó vừa nghe với một kho tên. Vì tên riêng chính là chỗ máy nghe sai nên không thể dựa vào chữ, phải dựa vào âm. Em thử trên tiếng Việt (bộ Speech-MASSIVE) và đã kiểm tra dữ liệu là khả thi."*
