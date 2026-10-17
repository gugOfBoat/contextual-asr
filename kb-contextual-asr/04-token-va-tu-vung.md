# Chương 4 — Đơn vị đầu ra và từ vựng

> **Mục tiêu chương:** hiểu vì sao **cách chia token** là một trong những nguyên nhân gốc khiến contextual biasing khó, và vì sao nó đặc biệt hiểm với tiếng Việt + thuật ngữ ngoại lai. Chương ngắn nhưng đây là chỗ nhiều người bỏ qua rồi trả giá bằng những kết quả không giải thích được.

---

## 4.1 Các lựa chọn đơn vị đầu ra

| Đơn vị | Kích thước $|\mathcal{V}|$ | Ưu | Nhược |
|---|---|---|---|
| **Ký tự (char)** | 30–200 | không bao giờ OOV; đơn giản | chuỗi dài, model phải học chính tả từ đầu |
| **Subword (BPE / SentencePiece-unigram)** | 500–50.000 | cân bằng; chuẩn hiện nay | từ hiếm bị **băm vụn** |
| **Từ (word)** | 100k+ | 1 bước = 1 từ | OOV nghiêm trọng, softmax khổng lồ |
| **Âm vị (phoneme)** | 30–60 | khớp với âm học; tách chính tả khỏi phát âm | cần G2P + từ điển; phải có bước phone→chữ |
| **Âm tiết (syllable)** | ~7.000 (tiếng Việt) | khớp cấu trúc tiếng Việt | khó với từ mượn |

**Thực tế hiện nay**: hầu hết dùng subword. Whisper dùng BPE đa ngữ ~50k; các model tiếng Việt chuyên biệt hay dùng char hoặc BPE nhỏ (1k–5k) vì âm tiết tiếng Việt có cấu trúc chặt.

### BPE hoạt động ra sao (đủ để hiểu hệ quả)

Bắt đầu từ ký tự, lặp lại: tìm **cặp ký hiệu liền nhau xuất hiện nhiều nhất** trong corpus → gộp thành ký hiệu mới → thêm vào từ vựng. Dừng khi đủ $K$ ký hiệu.

Hệ quả trực tiếp và quan trọng: **từ càng hiếm trong corpus huấn luyện tokenizer thì càng bị chia thành nhiều mảnh.** Từ phổ biến = 1 token; tên riêng lạ = 5–8 token.

---

## 4.2 Fertility — đo định lượng mức độ "băm vụn"

**Fertility** = số token trung bình trên một từ (hoặc một âm tiết).

Với tokenizer đa ngữ của Whisper, tiếng Việt bị băm nhiều hơn tiếng Anh đáng kể vì dấu thanh làm ký tự Unicode tổ hợp hiếm gặp hơn trong dữ liệu huấn luyện tokenizer ⚠️ *(nên tự đo trên tokenizer bạn dùng — code ở §4.6, đừng tin con số nghe kể)*.

**Vì sao fertility quan trọng đến vậy với biasing** — lập luận định lượng:

Xác suất model sinh đúng cả một cụm $c$ gồm $L$ token là tích:

$$P(c) = \prod_{i=1}^{L} P(c_i \mid c_{<i}, x)$$

Giả sử mỗi bước model đúng với xác suất trung bình $p$. Thì:

| $L$ (số token) | $p = 0.9$ | $p = 0.8$ |
|---|---|---|
| 1 | 0.90 | 0.80 |
| 3 | 0.73 | 0.51 |
| 6 | 0.53 | 0.26 |
| 9 | 0.39 | 0.13 |

Một thuật ngữ như `amoxicillin` (dễ thành 6–8 token) có xác suất đúng **hoàn toàn** thấp hơn hẳn một từ 1 token, *ngay cả khi model nghe rất rõ*. Và trong beam search, chỉ cần **một** bước trong 8 bước bị đẩy ra khỏi beam là mất cả cụm.

> Đây là lập luận nền cho **họ 4 — Dynamic Vocabulary** (báo cáo `01` §1.4): thay vì bắt model đi đúng 8 bước, thêm hẳn **một token duy nhất** `<amoxicillin>` vào từ vựng lúc suy luận. $L$ từ 8 xuống 1. Rất thanh lịch — và giờ bạn hiểu chính xác nó giải quyết vấn đề gì.
>
> Cũng là lập luận nền cho **trie** (họ 3): trie ràng buộc rằng sau `amo` chỉ còn vài nhánh hợp lệ, tức nâng $p$ ở các bước sau lên gần 1.

---

## 4.3 Đặc thù tiếng Việt

**(a) Âm tiết là đơn vị chính tả.** `học sinh` = 2 "từ" khi đếm WER theo quy ước phổ biến. Hệ quả: **WER tiếng Việt không so trực tiếp được với WER tiếng Anh** — cùng một lượng thông tin, tiếng Việt có nhiều "từ" hơn.

**(b) Dấu thanh và Unicode — cạm bẫy im lặng nhất trong toàn bộ dự án.**

Chuỗi `"hoà"` có **hai** cách mã hoá Unicode khác nhau:
- **NFC** (dựng sẵn): `h` + `o` + `à`(U+00E0) → 3 code point
- **NFD** (tổ hợp): `h` + `o` + `a` + `◌̀`(U+0300) → 4 code point

Hai chuỗi này **trông giống hệt nhau trên màn hình** nhưng `!=` trong Python, tokenize khác nhau, và làm B-WER sai lệch mà không báo lỗi. Thêm nữa là quy ước đặt dấu: `hoà` vs `hòa`, `thuý` vs `thúy`.

> **Quy tắc bắt buộc:** ngay ở bước nạp dữ liệu, chuẩn hoá tất cả (reference, hypothesis, **và bias list**) về NFC bằng `unicodedata.normalize("NFC", s)`. Ghi rõ trong luận văn. Đây là loại lỗi khiến toàn bộ số liệu vô giá trị và rất khó phát hiện về sau.

**(c) Từ mượn và tên riêng — nơi chính tả và phát âm ly khai.**

| Viết trong bias list | Người nói thật sự phát âm |
|---|---|
| `paracetamol` | *pa-ra-xê-ta-môn* |
| `vaccine` | *vắc-xin* |
| `insulin` | *in-su-lin* / *anh-su-lin* |
| `Ibuprofen` | *i-bu-pro-phen* |

Model ASR tiếng Việt (tokenizer tiếng Việt, huấn luyện trên âm tiếng Việt) sẽ có xu hướng sinh ra **âm tiết tiếng Việt**, ví dụ `pa ra xê ta môn`. Nếu bias list ghi `paracetamol` liền một chữ thì:
- Trie không khớp (chuỗi token hoàn toàn khác).
- Boosting không bao giờ được kích hoạt.
- Retrieval theo embedding chữ cũng lệch.

⇒ **Biasing thất bại không phải vì thuật toán, mà vì khoá tra cứu sai.** Đây là giả thuyết mạnh nhất giải thích kết quả "Dynamic Vocabulary không cải thiện gì trên ViMedCSS" trong báo cáo `00`, và là một thí nghiệm chẩn đoán rất đáng làm sớm (chương 2 §2.8).

**(d) Chuẩn hoá văn bản.** Số (`15` vs `mười lăm`), viết hoa, dấu câu, gạch nối (`vắc-xin` vs `vắc xin`). Báo cáo `00` §7.4 đã cho thấy khoảng cách O-WER vs N-WER có thể lên tới **hơn 20 điểm tuyệt đối** trên cùng một model. Với bias list thì vấn đề nhân đôi: bạn phải quyết định **dạng chuẩn (canonical form)** của mỗi thực thể và áp cùng quy tắc cho cả reference lẫn hypothesis.

---

## 4.4 Từ tokenizer đến trie — cầu nối sang chương 7

Mọi cơ chế biasing ký hiệu đều làm việc trên **chuỗi token, không phải chuỗi chữ**. Nên bước đầu tiên luôn là:

```
bias list (chữ) ──tokenizer của CHÍNH model──► chuỗi token ──► dựng prefix tree (trie)
```

```
Ví dụ trie cho ["amoxicillin", "amlodipine", "paracetamol"]:

root
 ├─ "am" ─┬─ "ox" ─ "ic" ─ "ill" ─ "in"   ● (kết thúc: amoxicillin)
 │        └─ "lo" ─ "dip" ─ "ine"          ● (kết thúc: amlodipine)
 └─ "para" ─ "ce" ─ "tam" ─ "ol"           ● (kết thúc: paracetamol)
```

Tại mỗi bước decode, trạng thái = một nút trên trie. Từ nút đó, tập token "hợp lệ để đi tiếp" là các cạnh ra. Le et al. (Interspeech 2021) mã hoá trạng thái này thành **vector nhị phân** $h^{bias} = [h^{start}; h^{cont}] \in \{0,1\}^{2D}$ rồi chiếu vào model — chi tiết ở chương 7 §7.4.

> **Cạm bẫy #5:** dựng trie bằng tokenizer **khác** với tokenizer của model. Rất hay xảy ra khi bạn lấy code biasing từ repo A áp lên model của repo B. Kết quả: trie không bao giờ khớp, biasing "chạy" nhưng không có tác dụng, và **không có lỗi nào được ném ra**. Luôn kiểm tra bằng cách in ra token của vài cụm bias và so với token của reference tương ứng.

---

## 4.5 Bảng quyết định: chọn đơn vị nào cho đề tài tiếng Việt

| Nếu bạn... | Nên |
|---|---|
| Fine-tune Whisper/PhoWhisper | Giữ nguyên tokenizer gốc (đổi sẽ hỏng pretrain). Đo fertility để biết mình đang đối mặt với gì |
| Train from scratch tiếng Việt thuần | BPE 1k–5k, hoặc char — khớp cấu trúc âm tiết |
| Cần biasing thuật ngữ ngoại lai | Cân nhắc **thêm nhánh âm vị**: giữ output subword nhưng bias trên biểu diễn phát âm |
| Cần code-switching Việt–Anh | BPE chung hai ngôn ngữ; kiểm tra kỹ fertility của phần tiếng Anh |

---

## 4.6 Code: tự đo trước khi tin

```python
from transformers import AutoTokenizer
import unicodedata

tok = AutoTokenizer.from_pretrained("vinai/PhoWhisper-small")   # đổi thành model bạn dùng

def probe(s):
    s = unicodedata.normalize("NFC", s)                  # LUÔN chuẩn hoá trước
    ids = tok.encode(s, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    n_syl = len(s.split())
    print(f"{s!r:35} -> {len(ids):2d} token | fertility {len(ids)/max(n_syl,1):.2f} | {pieces}")

for s in ["nhà", "học sinh", "bệnh nhân bị viêm phổi cấp",
          "paracetamol", "amoxicillin", "pa ra xê ta môn",
          "vaccine", "vắc-xin", "Nguyễn Thị Hồng Nhung"]:
    probe(s)

# Kiểm tra bẫy NFC/NFD:
a, b = "hoà", unicodedata.normalize("NFD", "hoà")
print(a == b, len(a), len(b), tok.encode(a), tok.encode(b))   # -> False, và token khác nhau
```

Chạy đoạn này **trước** mọi thí nghiệm biasing. Nó trả lời hai câu hỏi quyết định: thuật ngữ của bạn dài bao nhiêu token, và bias list có khớp dạng chuẩn hoá với reference không.

---

## Bài tập tự kiểm tra

1. Chạy `probe()` trên 20 thuật ngữ trong miền của bạn. Tính fertility trung bình. Với $p=0.85$, xác suất sinh đúng trọn cụm trung bình là bao nhiêu?
2. Nếu bạn thêm token `<paracetamol>` vào từ vựng, phải thay đổi những gì trong model? (Gợi ý: ma trận embedding đầu vào **và** ma trận output/softmax — xem chương 7 §7.6.)
3. Cho reference `"tiêm vắc xin"` và bias list `["vaccine"]`. Liệt kê **ba** cách xử lý khác nhau để biasing có cơ hội hoạt động, kèm ưu/nhược từng cách.
4. Vì sao WER tiếng Việt tính theo âm tiết lại "khắt khe" hơn WER tiếng Anh tính theo từ? Nêu một ví dụ cụ thể.

---

*Chương tiếp → [`05-giai-ma-va-lm.md`](05-giai-ma-va-lm.md): giờ ta đã biết model sinh ra phân phối trên token; chương sau nói về **cỗ máy tìm kiếm** biến phân phối đó thành câu — và đó là nơi cơ chế biasing rẻ nhất, phổ biến nhất được cài vào.*
