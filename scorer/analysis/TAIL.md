# Đuôi khó: từ đúng nào bị scorer bỏ sót? (13/11/2026)

> Gap S2 trong `PROPOSAL.md` §0.6.
>
> - Script: `scorer/analysis/tail.py`, chạy CPU. Bảng đầy đủ: `tail_raw.md`.
> - "Sót" = từ hiếm đúng **không lọt top-50** của list IS21 N2000.
> - Dữ liệu: `prob_N2000.json` của các checkpoint 18/10. Hou25 là bản tái hiện 2 epoch; pmaxsim và pxattn dùng âm vị. Tất cả train trên clean-100, dùng encoder Whisper-large-v3.
> - Đặc điểm từ:
>   - độ dài từ và thời điểm bắt đầu lấy từ `align.tsv` (MFA);
>   - số âm vị lấy từ g2p;
>   - "Whisper / Qwen không prompt đúng" = từ đó có xuất hiện trong hyp khi không có context.
> - Mỗi lượt = một từ hiếm khác nhau trong một câu.

## Tổng

| | test-other (5248 lượt) | test-clean (5692 lượt) |
|---|---|---|
| Hou25 sót | 4.61% (242) | 2.23% (127) |
| pmaxsim sót | 1.56% (82) | 0.49% (28) |
| pxattn sót | 2.29% (120) | — |
| Sót ở mọi hệ | 0.59% (31) | 0.25% (14) |
| **Chỉ Hou25 sót** | **174 / 242** | **113 / 127** |

## Phát hiện

### 1. Hơn 2/3 số sót của Hou25 là lỗi riêng của mô hình, không phải do audio khó
- 174 / 242 từ (test-other) và 113 / 127 từ (test-clean) Hou25 sót, nhưng scorer âm vị vẫn bắt được.
- Vậy audio **có** tín hiệu, chỉ là kiến trúc / biểu diễn của Hou25 bỏ lỡ.

### 2. Audio sau giây thứ 30 bị mất hẳn
- Từ bắt đầu sau giây 30: sót **83–100%** ở mọi hệ (test-other 6 lượt, test-clean 1 lượt).
- Câu dài hơn 30 s: sót 10.8–11.9%.
- Nhóm này chiếm **16% / 14%** của "sót ở mọi hệ".

Lý do: encoder Whisper chỉ nhận 30 s, phần sau bị cắt. Đây là lỗi kỹ thuật **có chung ở mọi scorer dựa trên encoder Whisper (Hou25, CB-Whisper)**. Sửa được bằng cách chia cửa sổ trượt.

Vì @99 do khoảng 1% từ khó nhất quyết định (khoảng 52 lượt trên test-other), sửa chỗ này có thể kéo @99 xuống đáng kể. **Cần đo.**

### 3. Đuôi trùng với các từ mà ASR cũng nghe sai
| Whisper không prompt | sót Hou25 | sót pmaxsim | phần trong "sót mọi hệ" |
|---|---|---|---|
| sai (986 lượt, other) | 14.5% | 7.5% | **94%** |
| đúng (4262 lượt) | 2.3% | 0.2% | 6% |

- Từ khó với scorer cũng chính là từ khó với ASR.
- Đây lại là **nhóm cần scorer nhất**: từ ASR đã nghe đúng thì không cần nhắc.

→ Metric chính nên báo thêm **recall trên nhóm từ ASR nghe sai** (giống "từ khó" của `EVAL_PROTOCOL.md`), không chỉ recall trên mọi từ.

### 4. Từ ngắn và từ có dấu `'` hoặc `-`
| Nhóm (test-other) | Hou25 | pmaxsim | pxattn |
|---|---|---|---|
| Từ < 0.25 s | 11.8% | 6.9% | 8.4% |
| ≤ 3 âm vị | 8.6% | 4.8% | 3.8% |
| Có `'` hoặc `-` (t'other, adn't, e's) | 13.2% | 4.7% | 7.5% |
| Ngoài kho 209k (không có sẵn TTS) | 11.4% | 3.4% | 7.4% |
| Trung bình | 4.6% | 1.6% | 2.3% |

- Các từ có dấu thường là cách viết giọng địa phương (`t'other`, `adn't`, `'e's`). TTS và g2p đọc kiểu chuẩn, nên khác hẳn cách người đọc thật phát âm.
- Gap 2 trong PROPOSAL (một cách phát âm chuẩn duy nhất) có số đo ở đây.

### 5. Câu càng nhiều từ hiếm, Hou25 càng sót nhiều
- 1 từ: 2.3%; 2–4 từ: 4.1%; 5–9 từ: 6.1%; ≥ 10 từ: 10.5%.
- Scorer âm vị tăng ít hơn: 1.0 → 3.6%.
- Câu nhiều từ hiếm thường là câu dài, nên đây là biến nhiễu với độ dài câu. Khớp với giả thuyết "lấy trung bình cả bản đồ làm loãng vệt" (S1).

### 6. Ví dụ đuôi thật (sót ở mọi hệ)
- **Tên riêng đọc lạ, lặp nhiều lần trong cùng chương:**
  - `pileser` sót 4 lần (4198-61336-*);
  - `zarathustra` sót 3 lần;
  - `proas` sót 2 lần.

  Một người đọc, một cách đọc, nên sai một lần là sai cả chương.
- **Giọng địa phương hoặc từ rút gọn:** `t'other`, `adn't`, `e's`, `ve`, `haid`, `souse`.
- **Từ sau giây 30:** `colossal`, `statues`, `salon`, `vecchio`.
- **Từ ngắn, hay bị nuốt:** `lain` (trong "mc lain"), `sine`, `crux`.

## Hệ quả cho thiết kế scorer

| Phát hiện | Việc cần làm | Ước chi phí |
|---|---|---|
| Sau giây 30 bị mất | Encoder chạy theo cửa sổ trượt (30 s, chồng 5 s) cho câu dài. **Sửa trước mọi thứ khác** | Thấp, không cần train lại nếu scorer đọc được cửa sổ ghép |
| 2/3 số sót là lỗi riêng của Hou25 | Đọc cục bộ thay vì lấy trung bình; so với cách biểu diễn âm vị (D1, D4) | Train lại |
| Từ có `'` hoặc `-`, từ ngắn | Biểu diễn từ có nhiều cách đọc (nhiều giọng TTS hoặc nhiều phiên âm), loại bỏ dấu khi tạo cách đọc | Trung bình |
| Đuôi = từ ASR nghe sai | Báo recall trên nhóm "ASR nghe sai" làm metric chính | Không tốn |

## Còn thiếu
- Chưa có @99 sau khi sửa lỗi 30 s: cần điểm đầy đủ 2000 từ, đang nằm trên server.
- Chưa có số trên checkpoint Hou25 của Hoàng (tốt hơn bản 2 epoch).
- Nhãn "giọng địa phương" mới là nhìn bằng mắt. Phải nghe 50 ví dụ (R5) mới chắc.
