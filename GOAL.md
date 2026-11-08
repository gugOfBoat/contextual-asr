# Mục tiêu paper (chốt với user 08/11/2026)

> File này thay cho các md mô tả hướng cũ. Đọc file này trước tiên.
> Quy ước: **[đo]** = đã chạy thật; **[ý tưởng]** = chưa chạy.

## 1. Bài toán trong một câu
Có audio và một list bias **dài, nhiều nhiễu** (IS21, N100 → N2000). Cần một **scorer** trả về cho mỗi từ **(p, t_đầu, t_cuối)**:
- dùng **p** để lấy đúng vài từ, thay vì nhét top-50;
- dùng **vị trí** để can thiệp **lúc decode** đúng chỗ từ được nói.

Scorer là module cắm ngoài. Xuất phát trên Whisper; cơ chế phải áp được cho mọi SpeechLLM.

## 2. Đối thủ chính: Hou25 (Interspeech 2025), IS21 + LibriSpeech
Hệ gồm 2 pha:

| Pha | Hou25 làm | Vấn đề |
|---|---|---|
| 1. Lấy bias từ list | scorer (TTS + attention + CNN) → **top-50 cố định** | câu không có từ hiếm vẫn nhận 50 từ nhiễu; cần 256 từ mới giữ được 99% từ đúng [đo] |
| 2. SpeechLLM decode | nhét 50 từ vào prompt, **không có vị trí** | prompt chỉ cho biết "có những từ gì", không cho biết "ở đâu" |

Tại sao phải có vị trí [đo, repo Hoàng, Whisper]:
- F12: lỗi còn lại chủ yếu là **nghe đúng nhưng viết chính tả quen hơn**, chiếm 3.1 / 5.6 điểm B-WER; prompt không sửa được.
- F13 / F16: đưa thứ tự vào prompt chỉ lời ≤ 0.7 điểm → kênh prompt có trần thấp.
- F14 (Table 9): vệt chéo S1 của Hou25 định vị được từ, sai số 0.05 s, kể cả những từ Whisper còn sai → thông tin vị trí có thật, nhưng Hou25 bỏ đi.
- Boost không có vị trí thì chèn từ bừa: oracle prompt đã chèn 419 bias word trên test-other.

Ghi chú novelty: người khác có thể đã dùng vị trí theo cách khác (ví dụ IBM 2604.12398 chỉ dùng làm loss phụ lúc train). Paper **so trực tiếp với Hou25 trên IS21**, không claim "đầu tiên dùng vị trí".

## 3. Ba đóng góp
- **C1. Scorer mới** ra (p, t_đầu, t_cuối) [ý tưởng].
  - Ứng viên: bản đồ M (từ × frame audio) + bộ đọc "đi theo thứ tự" → p = độ sáng đường đi, vị trí = hai đầu đường đi.
  - Biểu diễn từ (âm vị / TTS / khác): **chưa quyết**, tìm hiểu sau.
  - pxattn cũ không còn là trọng tâm, cùng lắm là một hàng baseline.
- **C2. Đánh giá vị trí kiểu Table 9**, cùng một mốc alignment cho mọi hệ:
  - sai số trung vị;
  - % nằm trong span (±0.2 s);
  - % đúng thứ tự;
  - riêng nhóm từ ASR còn sai.
- **C3. Can thiệp decode có gate theo vị trí, model frozen** [ý tưởng].
  - Whisper trước: dùng cross-attention để biết decoder đang ở frame nào.
  - Sau đó tìm cách tổng quát cho SpeechLLM khác (Qwen3-ASR, …).

## 4. Mục tiêu đo được (test-other, N2000)

| Thứ | Mốc hiện có [đo] | Mục tiêu |
|---|---|---|
| R_B#10 | Hou25 (Hoàng) 93.8 | ≥ 97.4 (bằng #50 của Hou25) |
| Từ nghe giống lọt top-50 (H#50) | Hou25 17.26 | không tệ hơn |
| Sai số vị trí | Hou25 0.05 s | ≤ 0.05 s |
| B-WER, U-WER, số bias word chèn sai | Hou25 top-50 prompt | thấp hơn Hou25, không tăng chèn sai, kể cả với `missing_N` |

## 5. Bảng kết quả chính (khung, chưa có số)

| Hệ | Chọn từ | Can thiệp | B-WER | U-WER | Chèn sai |
|---|---|---|---|---|---|
| nguyên list | — | prompt | | | |
| Hou25 | top-50 | prompt | | | |
| scorer mới | p ≥ θ | prompt | | | |
| scorer mới | p ≥ θ | boost không gate | | | |
| **scorer mới** | p ≥ θ | **boost có gate vị trí** | | | |
| oracle | từ đúng | prompt | | | |

## 6. Việc tiếp theo
1. Chẩn đoán (chỉ cần CPU): từ đúng lọt khỏi top-10 là do bị **từ nghe giống** vượt lên, hay do **điểm thấp**?
2. Thiết kế C1, duyệt với user trước khi code.
3. Thử boost có gate trên Whisper với vị trí lấy từ S1 của Hou25 (đã có sẵn) → biết sớm C3 có lời không, chưa cần scorer mới.

Hạn Interspeech 2027: 09/02/2027.
