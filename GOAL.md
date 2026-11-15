# Mục tiêu paper (chốt với user 08/11/2026)

> File này thay cho các md mô tả hướng cũ. Đọc file này trước tiên.
> Quy ước: **[đo]** = đã chạy thật; **[ý tưởng]** = chưa chạy.

## 1. Bài toán trong một câu
Có audio và một list bias **dài, nhiều nhiễu** (IS21, N100 → N2000). Cần một **scorer** trả về cho mỗi từ **(p, t_đầu, t_cuối)**:
- dùng **p** để lấy đúng vài từ, thay vì nhét top-50;
- dùng **vị trí** để can thiệp **lúc decode** đúng chỗ từ được nói.

Scorer là module cắm ngoài. Xuất phát trên Whisper; cơ chế phải áp được cho mọi SpeechLLM.

**Chốt 11/11:** *một scorer cắm vào mọi SpeechLLM, train một lần, đẩy từ đúng lúc mà không cần biết bên trong model.*
- Scorer dùng encoder riêng.
- Từ ASR chỉ cần prompt, logits + tokenizer, và text.
- Giây decoder đang nghe được suy ra bằng cách căn chữ đã viết với audio.

Gap, quan sát, giả thuyết G1–G6: **`PROPOSAL.md` mục 0**.

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

Ghi chú novelty: paper **so trực tiếp với Hou25 trên IS21**, không claim "đầu tiên dùng vị trí". Các công trình gần cần trích và phân biệt (mới đọc abstract / tóm tắt, trừ chỗ ghi §):
- IBM 2604.12398 (§2.3, đã đọc toàn văn): dự đoán vị trí bias word bằng CTC, chỉ làm loss phụ lúc train, bỏ đi lúc inference; fine-tune Granite 8B, list ≤ 200.
- CTC-WS, Andrusenko et al., Interspeech 2024 (arXiv 2406.07096): spot từ trên log-prob CTC, ghi đè đoạn frame tương ứng trong output greedy. Cần model CTC/Transducer; không dùng cho decoder attention hay SpeechLLM.
- Simul-Whisper (arXiv 2406.10052 §2.1): mỗi bước decode lấy frame đang nghe bằng argmax của tổng attention các alignment head (lọc trung vị 7). Mình mượn đúng cách này để gate; họ dùng cho streaming, không dùng cho biasing.
- TCPGen cho Whisper (arXiv 2306.01942): biasing bằng cây tiền tố, có train thêm module, không có vị trí.

## 3. Ba đóng góp
- **C1. Scorer mới DAS** ra (p, t_đầu, t_cuối) [ý tưởng, user đồng ý hướng 08/11]:
  - ① tìm đoạn từ, không cần list, span được dạy bằng forced alignment của train;
  - ② truy hồi có tranh nhau: softmax trên list cho từng đoạn;
  - ③ căn chỉnh âm vị đơn điệu trong đoạn, thay CNN; điểm khớp + ranh giới;
  - kết hợp ①②③, có lớp "không từ nào trong list".

  Mỗi khối phải dựa trên một gap đo được ở Phần A (`PLAN.md` P1). Khối nào không có gap thì bỏ. Bản dự phòng: định vị → kiểm chứng → cạnh tranh trên S1 của Hou25 (mục 3a).
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

## 3a. Dùng thời gian bên trong scorer (rút từ ablation của Hoàng)

Bằng chứng, repo `itshoang2024/hou25_interspeech`, NOTES F7–F16 [đo]:

| # | Thí nghiệm | Số | Rút ra |
|---|---|---|---|
| 1 | Can thiệp + train lại chỉ với S1 (F10) | bỏ S1 thì mAP 0.08; bỏ cross-attention chỉ mất ≤ 0.016; attention gần như đều | Scorer Hou25 thực chất = **S1 + CNN đọc vệt chéo** |
| 2 | Dò vệt chéo S1 (F14, Table 9) | 0.05 s; 99% nằm trong span; đúng thứ tự 96–98% | Vị trí có sẵn trong S1, nhưng bị **global average pool** vứt đi |
| 3 | S1-only: đọc điểm tại đỉnh CAM | AUC 0.992 so với 0.975 của chính output model | **Pool làm loãng vệt ngắn** trên câu dài |
| 4 | Bản đầy đủ: độ cao đỉnh so với p | đều khoảng 0.99 | Với từ nhiễu ngẫu nhiên, thời gian không thêm "có/không"; **chưa thử với từ nghe giống** |
| 5 | Prompt xếp theo thời gian: cả 50 từ / chỉ từ chắc chắn (F15/F16) | +0.42 / −0.17 | Vị trí chỉ có giá trị khi đi kèm p cao |
| 6 | Phân loại lỗi còn lại (F12) | nghe giống 64–70%; hạng 11–50 sai gấp khoảng 8 lần top-10 | Lỗi nằm ở pha decode → dùng vị trí để gate boost |
| 7 | Đổi nguồn từ nhiễu (F7) | recall không đổi | Scorer nghe audio thật, không học mẹo theo cách IS21 bốc từ nhiễu |

Ghi chú: S1 có vệt chéo dốc khoảng 1 vì **hai phía cùng là audio, cùng encoder, cùng 50 frame/s**. Âm vị không có độ dài, nên đổi sang âm vị là mất vệt có sẵn. Muốn dùng âm vị thì phải tạo lại vệt (kéo dài theo thời lượng dự đoán, hoặc bộ căn chỉnh theo thứ tự).

Hướng C1 [ý tưởng]:
```
audio + từ ─► bản đồ S (kiểu S1) ─► (1) ĐỊNH VỊ: tìm vệt → t_đầu, t_cuối
                                    (2) KIỂM CHỨNG: chấm p trong cửa sổ đó, không lấy trung bình cả câu   (#3)
cả list ───────────────────────────► (3) CẠNH TRANH: các từ trỏ cùng một đoạn → chỉ giữ từ khớp nhất       (#5, nhắm vào từ nghe giống)
```
Cách kể chuyện: Hou25 vứt thời gian ở bước pool; ta chứng minh thời gian có trong S1 và pool làm loãng tín hiệu; vì vậy đưa thời gian thành thành phần chính của scorer.

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
> Kế hoạch chi tiết (subtask, DoD, hạn, người làm): **`PLAN.md`**. Danh sách dưới đây là bản tóm tắt cũ.

1. **Xin `exp/` của Hoàng:** `exp/localize/` (thời gian dự đoán của mọi ứng viên), `exp/scorer/is21/scores`, các checkpoint.
2. Chẩn đoán (chỉ cần CPU): từ đúng lọt khỏi top-10 là do bị **từ nghe giống** vượt lên, hay do **điểm thấp**?
3. Thử **(3) cạnh tranh** (CPU, không train): trùng đoạn thì giữ từ điểm cao hơn, rồi tính lại R#10 và H#50.
4. Thử **(2) kiểm chứng tại đỉnh** trên checkpoint của Hoàng (GPU, ngắn): đọc điểm trong cửa sổ quanh vệt thay cho average pool. Đo cả trên list từ nghe giống (`near_N2000`, `ctx_hard_N2000`).
5. **Thử trần cho C3:** boost có gate trên Whisper với vị trí hoàn hảo (forced alignment), rồi với vị trí lấy từ S1 của Hou25. Điểm quyết định: nếu không thắng prompt top-50 thì xem lại hướng.
   - Code đã có (08/11): `bench/align_words.py` (forced alignment wav2vec2 → `protocol/<split>/align.tsv`) và `bench/gate_boost.py` (greedy Whisper; mỗi bước lấy giây decoder đang nghe từ alignment heads; chế độ prompt / boost không gate / gate span / gate center).
   - Chạy thử 8 câu test-clean, whisper-base, CPU [đo, chỉ kiểm cơ chế, không phải kết quả]: giây decoder đang nghe khi sinh bias word khớp span forced alignment trong khoảng 0.0–0.06 s; boost không gate làm mất chữ "what" ("…arrows ~~what~~ no hamlet"), gate giữ lại được.
   - **Đã chạy 09/11** (`bench/results.md` mục 6), test-other:
     - Oracle + gate MFA: B-WER 7.90.
     - Oracle + gate S1: 8.42.
     - Cả hai thắng prompt Hou25 (9.69).
     - Top-50 + gate S1 hỏng (WER 15.8, câu lặp), vì từ nhiễu cũng được boost.
   - Bước kế: thêm quyết định "từ có trong câu không" trước khi boost (ngưỡng điểm / top-k nhỏ), rồi chạy lại top-50; chạy gate trên test-clean. Cần duyệt trước khi code.
   - A1 (09/11): đổi cách gộp không tăng R#10; vị trí nằm ở S1 (92–96% trong span), không ở CAM.
6. Thiết kế C1 chi tiết, duyệt với user trước khi code.
7. **(11/11) Kế hoạch mới: `PLAN.md` bản 2** (quan sát → giả định → gap → kiến trúc → thí nghiệm → đóng góp; phần R, B, K, D, E, F). Bước kế theo `PROPOSAL.md` mục 0: G1 (vị trí được lệch bao nhiêu), G2 (gate so với prompt và sửa sau, chia lỗi theo nhóm), G6a (vị trí decoder suy ra từ chữ đã viết so với alignment heads). Cả 3 chỉ decode Whisper, không train. Cần xin GPU và duyệt code.

8. **(14/11) Đến buổi gặp thầy 20/11: chỉ làm scorer.** Roadmap: `scorer/PLAN-2011.md` (giả định S1–S5, thí nghiệm E0–E5). Adapter để sau.
   - E0 đã xong (`scorer/analysis/SMALL_R.md`): đo với R nhỏ, Sạch@1 của Hou25 là 72.8, pmaxsim 85.7.
   - Hou25 thua vì **báo động giả với từ khác hẳn**, và báo động giả tăng theo độ dài câu (1.4 → 11.4 từ/câu). Không phải vì từ nghe giống.

9. **(15/11) Kết quả chạy 2× 4090: `scorer/analysis/REPORT-1115.md`.**
   - Hou25 gộp theo vệt chéo + train lại: Sạch@1 73.3 → 82.9, định vị 98.9% (giả thuyết của Hoàng đúng).
   - Scorer âm vị vẫn là nền tốt nhất (87.5).
   - pmono + đẩy có vị trí, không prompt, thắng Hou25 prompt về B-WER (8.14 so với 9.69 trên test-other; 3.79 so với 5.15 trên test-clean).
   - Sáng 15/11: scorer hai nhánh (khớp tự do + khớp đúng thứ tự) có dạy vị trí. Một model 89.3 Sạch@1; cộng điểm hai model 90.8. Vào Whisper không prompt: WER 3.14–3.22, B-WER 7.53–7.65 (Hou25 3.39 / 9.69). Kênh prompt bão hoà.
   - Bước tiếp theo: bảng 1b trong báo cáo (chấm bản train có dạy vị trí; ghép pmaxsim + pmono; tranh nhau tại chỗ + "không từ nào" được train; nhóm từ ASR nghe sai; 960h; đủ N100–2000).

Hạn Interspeech 2027: 09/02/2027.
