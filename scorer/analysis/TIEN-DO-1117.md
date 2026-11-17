# Bàn giao 17/11/2026: scorer biết từ nằm ở đâu — đã làm gì, ra gì, file ở đâu

> Viết cho **Hoàng** (và ai không theo dõi các lượt chạy 13–17/11). Đọc mục 0 là đủ nắm; mục 1–4 để tra.
> - Số liệu gốc: `scorer/analysis/REPORT-1115.md` (bảng đầy đủ + khoảng tin cậy) và các file `.md` cùng thư mục ghi ở từng dòng. File này **không có số mới**, chỉ gom lại và sửa chỗ cũ bị lệch.
> - Phạm vi: LibriSpeech test-other (2939 câu, 5248 lượt từ hiếm), test-clean (2620); bias list IS21 N2000 nếu không ghi khác. Scorer train trên train-clean-100. ASR = Whisper-large-v3-turbo đóng băng, greedy. **Chưa có Qwen3-ASR.**
> - Quy ước: **[đo]** = chạy thật; **[diễn giải]** = nhóm suy luận, chưa kiểm riêng.

## 0. Tóm tắt một trang

**Câu hỏi:** Hou25 chấm từng từ rồi nhét top-50 vào prompt, làm mất "từ nằm ở giây nào". Giữ lại vị trí thì có lợi gì?

| # | Câu hỏi | Trả lời [đo] | Độ chắc |
|---|---|---|---|
| 1 | Hou25 yếu cỡ nào khi đo ở hạng 1? | Recall@1 (lọc) **72.6**; ở top-50 mọi hệ đều 95–98% nên không thấy | chắc |
| 2 | Hou25 sai vì đâu? | Từ nhiễu nổi lên, tăng theo độ dài câu (1.4 → 12.4 từ nhiễu/câu ≥ 20 s), do **lấy trung bình cả bản đồ** | chắc về hướng |
| 3 | Giả thuyết vệt chéo của Hoàng? | **Đúng khi train lại**: 73.3 → 82.9 (+9.5 [+7.9; +11.2]), định vị 98.9% | chắc |
| 4 | Nền scorer nên là gì? | **Âm vị** (87.5). Hou25 vệt chéo vẫn kém 4.6 điểm, chậm ~200 lần | chắc |
| 5 | Khớp tự do / khớp đúng thứ tự bù nhau? | Có: cộng điểm +2.6, hai model cùng kiểu chỉ +1.1 | chắc về hướng |
| 6 | Scorer đề xuất (1 model, 2 nhánh, dạy vị trí) | **89.3** (+1.8 [+0.9; +2.6] so với tự do); test-clean 94.7; nhóm từ Whisper nghe sai 70.9 | chắc |
| 7 | Vị trí đúng tới đâu? | Tâm lệch **0.02 s** so với MFA, **99.1%** tâm trong đoạn thật ± 0.1 s. Hou25 gốc: 0.69 s, 44% | chắc (chỉ đo trên từ đúng) |
| 8 | Lọc list được không? | Cùng ngưỡng p ≥ 0.5 + tranh nhau tại chỗ: precision **55.2% → 85.9%**, giữ từ đúng 73.0% → 92.4%, còn ~1.9 từ/câu | chắc |
| 9 | Vào Whisper, không prompt, thắng Hou25? | **B-WER** test-other 9.69 → **7.65** (−2.04 [−2.76; −1.37]); test-clean 5.15 → 3.83. **WER tổng chưa**: −0.17 [−0.39; +0.12] | B-WER chắc; WER chưa |
| 10 | Whisper có cần vị trí? | Cần: nhắc đúng lúc 3.22 so với nhắc mọi lúc 3.66, −0.44 [−0.55; −0.35] | chắc |

**Sửa so với tài liệu cũ** (`TIEN-DO-1115.md`, slide `hop-thay.pdf`): precision "3–4% → 91%" đã bỏ (3–4% là top-50 không cắt, 91% là bản cộng 2 model) → dùng **55% → 86% cùng ngưỡng**. B-WER 7.53 / WER 3.14 là bản cộng 2 model → số chính giờ là **bản 1 model: 7.65 / 3.22**. Mọi số "đề xuất" ở đây là **cùng một model** (`pdual2_dl`).

**Còn lệch có lợi cho nhóm** (phải sửa trước khi viết paper):
1. Hou25 train 2 epoch, chọn checkpoint trên dev-clean; scorer âm vị 6 epoch, chọn trên dev-other.
2. `pdual2_dl` là bản tốt nhất trong 4 bản hai nhánh **xếp theo test** (88.3–89.3).
3. So Whisper là so **cả hệ** (prompt top-50 với nhắc đúng lúc). Thiếu dòng "Hou25 vệt chéo + nhắc đúng lúc" để tách phần của scorer.
4. Nhóm câu ≥ 20 s chỉ có 40 câu.

**Chỗ yếu lớn nhất:** audio không có từ nào trong list (list toàn nhiễu) vẫn làm WER tệ đi: 4.49 so với 4.25 khi không dùng gì.

## 1. Scorer đề xuất là gì (`scorer/models.py`, `DualSim(two=True)`)

```
audio ─ encoder Whisper-large-v3 (đóng băng, 1280-d, 20 ms/khung) ─ Linear 256 + 1 lớp Transformer ─┐
                                                                                                    ├─ bảng cos (âm vị × khung)
từ ─ G2P ─ âm vị ─ nhúng + vị trí trong từ ─ 2 lớp Transformer (256), tính sẵn cho kho 209k ──────┘
   nhánh tự do  (MaxSim):  mỗi âm vị lấy khung giống nhất → trung bình                 → ít sót
   nhánh thứ tự (MonoSim): quy hoạch động, âm vị sau cách âm vị trước 1–15 khung (≤ 0.3 s) → ít nhận nhầm, ra [đầu, cuối]
   điểm = a·thứ_tự + a2·tự_do + c ;  p = sigmoid(điểm) ;  vị trí = lần ngược đường đi của nhánh thứ tự
```

- Mỗi nhánh có bộ chiếu riêng (2 bảng). Dùng chung một bảng chỉ 88.3.
- **Dạy vị trí** lúc train: từ đúng chỉ được khớp trong span MFA ± 0.1 s của nó.
- 5.4 M tham số, chấm 4–7 ms/câu. Train 6 epoch train-clean-100 (feature encoder cache sẵn), chọn ckpt theo Recall@1 dev-other N2000.
- Ví dụ số dễ hiểu (bảng đồ chơi 4 âm vị × 8 khung): từ nhiễu có ô sáng rời rạc được nhánh tự do chấm 0.75 (từ đúng 0.85), nhánh thứ tự chỉ 0.20. Xem slide 13 của `meetings/2026-11-20/bao-cao-chi-tiet.pdf`.

## 2. Bảng số chính (test-other, N2000)

### 2.1 Chọn từ

| Hệ | Recall@1 | List nghe giống | Từ Whisper nghe sai | Nhiễu/câu ≥ 20 s | Sai số tâm | % tâm trúng | Nguồn |
|---|---|---|---|---|---|---|---|
| Hou25 (nhóm tái hiện, 2 epoch) | 72.6 | 51.4 | 45.9 | 12.4 | 0.69 s | 44.0 | REPORT §3 |
| Hou25 vệt chéo (train lại) | 82.9 | – | 60.9 | 1.9 | 0.05 s | 98.9 | REPORT §5 |
| khớp tự do (`pmaxsim_c4`) | 87.5 | 64.7 | 64.5 | 1.3 | 0.05 s | 99.0 | `e_loc_other.md` |
| khớp đúng thứ tự (`pmono_c4`) | 85.8 | 60.5 | 64.4 | 0.9 | 0.02 s | 98.9 | `e_loc_other.md` |
| **đề xuất (`pdual2_dl`)** | **89.3** | **67.7** | **70.9** | **0.9** | **0.02 s** | **99.1** | `e_dual_other.md` |
| cộng điểm 2 model riêng | 90.8 | 69.6 | – | – | 0.02 s | – | REPORT §7c |

- Recall@1 (lọc) = filtered Hits@1 (Bordes et al. 2013, §4.2): % lượt từ đúng không có từ **nhiễu** nào đứng trên.
- Định vị đo so với căn chỉnh MFA, chỉ trên từ đúng. Lệch đầu từ 10–90%: tự do [−0.12; +0.08] s, thứ tự [+0.03; +0.10] s (không bao giờ mở sớm). Bản đề xuất chưa tách lệch đầu/cuối.

### 2.2 Lọc list (`nms_fair_other.md`, tính lại 17/11)

Top-50 → hai từ chồng nhau > 30% thì giữ từ điểm cao hơn → giữ p ≥ τ.

| Hệ | τ | Giữ từ đúng (không tranh → tranh) | Precision (không tranh → tranh) |
|---|---|---|---|
| Hou25 | 0.5 | 87.0 → 73.0 | 37.4 → 55.2 |
| Hou25 vệt chéo | 0.5 | 90.7 → 88.3 | 63.4 → 72.9 |
| **đề xuất** | 0.5 | 93.3 → **92.4** | 80.8 → **85.9** |
| đề xuất | 0.3 (dùng cho Whisper) | 96.3 → 95.0 | 58.4 → 69.9 |

Tranh nhau làm Hou25 **mất** 14 điểm từ đúng, vì vị trí của nó sai nên loại nhầm từ đúng [diễn giải].

### 2.3 Whisper (WER / B-WER, script IS21 chính thức, chuẩn hoá Whisper)

| | test-other | test-clean |
|---|---|---|
| không list | 4.25 / 17.92 | 2.16 / 8.64 |
| Hou25: prompt top-50 | 3.39 / 9.69 | 1.66 / 5.15 |
| **đề xuất + nhắc đúng lúc, không prompt** | **3.22 / 7.65** | **1.58 / 3.83** |
| trần (từ + vị trí thật) | 2.99 / 5.99 | 1.53 / 2.77 |

- Nhắc đúng lúc: alignment heads cho biết Whisper đang nghe giây nào (như Simul-Whisper, arXiv 2406.10052 §2.1); từ nào scorer đặt ở giây đó thì cộng λ = 5 vào token đầu; mỗi từ nhắc tới khi đã viết. Ngưỡng p ≥ 0.3, λ chọn trên dev-other 400 câu.
- Theo cỡ list (cùng cách): N100 B-WER 7.14, N500 7.37, …, N2000 7.65. Chưa có Hou25 ở N100–1000.
- Scorer tốt hơn 17 điểm mà đưa top-50 vào **prompt** vẫn chỉ 10.03 (Hou25 9.69): kênh prompt bão hoà.

## 3. File nằm đâu

| Việc | File |
|---|---|
| Báo cáo số đầy đủ 15/11 | `scorer/analysis/REPORT-1115.md` |
| Giải thích dài từng thí nghiệm (số precision/B-WER cũ, xem mục 0) | `scorer/analysis/TIEN-DO-1115.md` |
| Bảng hai nhánh, định vị, Đúng@chỗ | `scorer/analysis/e_dual_other.md`, `e_loc_other.md`, `e_c4_other.md` |
| Lọc list cùng ngưỡng | `scorer/analysis/nms_fair_other.md` (script `nms_at_place.py`) |
| Cộng điểm hai model | `scorer/analysis/ensemble_other.md` (`ensemble_cpu.py`) |
| Đo với R nhỏ, đuôi lỗi | `SMALL_R.md`, `TAIL.md` |
| Code scorer | `scorer/models.py` (MaxSim, MonoSim, DualSim, Hou25), `scorer/training/`, `scorer/inference/` |
| Kế hoạch trước khi chạy | `scorer/PLAN-2011.md` |
| Slide gặp thầy 20/11 + lời nói | `meetings/2026-11-20/bao-cao-chi-tiet.pdf` (32 trang), `kich-ban-chi-tiet.md` |
| Qwen3-ASR: kiến trúc, chỗ cắm scorer | `related-work/qwen3asr-kien-truc-va-cong-bias.md`, `phan-tich-4-paper-va-qwen3asr.md`, `scorer-x-asr-3-vi-tri-can-thiep.md` |

Checkpoint, feature cache, `out/rank_*.json` nằm trên máy 4090 ở `~/model_tmp/` (không đưa lên git vì nặng).

## 4. Việc tiếp theo (thứ tự ưu tiên)

| # | Việc | Vì sao | Xong khi |
|---|---|---|---|
| 1 | Dạy scorer nói "chỗ này không có từ nào" | list toàn nhiễu làm hại WER; nhiễu nghe như từ thường (`flauve` ≈ "fall") lọt | list toàn nhiễu không làm WER tệ đi |
| 2 | Nâng nhóm từ Whisper nghe sai | chỗ cần scorer nhất, đang 70.9 | +3 điểm |
| 3 | Train 960 h; 1 model đuổi kịp bản cộng điểm | 89.3 so với 90.8 | Recall@1 ≥ 90.5 |
| 4 | **So công bằng**: Hou25 cùng số epoch, chọn trên dev-other; chọn biến thể trên dev; thêm "Hou25 vệt chéo + nhắc đúng lúc"; đủ N100–2000 | 4 chỗ lệch ở mục 0 | không còn lệch |
| 5 | Qwen3-ASR | mới có Whisper | có một dòng Qwen3-ASR |
| 6 | Khảo sát cách chỉnh logit khác (ý thầy) | cần baseline phía ASR | chọn 1–2 baseline |

**Cần từ Hoàng:** checkpoint Hou25 của Hoàng (để thay bản tái hiện 2 epoch ở việc 4), và ý kiến về chia việc 1–6.
