# Scorer có vị trí: kết quả chạy 14–15/11/2026

> Chạy trên 2× RTX 4090 (GPU 0, 1), container `qwen-3.5`.
> - Dữ liệu: LibriSpeech test-other (2939 câu), test-clean (2620), dev-other (chọn tham số).
> - List: IS21 N2000, gồm bản toàn nhiễu (`missing_N2000`) và bản nhiều từ nghe giống (`hard_N2000`).
> - Mọi scorer train trên **train-clean-100**, trừ dòng ghi "960h"; encoder Whisper-large-v3 đóng băng.
> - ASR: Whisper-large-v3-turbo đóng băng, greedy.
> - Script: `scorer/runs/2026-11-14/run_e.sh`; đo bằng `scorer/analysis/eval_e.py`, `nms_at_place.py`, `bench/e1_table.py`.
> - Quy ước: **[đo]** = đã chạy thật; **[diễn giải]** = suy luận của nhóm.

## 1. Kết luận chính

1. **Vị trí là thứ Hou25 bỏ phí, và giả thuyết của Hoàng đúng.** Chỉ đổi bước gộp của Hou25 từ "trung bình cả bản đồ" sang "đọc theo vệt chéo" rồi train lại: Sạch@1 từ 73.3 lên 82.9 (+9.5 [+7.9; +11.2]), báo nhầm ở câu dài giảm 3 lần, và định vị được 98.9% (mục 5).
2. **Scorer âm vị vẫn là nền tốt nhất:** Sạch@1 87.5, định vị 99%, chấm 4–7 ms/câu, nhanh hơn Hou25 khoảng 200 lần, không cần TTS (mục 3, 6).
2b. **Khớp tự do và khớp đúng thứ tự bù nhau:** cộng điểm hai scorer cho Sạch@1 90.1 (+2.6 [+1.9; +3.3]), list nghe giống +4.0; lợi vượt mức gộp hai model cùng kiểu (+1.1) (mục 6b). Từ đây có kiến trúc đề xuất: scorer hai nhánh.
3. **Có vị trí thì lọc được list:** cho các từ tranh nhau tại cùng chỗ, từ 2000 từ còn khoảng 2 từ/câu với precision 80–90%, giữ 90–96% từ đúng. Top-50 của Hou25 có precision 3–4% (mục 4).
4. **Thắng Hou25 khi đưa vào Whisper mà không cần prompt:** B-WER 8.14 so với 9.69 trên test-other (Δ −1.56, có ý nghĩa); 3.79 so với 5.15 trên test-clean. WER tổng ngang hoặc hơn nhẹ (3.24 so với 3.39; chưa có ý nghĩa thống kê). Thêm prompt thì B-WER còn 6.81 / 3.22 và WER thắng có ý nghĩa, nhưng lỗi ở từ thường tăng nhẹ (mục 7).
5. **Whisper cần vị trí:** cùng một list, đẩy đúng lúc tốt hơn đẩy mọi lúc 0.49 WER, có ý nghĩa (mục 7).
6. **Còn cách trần 2 điểm B-WER** (8.14 so với 5.99). Một nửa do vị trí, một nửa do list. Scorer cũng chưa "dám nói không" khi audio không có từ nào (mục 7).

7. **(cập nhật sáng 15/11) Scorer đề xuất:**
   - *Kiến trúc:* hai nhánh đọc trên bản đồ âm vị × thời gian (khớp tự do + khớp đúng thứ tự), dạy vị trí lúc train.
   - *Một model:* Sạch@1 89.3 (pmaxsim 87.5, Hou25 73); nhóm từ Whisper nghe sai 70.9 (64.5 / 46); định vị 0.02 s (mục 7c).
   - *Cộng điểm hai model train riêng:* Sạch@1 90.8, precision 91% khi giữ 91% từ đúng (mục 6c).
   - *Đưa vào Whisper, không prompt:* WER 3.14–3.22, B-WER 7.53–7.65 trên test-other (Hou25: 3.39 / 9.69); B-WER giảm 2.0–2.2 điểm, có ý nghĩa; WER tổng giảm 0.17–0.25, chưa có ý nghĩa thống kê (mục 7b, 7c).
8. **Kênh prompt đã bão hoà:** scorer tốt hơn 17 điểm mà đưa top-50 vào prompt vẫn chỉ ngang Hou25 (B-WER 9.5–10.0 so với 9.69). Lợi ích của scorer tốt chỉ lấy được qua vị trí + đẩy logit (mục 7b).

## 1b. Bước tiếp theo cho scorer (đề xuất)

| # | Việc | Dựa trên số nào | Kỳ vọng |
|---|---|---|---|
| 1 | ✅ Đã làm: dạy vị trí lúc train giúp (pmono 85.8 → 88.3) | mục 6c | – |
| 2 | ✅ Đã làm bản đầu: scorer hai nhánh một model đạt 89.3. **Còn lại:** thu hẹp khoảng cách với bản cộng điểm (90.8) bằng loss riêng cho từng nhánh hoặc lấy trung bình ckpt; ổn định việc chọn ckpt | mục 7c | Một model ≥ 90.5 |
| 3 | **Tranh nhau tại chỗ + ô "không từ nào" được train**, thay cho luật IoU bằng tay | mục 4; list toàn nhiễu ở mục 7 | Giảm chèn sai trên list toàn nhiễu; tăng precision ở recall cao |
| 4 | **Nhóm từ ASR nghe sai** (Sạch@1 chỉ 64.5) và list nghe giống (64.7): nhiều cách phát âm, nhiễu gần âm đặt đúng chỗ khi train | mục 3, 6 | Đây là nhóm quyết định B-WER |
| 5 | Train 960h (đã thấy +1.8 Sạch@1 với checkpoint cũ) | mục 3 | Bản cuối cho paper |
| 6 | Chạy đủ N100–2000, checkpoint Hou25 của Hoàng, Hou25-diag làm baseline mạnh; sau đó Qwen3-ASR | mục 8 | Bảng chính của paper |

## 2. Thước đo

Hou25 báo "từ đúng có lọt top-50 không". Ở mốc đó mọi hệ đều đạt 95–99%, nên khác biệt bị che. Báo cáo này dùng:

| Thước đo | Nghĩa |
|---|---|
| **Sạch@1** | % từ đúng không có từ nhiễu nào đứng trên nó (chỉ đếm nhiễu, để câu nhiều từ hiếm không bị phạt oan) |
| **Đúng@chỗ** | Tại đoạn audio của từ đúng, nó có đứng trên mọi từ nhiễu được scorer đặt cùng chỗ không |
| **Định vị** | Sai số tâm (giây), và % tâm nằm trong đoạn thật ± 0.1 s (mốc: căn chỉnh MFA) |
| **Tại ngưỡng p ≥ τ** | Giữ được bao nhiêu % từ đúng; lọt bao nhiêu từ nhiễu mỗi câu; precision |
| **Báo nhầm theo độ dài câu** | Số từ nhiễu p ≥ 0.5 mỗi câu, chia theo câu ngắn / dài |

## 3. Scorer hiện có: Hou25 so với scorer âm vị (test-other, N2000) [đo]

`hou25` = bản tái hiện của nhóm (TTS + cross-attention + CNN, 2 epoch). `pmaxsim` = scorer âm vị kiểu late-interaction của nhóm. Cả hai chạy có cửa sổ trượt 30 s.

| | Sạch@1 | Sạch@5 | Sạch@10 | Đúng@chỗ (câu 1 / 2–4 / ≥ 5 từ hiếm) | Sai số tâm | % tâm trúng | Thời gian chấm / câu |
|---|---|---|---|---|---|---|---|
| Hou25 | 72.6 | 87.1 | 90.9 | 90.2 / 91.4 / 92.1 (vị trí không tin được) | 0.69 s | 44.0 | ~1200 ms |
| pmaxsim (clean-100) | **85.7** | 95.2 | 96.8 | 92.6 / 94.1 / 96.0 | **0.05 s** | **98.8** | 4–7 ms |
| pmaxsim (960h) | **87.5** | 96.2 | 97.5 | 94.7 / 94.9 / 95.8 | 0.07 s | 98.8 | 4–7 ms |

Δ Sạch@1 so với Hou25: pmaxsim +13.1 [CI 95%: +11.5; +14.7]; pmaxsim 960h +14.8 [+13.3; +16.3].

Chi tiết:
- **Báo nhầm theo độ dài câu** (số từ nhiễu p ≥ 0.5 mỗi câu, câu < 5 s → ≥ 20 s): Hou25 1.4 → 12.4; pmaxsim 0.5 → 1.6.
- **Nhóm từ Whisper nghe sai** (986 lượt), Sạch@1: Hou25 45.9; pmaxsim 61.3. Đây là nhóm cần scorer nhất, và cũng là chỗ còn yếu nhất.
- **List nhiều từ nghe giống** (`hard_N2000`), Sạch@1: Hou25 51.4; pmaxsim 61.2 (+9.7 [+7.9; +11.4]). TTS không cho lợi thế khi gặp từ nghe giống. Trên list này, 21% số lần từ đúng bị vượt là do từ gần âm (list thường: 3–7%).
- **Lỗi cắt 30 s:** trước khi sửa, 5–6 / 6 từ bắt đầu sau giây 30 bị sót ở mọi hệ; sau khi dùng cửa sổ trượt: 0 / 6.

**Rút ra [diễn giải]:**
- Scorer âm vị xếp hạng tốt hơn hẳn Hou25, định vị gần như hoàn hảo, và nhanh hơn khoảng 200 lần.
- Hou25 gần như không định vị được từ bản đồ điểm sau CNN (44%).

## 4. Có vị trí thì cho các từ "tranh nhau tại chỗ" (không train thêm) [đo]

Cách làm: trong top-50, hai từ có đoạn audio chồng nhau (IoU > 0.3) thì chỉ giữ từ điểm cao hơn. Sau đó lấy các từ có p ≥ τ.

| test-other, N2000 | τ | Giữ từ đúng | Nhiễu / câu | Precision |
|---|---|---|---|---|
| Hou25 top-50 (cách của paper) | – | 95.4% | ~48 | ~3–4% |
| pmaxsim | 0.5 | 91.2 → **89.9** | 0.71 → **0.40** | 69.5 → **80.0** |
| pmaxsim | 0.3 | 95.9 → 92.9 | 3.73 → 1.38 | 31.5 → 54.6 |
| pmaxsim 960h | 0.5 | 94.4 → 92.5 | 1.07 → 0.52 | 61.2 → 76.2 |
| pmaxsim, test-clean | 0.5 | 96.4 → 95.7 | 0.52 → 0.24 | 80.1 → **89.5** |
| pmaxsim, list nghe giống | 0.5 | 91.2 → 81.6 | 4.64 → 0.62 | 26.0 → **70.3** |

(dạng "a → b": trước → sau khi cho tranh nhau)

**List toàn nhiễu** (`missing_N2000`, audio không có từ nào trong list), τ = 0.5:

| | Từ lọt / câu | Câu trả về rỗng |
|---|---|---|
| Hou25 | 2.60 | 35% |
| pmaxsim | 0.71 → 0.61 | 56% |
| pmaxsim, τ = 0.7 | 0.12 | 90% |

**Rút ra [diễn giải]:**
- Từ list 2000 từ, scorer có vị trí trả về khoảng 2 từ mỗi câu với precision 80–90%, mà vẫn giữ khoảng 90–96% từ đúng. Đây là điều top-50 của Hou25 không làm được.
- Vị trí là thứ cho phép loại nhiễu: hai từ không thể cùng nằm ở một chỗ.

## 5. Giả thuyết của Hoàng: đổi cách gộp của Hou25 rồi train lại (E2) [đo]

Ba bản Hou25 train chung một lượt (cùng dữ liệu, seed, lr 5e-5, 2 epoch, chọn ckpt theo dev-clean), chỉ khác bước gộp bản đồ điểm thành một số. Chấm đủ 2939 câu test-other, N2000.

| Cách gộp | Sạch@1 | Δ so với trung bình [CI 95%] | Báo nhầm, câu ≥ 20 s | Sạch@1, từ Whisper nghe sai | Sai số tâm | % tâm trúng |
|---|---|---|---|---|---|---|
| Trung bình cả bản đồ (Hou25 gốc) | 73.3 | – | 6.1 | 46.3 | 0.06 s | 79.3 |
| Top-k frame cao nhất | 73.8 | +0.5 [−0.9; +1.9] | 8.0 | 47.2 | 1.89 s | 3.6 |
| **Vệt chéo cao nhất (diag)** | **82.9** | **+9.5 [+7.9; +11.2]** | **1.9** | **60.9** | **0.05 s** | **98.9** |
| (tham chiếu) pmaxsim, train lại | 87.5 | +14.2 [+12.6; +15.7] | 1.3 | 64.5 | 0.05 s | 99.0 |

**Rút ra:**
- Giả thuyết của Hoàng **đúng khi train lại**: đọc điểm theo vệt chéo thay cho trung bình làm Sạch@1 tăng 9.5 điểm, báo nhầm ở câu dài giảm 3 lần, và cùng một model định vị được từ (98.9%).
- A1 trước đây (chỉ đổi lúc test) không thấy lợi, vì CNN chưa được train cho cách đọc mới.
- Lấy top-k không theo hình vệt thì không giúp. Thứ có ích là **hình dạng vệt** (đúng thứ tự, liền nhau), không phải việc "đọc cục bộ" nói chung.
- Hou25-diag vẫn thua scorer âm vị 4.6 điểm và chậm hơn khoảng 200 lần. Nó nên là **baseline mạnh** trong paper ("Hou25 + vị trí"), không phải nền.

## 6. Scorer mới pmono (khớp âm vị đúng thứ tự) so với pmaxsim, train cùng điều kiện [đo]

Cả hai train 6 epoch trên clean-100 (feature encoder cache sẵn), chọn ckpt theo Sạch@1 trên dev-other N2000.

| test-other | Sạch@1 | Sạch@1 list nghe giống | Nhiễu p ≥ 0.5 / câu (câu ≥ 20 s) | Precision / recall ở p ≥ 0.5 | Sai số tâm | Lệch biên đầu, 10%–90% |
|---|---|---|---|---|---|---|
| pmaxsim_c4 | **87.5** | **64.7** | 1.3 | 75.8 / 91.7 | 0.05 s | [−0.12; +0.08] s |
| pmono_c4 | 85.8 (−1.7 [−2.7; −0.7]) | 60.5 (−4.2 [−5.7; −2.6]) | **0.9** | **82.3** / 88.4 | **0.02 s** | **[+0.03; +0.10] s** |

- Ép thứ tự **không** làm xếp hạng tốt hơn; trên list nghe giống còn kém hơn.
- Đổi lại pmono ít báo nhầm hơn, và biên rất ổn định: gần như không bao giờ mở sớm.
- Thử thêm (chỉ có số dev-other, 500 câu, sai số khoảng ±1.5):
  - Train với 16 nhiễu + BCE cân bằng: kém hơn 4 nhiễu (pmaxsim 82.1 so với 87.7 sau 3 epoch) → bỏ dạng này.
  - Train **có dạy vị trí** (từ đúng chỉ được tính điểm trong span MFA): pmono 88.4 (không dạy: 87.3), pmaxsim 88.8 (88.2). **Chưa chấm trên test.**

## 6b. Hai cách khớp bù nhau: cộng điểm pmaxsim + pmono (CPU, 15/11) [đo]

Cộng logit của hai model trên cùng list (script `ensemble_cpu.py`, bảng `ensemble_other.md`). Không train gì thêm.

| | Sạch@1 test-other | Sạch@1 list nghe giống | Sạch@1 test-clean |
|---|---|---|---|
| pmaxsim_c4 (khớp tự do) | 87.5 | 64.7 | 93.4 |
| pmono_c4 (khớp đúng thứ tự) | 85.8 | 60.6 | 92.9 |
| **Cộng 1:1** | **90.1** (+2.6 [+1.9; +3.3]) | **68.8** (+4.0 [+3.0; +5.0]) | **95.1** (+1.7 [+1.1; +2.1]) |

Ở ngưỡng p ≥ 0.5 sau khi tranh nhau tại chỗ (test-other): giữ 90.3% từ đúng, 0.16 từ nhiễu/câu, **precision 91.2%** (pmaxsim riêng: 85.0%; pmono riêng: 87.3%).

**Đối chứng** (cùng script đo, test-other; số lệch nhẹ so với bảng trên vì cách xử lý hoà điểm):
- pmaxsim_c4 riêng: 87.2.
- Cộng **hai bản pmaxsim** (cùng kiểu, cùng dữ liệu): 88.3 (+1.1).
- Cộng pmaxsim + pmono: 89.8 (+2.6).
- Cộng pmaxsim + pmaxsim train 960h (gấp 9.6 lần dữ liệu): 89.7.

**Rút ra [diễn giải]:**
- Khoảng +1.1 là lợi ích chung của việc gộp hai model; khoảng **+1.5 còn lại là do hai cách khớp nhìn khác nhau**. Thêm nhánh khớp đúng thứ tự cho lợi ngang với thêm 9.6 lần dữ liệu train.
- Đây là kiến trúc scorer đề xuất: **một model, hai nhánh đọc trên cùng bản đồ giống nhau** (nhánh tự do để không sót, nhánh đúng thứ tự để loại nhiễu và cho biên), train chung. Hai nhánh dùng chung encoder nên chi phí gần như không tăng.
- **Chưa đo:** bản một model train chung. Số ở trên là hai model train riêng rồi cộng điểm.

## 6c. Dạy vị trí lúc train (chấm trên test, 15/11) [đo]

Cách dạy: với từ đúng, điểm chỉ được tính trong đoạn MFA của nó (± 0.1 s); từ nhiễu vẫn được khớp ở bất kỳ đâu. Mọi thứ khác giữ nguyên (clean-100, 6 epoch).

| test-other, N2000 | Sạch@1 | Sạch@10 | Sạch@1, từ Whisper nghe sai | Đúng@chỗ (1 / 2–4 / ≥ 5 từ hiếm) | Sạch@1 list nghe giống |
|---|---|---|---|---|---|
| pmaxsim | 87.5 | 97.4 | 64.5 | 94.9 / 94.9 / 95.8 | 64.7 |
| pmaxsim + dạy vị trí | 88.3 (+0.9 [+0.2; +1.5]) | 97.3 | 65.9 | 94.8 / 95.4 / 95.7 | 65.0 |
| pmono | 85.8 | 97.3 | 64.4 | 94.9 / 95.6 / 96.3 | 60.5 |
| **pmono + dạy vị trí** | **88.3** (+2.5 so với pmono) | **98.0** | **69.2** | **95.9 / 96.2 / 97.5** | 65.3 |
| **Cộng điểm hai bản có dạy vị trí** | **90.8** (+2.5 [+1.8; +3.1] so với pmaxsim + vị trí) | 98.2 | – | – | **69.6** (+4.6 [+3.5; +5.8]) |

Ở ngưỡng p ≥ 0.5 + tranh nhau tại chỗ (bản cộng điểm): giữ 91.1% từ đúng, 0.16 từ nhiễu/câu, precision 91.3%. Trên list toàn nhiễu: 0.24 từ/câu (pmaxsim cũ: 0.61).

**Rút ra:**
- Dạy vị trí lúc train giúp cả hai nhánh, giúp nhánh đúng thứ tự nhiều nhất (+2.5), và giúp nhiều nhất ở nhóm từ Whisper nghe sai (64.4 → 69.2).
- Chuỗi cải thiện trên cùng dữ liệu train: 87.5 (pmaxsim) → 88.3 (+ dạy vị trí) → 90.8 (+ nhánh đúng thứ tự). So với Hou25 tái hiện (72.6–73.3): +17 điểm.

## 7. Đưa vào Whisper-large-v3-turbo (N2000, greedy, có chống lặp) [đo]

Tham số chọn trên dev-other 400 câu: lấy từ có p ≥ 0.3 sau khi tranh nhau tại chỗ (khoảng 3 từ/câu); λ = 5 (không prompt) hoặc 3 (có prompt); biên 0.05 s; mỗi từ chỉ đẩy một lần.

| test-other | Từ / câu | WER | U-WER | B-WER | Chèn sai |
|---|---|---|---|---|---|
| Không context | 0 | 4.25 | 2.63 | 17.92 | 13 |
| **Hou25: prompt top-50** | 50 | 3.39 | 2.65 | 9.69 | 39 |
| pmaxsim: prompt top-50 | 50 | 3.42 | 2.63 | 10.07 | 56 |
| pmono, đẩy **không** vị trí | ~3 | 3.73 | 3.25 | 7.83 | 71 |
| **pmono, đẩy có vị trí (không prompt)** | ~3 | **3.24** | 2.66 | 8.14 | 56 |
| pmono: prompt top-50 + đẩy có vị trí | 50 + ~3 | 3.25 | 2.83 | **6.81** | 130 |
| Trần: từ đúng + vị trí MFA | – | 2.99 | 2.63 | 5.99 | 7 |

| test-clean | WER | U-WER | B-WER |
|---|---|---|---|
| Không context | 2.16 | 1.34 | 8.64 |
| Hou25: prompt top-50 | 1.66 | 1.22 | 5.15 |
| pmono, đẩy có vị trí (không prompt) | 1.64 | 1.37 | **3.79** |
| pmono: prompt + đẩy có vị trí | **1.55** | 1.33 | **3.22** |
| Trần | 1.53 | 1.37 | 2.77 |

Khoảng tin cậy 95% (bootstrap theo câu), so với Hou25 prompt:

| | test-other | test-clean |
|---|---|---|
| Không prompt, ΔB-WER | **−1.56 [−2.27; −0.88]** | **−1.35 [−1.90; −0.86]** |
| Không prompt, ΔWER | −0.15 [−0.37; +0.14] (chưa có ý nghĩa) | −0.02 [−0.17; +0.16] (chưa có ý nghĩa) |
| Không prompt, ΔU-WER | +0.02 [−0.19; +0.31] | +0.15 [−0.01; +0.32] |
| Prompt + đẩy, ΔB-WER | **−2.88 [−3.51; −2.29]** | **−1.92 [−2.42; −1.45]** |
| Prompt + đẩy, ΔWER | **−0.14 [−0.26; −0.03]** | **−0.12 [−0.20; −0.03]** |
| Prompt + đẩy, ΔU-WER | +0.18 [+0.07; +0.28] (tệ hơn) | +0.11 [+0.05; +0.18] (tệ hơn) |

**Có cần vị trí không?** Cùng list pmono, test-other: đẩy có vị trí so với không vị trí cho ΔWER −0.49 [−0.60; −0.39], ΔU-WER −0.58 [−0.69; −0.49]. **Có.**

**Audio không có từ nào trong list** (`missing_N2000`, test-other): không context 4.25; pmaxsim prompt top-50 4.49; pmono đẩy có vị trí 4.57 (chèn 81 từ); đẩy không vị trí 4.88; prompt + đẩy 4.87 (chèn 191). Mọi cách đều tệ đi khoảng 0.3 WER. Scorer chưa đủ "dám nói không".

**Phân rã khoảng cách tới trần** (dev-other 400 câu, B-WER): trần 3.99; từ đúng + vị trí do scorer đoán 5.61; list do scorer chọn + vị trí scorer 7.22. Vậy vị trí sai tốn khoảng 1.6 điểm, list sai (sót từ + nhiễu) tốn khoảng 1.6 điểm.

Ghi chú kỹ thuật:
- Greedy decode bị lặp chữ ở vài câu khi bị đẩy hoặc khi prompt quá ngắn (7–23 câu). Mọi hệ trong bảng dùng chung cơ chế decode lại khi tỉ lệ nén > 2.4 (như openai-whisper).
- Vị trí đoán mở sớm làm mất từ thường đứng trước ("the"). Dùng biên của pmono (không mở sớm) thì hết.
- Chưa chạy lại mốc "Hou25 prompt trên list toàn nhiễu" với chống lặp (bị tràn bộ nhớ). Số bản không chống lặp: WER 4.53.

## 7b. Bảng Whisper với scorer tốt nhất (cộng điểm hai nhánh, có dạy vị trí; chạy 15/11 sáng) [đo]

Scorer = pmaxsim_loc + pmono_loc cộng điểm; biên lấy từ pmono_loc. Tham số giữ nguyên như mục 7 (p ≥ 0.3 sau tranh nhau tại chỗ, λ = 5 hoặc 3, biên 0.05 s), không chọn lại trên test. Trung bình 2.3 từ/câu được đẩy.

| test-other | WER | U-WER | B-WER | Chèn sai |
|---|---|---|---|---|
| Không context | 4.25 | 2.63 | 17.92 | 13 |
| Hou25: prompt top-50 | 3.39 | 2.65 | 9.69 | 39 |
| Scorer mới: prompt top-50 | 3.43 | 2.65 | 10.03 | 49 |
| Scorer mới: đẩy **không** vị trí | 3.56 | 3.13 | 7.17 | 66 |
| **Scorer mới: đẩy có vị trí (không prompt)** | **3.14** | **2.62** | 7.53 | 49 |
| Scorer mới: prompt + đẩy có vị trí | 3.22 | 2.80 | **6.76** | 113 |
| Trần | 2.99 | 2.63 | 5.99 | 7 |

| test-clean | WER | U-WER | B-WER |
|---|---|---|---|
| Không context | 2.16 | 1.34 | 8.64 |
| Hou25: prompt top-50 | 1.66 | 1.22 | 5.15 |
| Scorer mới: đẩy có vị trí (không prompt) | 1.59 | 1.29 | 3.93 |
| Scorer mới: prompt + đẩy có vị trí | **1.47** | 1.26 | **3.12** |
| Trần | 1.53 | 1.37 | 2.77 |

So với Hou25 prompt (CI 95%, bootstrap theo câu):

| | test-other | test-clean |
|---|---|---|
| Không prompt: ΔWER | −0.25 [−0.47; +0.03] (sát ngưỡng, chưa có ý nghĩa) | −0.07 [−0.22; +0.10] |
| Không prompt: ΔB-WER | **−2.16 [−2.89; −1.54]** | **−1.22 [−1.76; −0.68]** |
| Không prompt: ΔU-WER | −0.03 [−0.23; +0.26] | +0.07 [−0.08; +0.24] |
| Prompt + đẩy: ΔWER | **−0.17 [−0.29; −0.06]** | **−0.19 [−0.27; −0.11]** |
| Prompt + đẩy: ΔB-WER | **−2.93 [−3.57; −2.38]** | **−2.02 [−2.51; −1.58]** |
| Prompt + đẩy: ΔU-WER | +0.15 [+0.05; +0.24] (tệ hơn) | +0.04 [−0.02; +0.10] |

Rút ra:
- **Kênh prompt đã bão hoà:** scorer tốt hơn 17 điểm Sạch@1 mà đưa top-50 vào prompt vẫn chỉ ngang Hou25 (10.03 so với 9.69). Scorer tốt chỉ phát huy khi có vị trí và đẩy lúc decode.
- **Vị trí là cần:** cùng list, đẩy có vị trí so với không vị trí cho ΔWER −0.42 [−0.51; −0.34].
- **Audio không có từ nào trong list:** WER 4.49 (test-other) và 2.31 (test-clean), so với không context 4.25 và 2.16. Vẫn tệ đi 0.15–0.25.

## 7c. Scorer hai nhánh trong MỘT model (pdual), train chung (15/11) [đo]

- Hai nhánh đọc bản đồ âm vị × frame rồi cộng điểm: nhánh khớp tự do + nhánh khớp đúng thứ tự.
- `pdual` = hai nhánh dùng chung một bản đồ (2.7 M tham số). `pdual2` = mỗi nhánh có bộ chiếu riêng (5.4 M tham số).
- Train 6 epoch trên clean-100, chọn ckpt theo Sạch@1 dev-other.

| test-other, N2000 | Sạch@1 | Sạch@10 | List nghe giống | Từ Whisper nghe sai | Sai số tâm | test-clean Sạch@1 |
|---|---|---|---|---|---|---|
| pmaxsim (mốc) | 87.5 | 97.4 | 64.7 | 64.5 | 0.05 s | 93.4 |
| pdual, không dạy vị trí | 88.4 | 98.2 | 66.3 | 67.8 | 0.02 s | 93.7 |
| pdual, có dạy vị trí | 88.3 | 98.3 | 66.1 | 68.8 | 0.02 s | 94.2 |
| pdual2, không dạy vị trí | 88.3 | 98.3 | 67.1 | 68.1 | 0.02 s | 93.9 |
| **pdual2, có dạy vị trí** | **89.3** (+1.8 [+0.9; +2.6]) | **98.6** | **67.7** (+3.0 [+1.6; +4.4]) | **70.9** | 0.02 s | **94.7** (+1.3 [+0.5; +2.1]) |
| (cộng điểm 2 model train riêng, mục 6c) | 90.8 | 98.2 | 69.6 | – | 0.02 s | – |

Rút ra:
- Một model hai nhánh có dạy vị trí là **scorer đơn tốt nhất**, lấy được hơn nửa phần lợi của bản cộng điểm (89.3 so với 90.8).
- Hai nhánh cần **bản đồ riêng** mới bù được nhau; dùng chung một bản đồ thì chỉ ngang các bản đơn có dạy vị trí (88.3).
- Điểm dev dao động mạnh giữa các epoch (84.8–90.2), nên việc chọn ckpt còn nhiễu. Cần dev lớn hơn hoặc lấy trung bình ckpt.

**Bản một model theo cỡ list** (test-other; p ≥ 0.3 sau tranh nhau tại chỗ; đưa vào Whisper bằng đẩy có vị trí, không prompt):

| List | Sạch@1 | Giữ từ đúng | Nhiễu / câu | Precision | WER | U-WER | B-WER |
|---|---|---|---|---|---|---|---|
| N100 | 98.2 | 96.2% | 0.04 | 97.8% | 2.96 | 2.46 | 7.14 |
| N500 | 94.1 | 95.7% | 0.21 | 89.2% | 3.04 | 2.52 | 7.37 |
| N1000 | 92.6 | 95.7% | 0.39 | 81.4% | 3.08 | 2.57 | 7.47 |
| N2000 | 89.0 | 95.0% | 0.73 | 69.9% | 3.22 | 2.69 | 7.65 |

(Không context: 4.25 / 2.63 / 17.92. Chưa có mốc Hou25 chạy lại ở N100–1000 trong cùng pipeline.)

**Bản một model đưa vào Whisper, N2000:**

| | test-other WER / U / B | test-clean WER / U / B |
|---|---|---|
| Hou25 prompt top-50 | 3.39 / 2.65 / 9.69 | 1.66 / 1.22 / 5.15 |
| Một model: đẩy không vị trí | 3.66 / 3.24 / 7.22 | – |
| **Một model: đẩy có vị trí, không prompt** | 3.22 / 2.69 / 7.65 | 1.58 / 1.30 / 3.83 |
| Một model: prompt + đẩy có vị trí | 3.26 / 2.89 / **6.42** | 1.51 / 1.30 / **3.21** |

- So với Hou25 (không prompt): ΔB-WER −2.04 [−2.76; −1.37] trên test-other và −1.32 [−1.83; −0.81] trên test-clean. ΔWER −0.17 [−0.39; +0.12] và −0.08 [−0.23; +0.09], chưa có ý nghĩa.
- Có vị trí so với không vị trí: ΔWER −0.44 [−0.55; −0.35].
- List toàn nhiễu: WER 4.51 (test-other), 2.29 (test-clean).

## 8. Chưa làm / giới hạn

- Hou25 là bản tái hiện của nhóm (2 epoch), chưa phải checkpoint 3 epoch của Hoàng.
- Bảng Hou25 chỉ có ở N2000; N100 / 500 / 1000 mới chạy cho scorer của nhóm (mục 7c). Chỉ Whisper; chưa thử Qwen3-ASR.
- Bản cộng điểm (mục 6c, 7b) là hai model train riêng; bản một model (mục 7c) còn kém nó 1.5 điểm Sạch@1.
- pmaxsim trong bảng Whisper là checkpoint cũ (Sạch@1 85.7); bản train lại (87.5) mới có một dòng ghép (WER 3.33 / B-WER 8.46 trên test-other).
- τ, λ, biên chọn trên 400 câu dev-other: sai số chọn tham số còn lớn.
