# Kế hoạch tới Interspeech 2027 (lập 08/11/2026)

> Mục tiêu và lý do: `GOAL.md`. File này là danh sách việc: mỗi việc có mô tả, đầu vào, output, DoD (thế nào là xong), hạn, người làm.
> Người làm: **L** = Lâm, **H** = Hoàng, **C** = agent (Claude). Hạn là ngày cuối cùng.
> Cập nhật trạng thái mỗi thứ Hai: ☐ chưa · ◐ đang làm · ☑ xong · ✗ bỏ (ghi lý do).

## Mốc và điểm quyết định

| Mốc | Ngày | Câu hỏi | Nếu KHÔNG đạt |
|---|---|---|---|
| **M1** | 22/11 | Gate với vị trí hoàn hảo có thắng Hou25 top-50 không? (P2.1) | phần adaptive chỉ còn là phân tích phụ; paper tập trung vào scorer |
| **M2** | 24/11 | Phần A có lộ ra gap của Hou25 bằng số không? (P1.8) | khối nào của DAS không có gap tương ứng thì bỏ hoặc đơn giản hoá |
| **M3** | 06/12 | Khối ① tìm được đoạn từ hiếm khi chưa có list? (P3.2) | dùng chế độ "ô cho mọi từ"; vẫn hỏng thì lùi về định vị theo từng từ |
| **M4** | 03/01 | DAS xếp hạng ≥ Hou25 và định vị ≤ 0.05 s? (P4.3) | claim còn: "ngang Hou25, có vị trí, nhanh hơn, không TTS" |
| **M5** | 17/01 | Adaptive dùng output DAS giảm B-WER trên ≥ 2 model? (P6) | báo trên Whisper; Qwen để phần thảo luận |
| **M6** | 05/02 | Nộp bản cuối (hạn thật 09/02; Tết khoảng 06/02, cần kiểm lại lịch) | — |

## P0. Chuẩn bị (09–20/11)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P0.1 | Xin GPU | xin 1 card cố định cho tới tháng 2 | — | số card được đồng ý | ghi số card + thời hạn vào `AGENTS.local.md` | 10/11 | L |
| P0.2 | Xin file của Hoàng | `exp/scorer/is21/` (best.pt, scores/), `exp/scorer/is21_s1only/best.pt`, `exp/localize/`, feature TTS (hoặc script dựng lại), code `src/hou25/` | — | file trên server `~/model_tmp/hou25/` | mở được bằng Python; số file và dung lượng khớp danh sách Hoàng gửi | 11/11 | L, H |
| P0.3 | Dựng môi trường | container `model` có GPU; cài transformers 4.57.6, soundfile, librosa, g2p-en, rapidfuzz | P0.1 | container chạy được | (1) `gate_boost.py` chạy 10 câu bằng turbo; (2) scorer Hou25 load ckpt, tính lại điểm 10 câu, lệch so với file của Hoàng < 1e-3 | 12/11 | C |
| P0.4 | Forced alignment | `bench/align_words.py` cho test-clean, test-other, dev-clean, dev-other, train-clean-100 | P0.3 | `protocol/<split>/align.tsv`; `align_train100.tsv` | ≥ 98% từ có posterior ≥ 0.5; 20 từ lấy ngẫu nhiên lệch trung vị ≤ 0.02 s so với alignment F14 của Hoàng | 13/11 | C |
| P0.5 | Novelty check | đọc toàn văn: CTC-WS (2406.07096), PTC-Bias (2609.28727 §2.3–2.4), Simul-Whisper (2406.10052), QbE-STD kiểu tìm đoạn rồi so, detector kiểu DETR cho keyword spotting | — | `related-work/novelty-DAS.md` | bảng "họ làm gì / DAS khác ở đâu", mỗi dòng có arXiv ID + Section; kết luận claim nào an toàn | 20/11 | C |

## P1. Phần A: chọc vào scorer Hou25 tìm gap (13–24/11)

Dùng checkpoint của Hoàng, không train lại. Mọi số chính có CI bootstrap theo câu. N2000 là ô chính, N100 để đối chiếu.

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P1.0 | Điểm + vị trí Hou25 ở N2000 | chạy lại scorer + `localize.py` của Hoàng cho N100 và N2000, cả 2 tập; lưu điểm mọi từ + vị trí top-50 | P0.2, P0.3 | `scorer/hou25/scores_N*.json`, `localize_N*.json` | R#10, R#50 khớp NOTES F9 của Hoàng (±0.002) | 14/11 | C |
| ◐ P1.1 | A4: từ đúng rớt top-10 vì ai (08/11: bản chỉ dùng điểm, trên Hou25 của nhóm, xong → `scorer/analysis/RESULTS.md`: **từ na ná KHÔNG phải thủ phạm**; câu dài và từ ngắn mới là chỗ rớt) | với mỗi từ đúng ngoài top-10: kẻ vượt cách bao nhiêu âm vị, span có trùng không; cận trên R#10 khi hạ distractor trùng span; chạy thêm trên `ctx_hard_N2000` / `near_N2000` | P1.0 | `scorer/analysis-hou25.md` §A4 | bảng % kẻ vượt theo loại (nghe giống trùng chỗ / nghe giống khác chỗ / khác); R#10 trước và sau cận trên; R#10 list ngẫu nhiên so với list nghe giống | 16/11 | C |
| P1.2 | A1: average pool làm loãng | đổi bước gộp lúc test: trung bình / max / cửa sổ quanh vệt / cửa sổ quanh span thật | P1.0, P0.4 | §A1 | R#10, H#50 cho 4 cách × 4 nhóm độ dài câu | 18/11 | C |
| P1.3 | A2: CNN có cần không | chấm chỉ bằng độ sáng vệt chéo tốt nhất, không CNN | P1.0 | §A2 | R#10, R#50, H#50 của vệt so với output Hou25, N2000 | 18/11 | C |
| P1.4 | A6 + A7: hiệu chỉnh và chi phí | @99; biểu đồ hiệu chỉnh (10 bin) + ECE; phân bố p trên `missing_N2000`; thời gian/câu ở N100–2000 | P1.0 | §A6, §A7 | có hình hiệu chỉnh, ECE, số từ p ≥ 0.5 trên `missing_N`, bảng ms/câu theo N | 19/11 | C |
| P1.5 | A3: thứ tự âm | (a) đảo ngược frame TTS của từ đúng; (b) list distractor hoán vị âm (cùng âm vị, khác thứ tự, lọc từ kho 209k) | P1.0 | §A3 | p trung bình của từ đúng trước và sau khi đảo; % từ hoán vị vượt từ đúng | 20/11 | C |
| P1.6 | A5: tốc độ TTS | đổi tốc độ TTS 0.8× / 1.2× rồi chấm lại | P1.0 | §A5 | R#10 và sai số định vị ở 3 tốc độ | 21/11 | C |
| P1.7 | Tổng hợp + duyệt thiết kế | gom A1–A7 thành bảng gap; quyết định giữ / bỏ / đơn giản hoá từng khối của DAS | P1.1–P1.6 | `scorer/analysis-hou25.md` bản đủ; mục DAS trong `GOAL.md` cập nhật | mỗi gap có 1 số chính + CI; mỗi khối DAS ghi rõ dựa vào gap nào; **L duyệt** | 24/11 (**M2**) | C, L |

## P2. Thử trần cho decode (song song với P1, 12–29/11)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P2.1 | E1: gate với vị trí hoàn hảo | `gate_boost.py`, whisper-large-v3-turbo. 5 cách: Hou25 top-50 / + boost không gate / + gate span / + gate tâm ± w / oracle prompt. Chọn λ, m, θ trên dev (list dev từ `build_lists.py`) | P0.3, P0.4, P1.0 (top-50 N2000) | `bench/out/<split>/whisper_*gate*.tsv`; `bench/results.md` mục E1 | bảng B-WER, U-WER, số chèn sai × N100/N2000 × 2 tập; CI ghép cặp so với Hou25 top-50; ghi kết luận M1 | 22/11 (**M1**) | C |
| P2.2 | E2: gate với vị trí Hou25 dự đoán | như P2.1, vị trí lấy từ P1.0 | P2.1 | `bench/results.md` mục E2 | cùng bảng như P2.1; % lợi ích giữ lại so với E1 | 29/11 | C |

## P3. DAS khối ①: tìm đoạn từ (25/11–06/12)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P3.1 | Cache feature Whisper | encoder large-v3 đóng băng, fp16, cho train-clean-100 + dev + test | P0.3 | `~/model_tmp/feat/` trên server | đủ câu; đọc ngẫu nhiên 10 câu khớp khi tính lại | 27/11 | C |
| P3.2 | Train khối ① | Linear + 2 lớp Transformer; nhãn từng frame: ngoài / bắt đầu / trong từ + độ lạ (ngoài `common_words_5k`); nhãn từ P0.4 | P3.1, P1.7 (L duyệt) | `scorer/das/`; `scorer/results.md` mục DAS-① | trên test: recall span từ hiếm (IoU ≥ 0.5) ≥ 98%; sai số ranh giới trung vị ≤ 0.05 s; ≤ 5 ô lạ/câu (hoặc ghi rõ dùng chế độ "mọi từ") | 06/12 (**M3**) | C |

## P4. DAS khối ② ③ và train đủ (07/12–03/01)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P4.1 | Mã hoá từ + ② truy hồi | âm vị → embedding + 2 lớp Transformer → vector từ (tính sẵn 209k); softmax trên list; loss tương phản; negative 40% nghe giống / 60% ngẫu nhiên | P3.2 | code + ckpt | recall top-10 của từ đúng trong ô đúng ≥ 99% ở N2000 | 13/12 | C |
| P4.2 | ③ căn chỉnh đơn điệu | độ khớp âm vị × frame học được; cộng mềm kiểu CTC có trạng thái nền; Viterbi ra ranh giới | — | code + unit test | test tổng hợp: tìm đúng span của chuỗi gài sẵn; đạo hàm kiểm bằng sai phân hữu hạn; chạy ≤ 5 ms cho 10 ứng viên × 1 ô trên GPU | 13/12 | C |
| P4.3 | Train DAS đầy đủ + đánh giá | ①②③ + lớp "không từ nào"; train-clean-100 (công bằng với Hou25); đánh giá N100–2000, 2 tập | P4.1, P4.2 | **Bảng 2** bản nháp trong `scorer/results.md` | có R#10, R#50, @99, H#50, sai số tâm / khởi đầu / kết thúc, % trong span (0 / ±0.1 / ±0.2 s), ms/câu; so với Hou25. Đạt **M4** khi R#10 ≥ 93.8 (tối thiểu, bằng Hou25) và định vị ≤ 0.05 s; mục tiêu R#10 ≥ 97.4 | 03/01 (**M4**) | C |

## P5. Ablation và độ chắc (04–17/01)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P5.1 | Ablation DAS | bỏ ① (quét cửa sổ trượt) / bỏ ② tranh nhau / ③ → gộp trung bình / ③ bỏ ràng buộc thứ tự / ③ → S1 + CNN cục bộ / có-không loss ranh giới / âm vị so với TTS | P4.3 | **Bảng 3** bản nháp | đủ 7 dòng, test-other N2000, mỗi dòng 1 seed | 10/01 | C |
| P5.2 | 960h + 3 seed | train DAS trên 960h; 3 seed cho cấu hình chính (100h) | P4.3 | `scorer/results.md` | mean ± std của Bảng 2 qua 3 seed; số 960h ghi riêng | 17/01 | C |

## P6. Adaptive: dùng output DAS trên SpeechLLM (28/12–17/01)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P6.1 | Whisper + DAS | chọn từ p ≥ θ (bù cho đủ độ dài prompt) + gate theo span của DAS | P2.1, P4.3 | `bench/results.md` | B-WER, U-WER, số chèn sai, N100–2000 + `missing_N2000`; CI so với Hou25 top-50 | 10/01 | C |
| P6.2 | Sửa hộp đen (U3) | decode lần 1 → căn hyp theo thời gian → đoạn trùng span từ chắc chắn, hyp viết khác nhưng cách ≤ 2 âm vị → thay | P4.3 | `bench/span_correct.py` | chạy trên output Whisper; báo B-WER và số lần thay sai | 10/01 | C |
| P6.3 | Qwen3-ASR | U1 + U3; thử U2 (đọc attention sang audio token), giới hạn 1 tuần | P6.1, P6.2 | `bench/results.md` | bảng như P6.1 cho Qwen; U2 có số, hoặc ghi rõ lý do không làm được | 17/01 (**M5**) | C |

## P7. Kết quả cuối (18–24/01)

| ID | Việc | Mô tả | Phụ thuộc | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|---|
| P7.1 | Đóng băng số | điền Bảng 1/2/3, CI ghép cặp, hình B-WER theo N, hình ví dụ bản đồ + span | P5, P6 | `paper/tables/`, `paper/figs/` | mọi ô có số; mỗi số truy được về file output + lệnh chạy | 24/01 | C |

## P8. Viết paper (song song, 14/12–05/02)

| ID | Việc | Mô tả | Output | DoD | Hạn | Ai |
|---|---|---|---|---|---|---|
| P8.1 | Outline + Hình 1 | khung 4 trang theo `GOAL.md`; hình Hou25 so với DAS | `paper/` | outline từng mục + hình nháp; L duyệt | 14/12 | C, L |
| P8.2 | Intro + Analysis + Method | viết từ kết quả P1 và thiết kế P4 | bản nháp | đủ 3 mục, có trích nguồn kiểm chứng | 05/01 | C, L |
| P8.3 | Bản nháp đủ | thêm Setup, Results, Conclusion | bản nháp | đủ 4 trang + references, biên dịch được bằng template Interspeech | 24/01 | C, L |
| P8.4 | Thầy duyệt | gửi thầy | bản gửi | gửi trước 26/01 | 26/01 | L |
| P8.5 | Sửa + nộp | sửa theo góp ý; kiểm ẩn danh và giới hạn trang | bản nộp | nộp trước Tết | 05/02 (**M6**) | L |

## Ước lượng GPU (1 card RTX 4090)

| Phần | Giờ GPU (ước lượng, chưa đo) |
|---|---|
| P0.4 alignment | khoảng 2 |
| P1 (chấm lại Hou25, các biến thể) | khoảng 10 |
| P2 (E1, E2: decode nhiều cấu hình) | khoảng 15 |
| P3 + P4 (train DAS) | khoảng 30 |
| P5 (7 ablation + 960h + 3 seed) | khoảng 60 |
| P6 (Whisper + Qwen) | khoảng 20 |
| **Tổng** | **khoảng 140** |

## Rủi ro chung
- **GPU hoặc file của Hoàng tới trễ:** P1 và P2 trễ theo. Trong lúc chờ: làm P0.5, viết sẵn code P1 và P4.2 (unit test chạy trên CPU).
- **Feature TTS của Hou25 không lấy được:** P1.5, P1.6 và dòng "S1 + CNN" của P5.1 phải dựng lại TTS (khoảng 80 phút TTS + encode, 21 GB).
- **Hết giờ:** cắt theo thứ tự: P6.3 U2 → P5.2 960h → P1.6 → P1.5.
