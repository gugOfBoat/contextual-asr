# Bàn giao phiên làm việc (04/11/2026)

> Viết cho người hoặc Claude mới tiếp quản, không có lịch sử chat cũ. Đọc file này trước, rồi tới `STATE.md`.
> Số liệu đầy đủ không chép lại ở đây; mục 3 chỉ tới file gốc.

## 1. Đang làm gì, tới đâu

**Bài toán:** contextual ASR cho SpeechLLM nói chung (không phải "biasing cho Qwen"; Qwen chỉ là backbone thí nghiệm). Bias list càng dài và nhiều nhiễu thì mô hình càng bỏ sót từ đúng và tự chèn từ trong list. Target Interspeech 2027, hạn **09/02/2027**.

**Đóng góp đang xây:** scorer nghe audio. Vào: audio + list. Ra: mỗi từ **có được nói không (p)** và **nói ở đâu (t)**. Bản đang dùng, **chưa chốt**, là **pxattn**: âm vị (g2p) làm query, cross-attention vào encoder Whisper đóng băng. Hou25 dùng audio TTS thay cho âm vị.

**Giai đoạn hiện tại:** user muốn **dứt điểm scorer** rồi chuyển sang can thiệp logit lúc decode, dùng vị trí làm cổng.

```
Xong     chọn từ + đưa vào prompt; đo vị trí; giới tính; phân rã lỗi (phần trong pipeline nhóm)
Đang dở  S1: so ghép cặp với Hou25 của Hoàng → chờ file
Tiếp     S0 âm vị–âm vị thật → S2 temporal NMS → S3 decode lại → S4 3 seed → chốt scorer
Sau đó   D0 mổ logit → D1 boost có cổng p × thời gian → D2 sang Qwen → D3 đọc tài liệu logit
```

## 2. Việc ngay khi mở phiên mới

1. **Hỏi user kết quả buổi gặp thầy.** Lúc viết file này chưa có. Thầy có thể đổi thứ tự hoặc hướng: 5 câu cần thầy quyết nằm ở `meetings/2026-11-04/brief.md` mục 8. Ghi biên bản vào `meetings/`.
2. **Hỏi user đã xin được file của Hoàng chưa** (`exp/` không có trong git của `itshoang2024/hou25_interspeech`):
   - `exp/scorer_n{100,500,1000,2000}_k50_asc/<subset>.tsv`
   - `exp/scorer/is21/scores/<subset>.json`
   - `exp/localize/is21/*.json`

   Có file thì chạy `bench/error_breakdown.py` + `bench/compare.py` để xong S1. Trong repo Hoàng còn `exp/tl_pxattn_*`, `exp/tl_pmaxsim_*` (số của scorer nhóm chạy trong pipeline của Hoàng); so ghép cặp nên dùng các file này.
3. **S0 không phụ thuộc Hoàng và làm trước S2–S4**; xem mục 5. User đã đồng ý đưa S0 vào kế hoạch (04/11), nhưng chưa có lệnh code: đề xuất thiết kế trước.

## 3. Bản đồ tài liệu (đọc theo thứ tự)

| File | Là gì |
|---|---|
| `meetings/2026-11-04/brief.md` | **Tổng hợp mới nhất, đọc đầu tiên**: bảng số A–D, 11 thí nghiệm + insight, S1–S5, D0–D3, chọn SpeechLLM, lịch 14 tuần |
| `scorer/PLAN-chot-scorer.md` | Vì sao kiến trúc như vậy, hơn/kém gì, đối chiếu biên bản 24/10 và 1/11, 5 cách áp vị trí |
| `scorer/runs/2026-11-02/REPORT.md` | Báo cáo đầy đủ về vị trí và "hai phía không cùng là âm vị" (Q1–Q6) |
| `scorer/runs/2026-11-02/s1_breakdown.md` | Bảng phân rã lỗi 7 hệ × 2 tập |
| `scorer/results.md` §7–11 | Sổ số liệu scorer |
| `bench/results.md` | Sổ số liệu ASR (Qwen) |
| `EVAL_PROTOCOL.md` | Metric chọn từ |
| `AGENTS.md`, `AGENTS.local.md` | Quy tắc dự án, server |
| `meetings/2026-11-chuan-bi/` | Slide buổi 1/11. **Đã cũ** ở câu "thắng 8/8", xem mục 4 |

## 4. Những điều dễ hiểu sai

- **Chưa chứng minh âm vị + attention là tốt nhất.** pxattn được chọn vì không cần TTS, nhanh, chỉ đổi một biến so với Hou25 (loại [C]). Âm vị–âm vị thật chưa chạy: hypmatch chỉ đổi chữ Qwen sang âm vị; PTC-Bias là số paper và đang hơn ở tìm từ (99,30 / 16,9). 2 lớp cross-attention, lớp Transformer phía audio, pool, 256 chiều chưa có ablation. Vị trí hiện chỉ được đọc ra trong `position.py`, `rank.py` chưa xuất và thời gian của pxattn chưa đi vào ASR.
- **"pxattn thắng Hou25 8/8 ô" chỉ đúng khi so với số trong paper.** Trong bảng `table1.tex` của Hoàng (cùng pipeline, mlx-whisper), pxattn **kém 0,2–0,9 B-WER** (1 seed, chưa có CI). Slide 1/11 còn câu cũ. Claim đúng: gần bằng Hou25, đổi lại không cần TTS, nhanh hơn ~3 lần, lọc theo ngưỡng được, có vị trí tới mức âm vị.
- **Không tính "kém 0,2–0,9" từ Bảng A của brief.** Bảng đó đặt hàng Hoàng (mlx-whisper) cạnh hàng nhóm (pipeline nhóm) để xem. Khoảng 0,2–0,9 lấy từ `table1.tex`, nơi cả hai cùng pipeline.
- **G (% khoảng trống lấp được) của `compare.py` vô nghĩa trên Whisper**: oracle bị lặp chữ, nên test-other có oracle 26,24 tệ hơn cả list. Đừng báo G cho Whisper.
- **Chuẩn hoá:** bảng Whisper dùng plain (để so với Hou25), bảng Qwen dùng wnorm. Không so chéo.
- **PTC-Bias là số trên Qwen3-ASR-0.6B**, nhóm chạy 1.7B. Chỉ là tham chiếu.
- **Giả thuyết chưa kiểm trực tiếp:** khoảng cách với Hou25 là do từ nghe giống (R_H#50 21–26 so với 17). Bằng chứng còn yếu: pmaxsim để lọt nhiều từ nghe giống và chèn nhiễu nhiều hơn pxattn, nhưng từ nhiễu viết đè lên từ đúng chỉ tốn 0,14 / 0,34 điểm (pxattn N2000). Vì vậy NMS (S2) có thể giảm R_H#50 mà B-WER giảm ít.
- **Giới tính:** ước từ F0 với ngưỡng 150 Hz, không phải nhãn thật của từng câu; đọc kèm `gender.md`.
- **Chưa kiểm:** tài liệu điều chỉnh logit (mới biết tên paper); cách lấy "đang ở frame nào" trên Qwen; ứng viên model thứ 3 (license, bộ nhớ, cách đưa context).

## 5. Việc tiếp theo cụ thể

| # | Việc | Cần gì | Xong khi |
|---|---|---|---|
| S0 | Âm vị–âm vị thật: head CTC trên encoder Whisper, chấm bằng căn chỉnh; thêm pxattn 1 lớp. Tiêu chí viết trước khi chạy | 1–2 ngày code, GPU ~10 h (ước lượng) | Thay pxattn nếu R_B#50 ≥ pxattn, R_H#50 không tệ hơn, vị trí ≤ 0,05 s; nếu không giữ pxattn |
| S1 | Ghép cặp pxattn với Hou25 (Hoàng) | file của Hoàng | Có CI, biết phần chênh từ đâu |
| S2 | Temporal NMS: hai ứng viên chồng thời gian thì giữ từ p cao hơn. Trước hết thêm đầu ra `(từ, p, t_đầu, t_cuối)` từ attention lớp 1. Ngưỡng IoU chọn trên dev (rủi ro: từ ghép, hai từ đúng sát nhau) | GPU ~2 h, hoặc CPU trên tập con | R_H#50 giảm ≥ 3, R_B#50 giảm ≤ 0,1 |
| S3 | Decode Whisper với top-50 sau NMS + prompt hybrid theo thời gian của pxattn | GPU ~3 h | B-WER ≤ Hou25 (Hoàng) trong CI |
| S4 | 3 seed kiến trúc được chọn | GPU 3 × ~5 h | Biết phương sai |
| S5 | Negative khó theo khoảng cách âm vị (chỉ khi S2 không đủ) | GPU ~5 h | như S2 |

Phase decode: xem `scorer/PLAN-chot-scorer.md` §6 và brief mục 5. **Method chưa chốt: đề xuất và được user duyệt trước khi code.** S1 là thí nghiệm user đã bảo làm; S0 mới được đưa vào kế hoạch, S0 và S2 trở đi chưa có lệnh chạy.

Lệnh S1 (từ `bench/`, đặt `S=../scorer/runs/2026-10-16/out`, lặp `sp` cho 2 tập):
```bash
python error_breakdown.py --split $sp --md ../scorer/runs/2026-11-02/s1_breakdown.md --sys none=out/$sp/whisper_none.tsv: N100_full=out/$sp/whisper_N100.tsv:protocol/$sp/ctx_N100.json pxattn_N2000=out/$sp/whisper_pxattn_N2000.tsv:$S/pxattn/$sp/prob_N2000.json
```
```bash
python compare.py --ref protocol/$sp/ref.tsv --a out/$sp/whisper_pxattn_N2000.tsv --b out/$sp/whisper_pmaxsim_N2000.tsv
```
Lưu ý: `--md` là ghi nối thêm, chạy lại sẽ thêm bảng vào cuối file. Python local: venv ghi trong `AGENTS.local.md`, đặt `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.

## 6. Trạng thái máy (kiểm 04/11)

**Local:** chưa commit gì từ commit `8193991`. Chưa có trên GitHub:
- `STATE.md`, `scorer/results.md` (đã sửa);
- `HANDOFF.md`, `bench/error_breakdown.py`, `log/2026-11-02.md`, `meetings/2026-11-04/`, `meetings/2026-11-chuan-bi/`, `scorer/PLAN-chot-scorer.md`, `scorer/runs/2026-10-28/prompt_use.py`, `scorer/runs/2026-11-02/` (mới).

Chỉ commit khi user bảo. `scorer/runs/2026-11-02/out/` nặng khoảng 35 MB, nên cân nhắc trước khi đưa lên git.

**Máy GPU dùng chung** (đăng nhập, đường dẫn, container, dữ liệu còn lại: `AGENTS.local.md`, mục "Trạng thái 04/11"):
- Container làm việc đang chạy bản chỉ CPU; dữ liệu và checkpoint vẫn còn, chưa dọn. Chỉ dọn khi user bảo và hỏi trước khi xoá data quan trọng.
- Phiên trước cả 4 card đều bận. Muốn dùng GPU: kiểm lại, **xin user đồng ý card**, rồi tạo lại container gắn đúng card đó. Không đụng container hay tiến trình của người khác.
- Không lưu password hay token vào file.

## 7. Bẫy kỹ thuật đã gặp

- **torch 2.14a trong image 26.08 trên CPU:** SDPA đa luồng cho logit cross-attention sai và đổi theo số luồng. Sửa: `_MATH = sdpa_kernel(SDPBackend.MATH); _MATH.__enter__()`, phải giữ biến kẻo context bị thu gom. Feature Whisper không bị ảnh hưởng; kết quả GPU trước đây không liên quan.
- **Batch có padding/mask cho logit lệch** khi đọc attention → chấm từng ứng viên (`one_by_one` trong `position.py`).
- **Trong container không có torchaudio** → F0 tự tính bằng numpy.
- **Không có `gh` CLI** → repo private của Hoàng dùng `git clone`. Bản clone cũ nằm ở scratchpad của account cũ, cần clone lại.
- **Prompt 0–1 từ làm Whisper lặp chữ**: lọc p ≥ θ với prompt naive sẽ hỏng trên Whisper; trên Whisper dùng top-50 hoặc bọc spoken. Qwen không bị.
- `error_breakdown.py` phải đếm cả lượt chèn từ hiếm thì mới khớp `score.py`.

## 8. Cách user muốn làm việc

- Trả lời tiếng Việt, ngắn, đi thẳng vào việc; user mới làm nghiên cứu nên giải thích bằng ví dụ, tránh thuật ngữ trần.
- Tách rõ **paper nói / ta đo / ta diễn giải**. Số phải kiểm từ nguồn; chưa đọc toàn văn thì nói rõ.
- Nói thẳng khi kết quả không đẹp. Từng có một claim sai ("thắng 8/8") đã phải đính chính.
- Research: đọc nguồn sơ cấp, dùng WebSearch/WebFetch trực tiếp, **không dùng Workflow / multi-agent** trừ khi user bảo.
- Không tự code method khi chưa duyệt. Commit chỉ khi được yêu cầu.
- Cuối phiên: cập nhật `STATE.md` mục 3–5 và ghi `log/YYYY-MM-DD.md`.

Memory cũ của Claude trên máy này có file `project-thesis-vigencs.md` mô tả các hướng đã bỏ; đừng dùng làm căn cứ. Đường dẫn transcript phiên này: `AGENTS.local.md`.
