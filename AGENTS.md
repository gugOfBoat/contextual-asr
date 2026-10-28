# Chỉ dẫn cho agent (Claude Code, Codex, agent cá nhân)

Dự án: **contextual ASR cho mô hình ASR dựa trên LLM (SpeechLLM) khi bias list lớn và nhiều nhiễu**: list càng dài thì từ đúng càng bị loãng và mô hình càng hay tự chèn từ trong list → cần cơ chế dựa vào audio để chọn (và định vị) đúng bias word, scale được tới hàng nghìn – hàng trăm nghìn từ. Tiếng Anh, target **Interspeech 2027** (hạn nộp 09/02/2027).
Đóng góp đang xây: **scorer** (audio + list → điểm từng từ), độc lập với mô hình ASR. **Qwen3-ASR-1.7B chỉ là một backbone thí nghiệm** (đang chạy thử); cắm vào nhiều SpeechLLM là phần thực nghiệm ở bước sau — không mô tả bài toán là "biasing cho Qwen". Bản đồ thư mục: `README.md`. Chỉ dẫn riêng của từng máy (đường dẫn, server): `AGENTS.local.md` nếu có (không đưa lên git).

## 0. Quy trình phiên
- Trả lời bằng **tiếng Việt**, ngắn gọn, đi thẳng vào việc. Người dùng mới làm nghiên cứu → giải thích bằng ví dụ/sơ đồ, tránh thuật ngữ trần.
- **Đầu phiên:** đọc `STATE.md` (và biên bản họp mới nhất trong `meetings/`).
- **Cuối phiên:** cập nhật `STATE.md` (mục 3–5) và ghi `log/YYYY-MM-DD.md`.
- Method **chưa chốt** → không tự ý code method khi người dùng chưa duyệt; đề xuất kế hoạch trước.

## 1. Quy tắc nghiên cứu
- Số liệu và claim về paper phải kiểm chứng từ nguồn gốc (PDF/arXiv/HTML), ghi kèm arXiv ID và Section/Table. Không chép từ trí nhớ. Chưa đọc toàn văn thì nói rõ "mới đọc abstract".
- Tách rõ **"paper nói"** và **"ta diễn giải"**. Không gán cho tác giả điều họ không viết.
- Mỗi lựa chọn setup phải ghi nguồn gốc, theo 3 loại:
  - **A** = theo nguồn (IS21, Hou25, PTC-Bias)
  - **B** = theo thông lệ
  - **C** = nhóm tự thiết kế, ví dụ `missing_N`, protocol 5 mức, ConEC
- Kết quả chỉ ghi vào `bench/results.md` khi **đã chạy thật**; ghi rõ normalizer (plain / wnorm), split, điều kiện context, số câu.
- `archive/` = hướng cũ đã bỏ. Không dùng làm căn cứ trừ khi người dùng mở lại.

## 2. Bối cảnh kỹ thuật đã chốt (chi tiết: `bench/README.md`)
- **Benchmark chính:** LibriSpeech test-clean (2620 câu) / test-other (2939 câu) + bias list IS21 (Le et al. 2021). File list: `bench/data/is21/`.
  - Từ hiếm = từ ngoài `common_words_5k`.
  - List của mỗi câu = từ hiếm đúng + N distractor, N ∈ {100, 500, 1000, 2000}.
  - Kho `all_rare_words.txt` (209k từ) lấy từ dữ liệu **train**, không phải rò rỉ test.
- **Điều kiện context** (`bench/protocol/<split>/ctx_*.json`):
  - `none`
  - `oracle`
  - `N{100,500,1000,2000}`
  - `missing_N{n}` (list chỉ có distractor)
  - (S2) kho 209k, không có list
- **Metric:** WER / U-WER / B-WER bằng `bench/score_is21_official.py` (sub=4, ins/del=3). B-WER tính trên từ hiếm của **reference**. Không thay bằng Levenshtein 1/1/1.
  - Báo số chính bằng `score.py --whisper-norm` (wnorm, khớp model card: none test-clean WER 1.63).
- **Paper mốc:**
  - Hou25 (Interspeech 2025): scorer TTS-audio + CNN chọn top-50 rồi đưa vào prompt; không có code, phải tự implement.
  - PTC-Bias (arXiv 2609.28727): retrieval theo phoneme + sửa sau decode; code: github.com/aizhiqi-work/PTC-Bias.
  - Cả hai đều cần list cho sẵn. Hướng của nhóm là **tự dựng list** (topic hoặc lần decode đầu → LLM).

## 3. Code (`bench/`)
| File | Việc |
|---|---|
| `build_is21.py` | dựng `protocol/<split>/` (`ref.tsv`, `audio.tsv`, `ctx_*.json`) |
| `run_qwen3asr.py` | Qwen3-ASR, context vào system prompt dạng `"w1, w2, ..."` |
| `run_whisper.py` | Whisper-large-v3-turbo (tái hiện Hou25); prompt cắt 223 token cuối; chạy tiếp được khi bị dừng |
| `score.py` | bọc scorer chính thức, `--whisper-norm`, chấm được tập con; `--ctx` phân rã lỗi từ hiếm (bỏ sót khi chọn / ASR không dùng được) |
| `build_lists.py` | list dev (theo quy tắc IS21), list nhiễu nghe giống `ctx_hard_N2000` + `near_N2000` |
| `eval_retrieval.py` | tầng (a): Recall_B#K, Recall_B@99, K99, Recall_H#K, P/R tại ngưỡng, từ khó (`EVAL_PROTOCOL.md`) |
| `compare.py` | ΔB-WER + khoảng tin cậy bootstrap, G (% khoảng trống lấp được) |
| `baseline_hypmatch.py` | baseline chọn từ không train (khớp âm vị với hyp lần 1, kiểu CTC-Filter) |
| `textnorm.py` | chuẩn hoá plain (lowercase, bỏ dấu câu) |

Output đặt tại `bench/out/<split>/<model>_<ctx><tag>.tsv`, kèm file `.raw.tsv` chứa text gốc.

Nguyên tắc: script ngắn, phẳng, giống style hiện có (argparse + vòng lặp). Không thêm abstraction khi chưa cần.

Scorer chấm bias word theo audio: `scorer/` (hướng dẫn `scorer/README.md`, số liệu `scorer/results.md`, file chạy `scorer/runs/`).
Dữ liệu lớn không có trong git: xem `README.md` mục "Không có trong git".

## 4. Môi trường
- **Máy local (Windows, không GPU):**
  - venv đặt ở đường dẫn **không dấu**, vì `nagisa` (phụ thuộc của `qwen-asr`) lỗi với đường dẫn có dấu tiếng Việt.
  - Đặt `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` khi script in tiếng Việt.
  - File tạm để trong scratchpad, không dùng `/tmp`: `/tmp` của Python khác `/tmp` của Git Bash.
  - Lệnh Bash gộp quá dài hay bị chặn → tách lệnh hoặc viết script.
  - Không chạy model lớn trên CPU (RAM thấp, tiến trình bị dừng).
- **GPU:** 1× RTX 4090 24 GB, Docker image `nvcr.io/nvidia/pytorch:*`. Không lưu password / token vào file.
  - Qwen N2000: OOM với bs16 → dùng `--bs 4` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Output nặng (feature, checkpoint trung gian, audio) để trên máy GPU; chỉ kéo về log, checkpoint cuối, kết quả.

## 5. Slide / tài liệu họp
- Slide Beamer theo template `archive/2026-09-calm-multispeaker/4-papers/slide/datasets-calm.tex` (Madrid, màu Primary `111B88`, Accent `F2A900`, T5 fontenc, biên dịch bằng `pdflatex`).
- Mỗi buổi họp có 1 thư mục `meetings/YYYY-MM-DD/` (slide + `imgs/`) và 1 file biên bản `meetings/YYYY-MM-DD.md`.
- Lỗi TikZ hay gặp: không đặt tên biến `\d \t \s` trong `foreach`, vì trùng macro dấu của T5.
