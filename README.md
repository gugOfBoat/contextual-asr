# Contextual biasing cho Qwen3-ASR (frozen) — target Interspeech 2027

Hạn nộp Interspeech 2027: 09/02/2027.
Trạng thái chi tiết, cập nhật mỗi phiên: **`STATE.md`** · Bước tiếp theo: **`NEXT_STEPS.md`** · Nhật ký: `log/`.

## Bài toán
ASR hay chép sai **từ hiếm** (tên riêng, thuật ngữ): nói *"missus **neverbend**"*, Qwen3-ASR chép *"missus **never been**"*.
Cách chữa: đưa trước một **bias list** các từ có thể xuất hiện vào prompt. Ngoài đời list thường **dài và lẫn nhiều từ không liên quan**
(vd cả danh bạ), mà list dài lại làm ASR rối.

Benchmark **IS21** (Le et al. 2021) mô phỏng đúng tình huống đó trên LibriSpeech: list của mỗi câu = từ hiếm thật của câu + N từ hiếm ngẫu nhiên, N = 100…2000.

| Qwen3-ASR-1.7B, B-WER test-clean / other (wnorm) | |
|---|---|
| không list | 7.40 / 16.08 |
| chỉ từ đúng (oracle, cận trên) | 4.01 / 7.73 |
| list N = 100 | 5.30 / 11.61 |
| list N = 2000 | 7.40 / 16.45 — list dài coi như vô dụng |

→ Cần một **scorer**: nghe audio, chấm mỗi từ trong list có thật sự được nói không, chỉ đưa vài từ đúng cho ASR.
Paper mốc: **Hou25** (Interspeech 2025, scorer TTS-audio + CNN, top-50) và **PTC-Bias** (arXiv 2609.28727, dò âm vị bằng CTC + sửa sau decode). Note: `related-work/`.

## Đã thử gì (chi tiết + số: `scorer/results.md`, `bench/results.md`)
| Ngày | Thí nghiệm | Kết quả chính | Ở đâu |
|---|---|---|---|
| 02–03/10 | Benchmark IS21 + baseline Qwen3-ASR (7 điều kiện × 2 tập), tái hiện xu hướng Hou25 trên Whisper | khớp model card (WER 1.63); list dài làm hại | `bench/` |
| 15/10 | Code scorer kiểu Hou25 (TTS) + 2 biến thể; dựng 21 GB feature TTS | train thử OK; **feature bị xoá** → bản Hou25 chưa train | `scorer/runs/2026-10-15/` |
| 16/10 | Scorer **âm vị** (không cần TTS): `pxattn`, `pmaxsim`, train 5 epoch | lọc N2000 còn ~3 từ/câu, giữ 95% từ đúng | `scorer/runs/2026-10-16/` |
| 16–17/10 | Đưa list đã lọc vào Qwen3-ASR (28 lượt) | **pxattn + lọc p ≥ 0.5: B-WER 4.78 / 9.71** (top-50 kiểu Hou25: 5.71 / 11.57) | `scorer/results.md` §4 |
| 16–17/10 | Nhãn "từ khó" (Qwen lần 1 chép sai) + `hxattn` (scorer biết lỗi của ASR) | Qwen sai 7% từ hiếm trên train; hxattn **kém** (6.58), nghi học tắt | `scorer/results.md` §2–4 |

So với PTC-Bias (số trong paper, N2000): ngang bản chỉ chọn từ CTC-Filter (4.81 / 9.45), còn thua bản đầy đủ (4.10 / 7.76).
Báo cáo họp 17/10 (10 slide + kịch bản): `meetings/2026-10-17/`.

## Bước tiếp theo
Dựng **setup chuẩn của Hou25** (Whisper-turbo + Qwen, N = 100…2000, top-50 và lọc ngưỡng, tái hiện scorer Hou25, thêm PTC-Bias) để biết scorer của nhóm hơn/kém ở đâu → **`NEXT_STEPS.md`**.

## Cấu trúc
| Thư mục | Nội dung |
|---|---|
| `bench/` | Benchmark: dựng protocol IS21, chạy Qwen3-ASR / Whisper, chấm WER/B-WER (scorer chính thức), `results.md`, output ASR (`out/`) |
| `scorer/` | Scorer chấm bias word theo audio: `data/ → training/ → inference/`, `results.md`, `runs/` (checkpoint, log), `docs/` (sơ đồ) |
| `related-work/` | Note từng paper (Hou25, PTC-Bias, …), bản đồ gap |
| `meetings/` | Slide + biên bản từng buổi họp |
| `method/` | Các bản đề xuất method (chưa chốt) |
| `kb-contextual-asr/` | Kiến thức nền |
| `log/` | Nhật ký theo ngày |
| `paper/` | Outline paper |
| `STATE.md`, `NEXT_STEPS.md` | Hiện trạng; kế hoạch tiếp |
| `AGENTS.md` (`CLAUDE.md`) | Quy ước làm việc cho agent / người (nguồn số liệu, nhãn [A]/[B]/[C], môi trường) |

## Không có trong git (tải / tạo lại)
| Thứ | Lấy lại thế nào |
|---|---|
| LibriSpeech audio | openslr.org/12 — `bench/data/README.md` |
| List IS21 N100…2000 (`*.biasing_*.tsv`, tới 67 MB) | `bash bench/data/download_is21.sh` |
| `bench/protocol/*/ctx_N*.json`, list đã lọc `ctx_<model>_*.json` | `python bench/build_is21.py`; `scorer/inference/rank.py` |
| PTC-Bias code + model | `bench/third_party/README.md` |
| Feature TTS 21 GB, mp3 TTS | **đã mất** — dựng lại bằng `scorer/data/tts.py` + `encode_bias.py` |
| File xếp hạng đầy đủ `rank_*.json` (691 MB), `last.pt` | không kéo về; tạo lại từ checkpoint |
| `archive/` (hướng cũ 07–09/2026) | chỉ giữ ở máy cá nhân |

## Môi trường
- Python ≥ 3.10, PyTorch, `transformers` 4.57, `qwen-asr`, `soundfile`, `librosa`, `g2p_en`, `rapidfuzz`, `edge-tts` (bản TTS).
- GPU: 1× RTX 4090 24 GB. Máy local không GPU chỉ dùng để chấm điểm / viết.
