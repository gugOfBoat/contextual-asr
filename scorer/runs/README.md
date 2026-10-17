# Kết quả chạy của scorer (kéo về từ máy GPU)

Bảng số và cách đọc: `../results.md`. Mỗi thư mục = 1 đợt chạy.

## `2026-10-15/` — dữ liệu train/dev + đợt TTS
| File | Là gì |
|---|---|
| `train.jsonl` / `dev.jsonl` | train-clean-100 (28.539 câu) / dev-clean (2.703 câu): `uid, path, words, pos` (pos = từ hiếm của câu). `path` là đường dẫn trên server (`/work/libri/...`) |
| `near.json` | 29.811 từ hiếm → 20 từ gần nhất theo edit distance (negative khó) |
| `vocab.txt` | 209.604 từ cần TTS / g2p |
| `log_tts.txt`, `log_data.txt`, `tts_fail.txt` | log edge-tts (0 lỗi) và encode Whisper |
| `bias.idx` (không đưa lên git) | chỉ mục của `bias.f16` |

**Không kéo về được:** `bias.f16` (21,3 GB feature TTS) và 209k file mp3 TTS — **đã bị xoá** (16/10).
Muốn chạy lại `hou25 / nocross / maxsim` (bản dùng TTS) phải dựng lại: `data/tts.py` (~80 phút) + `data/encode_bias.py` (~25 phút).

## `2026-10-16/` — scorer âm vị + nhãn "từ khó" + chạy Qwen
| File | Là gì |
|---|---|
| `ckpt/{pxattn,pmaxsim,hxattn}/best.pt` | checkpoint dùng cho mọi số trong `results.md` (11–20 MB) |
| `log_train_*.txt` | log train 5 epoch (loss, dev Recall@10 / Hard-R@5) |
| `log_pipe_phone.txt` | chấm list test + bảng Precision/Recall của pxattn, pmaxsim, cascade |
| `log_hx.txt` | như trên cho hxattn + kết quả Qwen của hxattn |
| `log_asr.txt` | Qwen3-ASR trên 9 kiểu list × 2 tập + B-WER |
| `log_pass1.txt`, `pass1_*.jsonl` | Qwen không context trên train/dev → `hyp` + nhãn `hard` / `easy` |
| `phones.tsv` | âm vị (g2p_en) của 217.029 từ |
| `out/<model>/<split>/prob_<list>.json` | `{uid: [[từ, xác suất], ...]}` top-50 — đủ để dựng lại list lọc theo ngưỡng bất kỳ |
| `out/<model>/<split>/time.json` | thời gian chấm trung bình mỗi câu |

Output Qwen của các đợt này nằm ở `bench/out/<split>/qwen3asr_<ctx>.tsv` (+ `.raw.tsv`).

**Không kéo về (cố ý):**
- `out/*/rank_*.json` (691 MB, xếp hạng 1.000 từ/câu) — tính lại được từ checkpoint bằng `inference/rank.py`.
- `ckpt/*/last.pt` — chỉ giữ `best.pt`.
- `bench/protocol/*/ctx_<model>_*.json` (104 MB) — dựng lại từ `prob_*.json` hoặc `rank.py`.
- LibriSpeech audio, cache HuggingFace — tải lại từ nguồn công khai.

Máy GPU vẫn giữ LibriSpeech, checkpoint và output đầy đủ để chạy tiếp.
