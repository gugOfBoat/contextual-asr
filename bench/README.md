# Benchmark contextual ASR — setup theo biên bản họp 17/09

Model nền: `Qwen/Qwen3-ASR-1.7B` (frozen). Ngôn ngữ: tiếng Anh. Mục tiêu: Interspeech 2027 (hạn 09/02/2027).

## 1. Dataset

| Vai | Bộ | Speech | Context có sẵn | Trạng thái |
|---|---|---|---|---|
| **S1 — chuẩn so sánh** | LibriSpeech test-clean/other + IS21 (Le et al. 2021) | thật | list/câu = từ hiếm đúng + N distractor, N ∈ {100,500,1000,2000} | ✅ đã dựng `protocol/`; audio test-clean có, **test-other chưa tải** |
| **S2 — không có list** | như S1, kho = `all_rare_words.txt` (209.291 từ) | thật | không có list/câu | ✅ có kho |
| **S3 — domain, giọng thật** | ConEC (Earnings-21/22 + slide, press release, danh sách người tham dự) | thật | context thật theo từng cuộc gọi | ⏳ chưa tải (audio ở repo rev earnings21/22) |
| S3' — domain, nhiều entity | ContextASR-Bench EN (MIT) | **TTS** | domain_label + entity_list | phụ |
| ✗ | thuhcsi/Contextual-Biasing-Dataset | thật | — | **tiếng Trung** (WenetSpeech, CB-Conformer ICASSP'23) → không dùng làm bộ chính |
| ✗ | ProfASR-Bench | **TTS** (kokoro) | prompt/domain | chỉ tham khảo |
| ✗ | Contextual Earnings-22 (2604.07354) | thật | local/global vocab | chưa public ("upon acceptance") |

## 2. Protocol context (mỗi điều kiện = 1 file `ctx_*.json`)

| Điều kiện | Ý nghĩa |
|---|---|
| `none` | không context |
| `oracle` | chỉ từ hiếm đúng → cận trên |
| `N{100..2000}` | setting của Hou25 / PTC-Bias (list **luôn chứa** từ đúng) |
| `missing_N{n}` | list chỉ có distractor → đo over-biasing / hallucination |
| (S2) kho 209k | method phải tự tìm list |

## 3. Metric

- **WER / B-WER / U-WER**: dùng `score_is21_official.py` (scorer gốc fbai-speech) để số so 1:1 với literature.
  - B-WER tính trên **từ hiếm của reference** (cột 3), KHÔNG phải cả list → baseline không đổi theo N.
  - Chi phí alignment sub=4, ins/del=3 (khác Levenshtein 1/1/1 của `kb-contextual-asr/tools/bwer.py`).
- **Recall@K, MRR, AvgLen**: `eval_retrieval.py` — đo riêng bước tạo candidate.
- (S3) NE-WER / keyword F-score theo đúng định nghĩa của từng bộ.

Sanity đã chạy (test-clean): hyp=ref → 0/0/0; xoá hết từ hiếm → B-WER 100, U-WER 0, WER 10.96.

## 4. Baseline

| Baseline | Code | Ghi chú |
|---|---|---|
| Qwen3-ASR, no context / + list trong system prompt | `run_qwen3asr.py` | cần GPU |
| PTC-Bias (arXiv 2609.28727) | ✅ github.com/aizhiqi-work/PTC-Bias | **đã báo Qwen3-ASR trên IS21**: N=2000, Qwen3-ASR+PTC-Bias B-WER 4.10/7.76, U-WER 1.17/2.68 |
| Hou25 scorer (Interspeech 2025) | ✗ không tìm thấy | tự implement: edge-tts + encoder Whisper-large-v3 + CNN, train clean-100 |
| CTC-WS (NeMo) | ✅ NeMo | baseline keyword boosting |

## 5. Chạy

Môi trường: venv có `qwen-asr`. **Phải đặt ở đường dẫn không dấu**: `qwen-asr` import `nagisa`, mà `nagisa` không đọc được model nếu đường dẫn có ký tự tiếng Việt có dấu.

```bash
python build_is21.py                                   # dựng protocol/ (~450MB)
python run_qwen3asr.py --split test-clean --ctx none oracle N100 N2000 missing_N100 --limit 50   # smoke test
python score_is21_official.py --refs protocol/test-clean/ref.tsv --hyps out/test-clean/qwen3asr_N100.tsv
python eval_retrieval.py --ref protocol/test-clean/ref.tsv --cand <cand.json>
```

## 6. Việc còn mở
1. Cần GPU để chạy `run_qwen3asr.py` (máy hiện tại không có GPU).
2. Tải LibriSpeech test-other (~330MB) và ConEC + audio Earnings-21/22.
3. Kiểm chuẩn hoá text: Qwen3-ASR xuất có hoa/dấu câu/chữ số, ref LibriSpeech viết số bằng chữ → xem `.raw.tsv` sau smoke test rồi chốt normalizer.
4. Đối chiếu số baseline Qwen3-ASR của mình với số của PTC-Bias (cùng setting) để chắc pipeline đúng.
