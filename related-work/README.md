# Related work

Mỗi paper đọc sâu = 1 file `<arxiv-id hoặc tên-năm>.md`: bài toán, setting (pool cho sẵn? train?), dataset, số chính (kèm Table), khác gì với mình.

> ID dưới đây lấy từ ghi chép cũ. Mở nguồn gốc kiểm lại trước khi trích dẫn vào paper.

## Phải vượt / so trực tiếp
| Paper | ID | Ghi chú |
|---|---|---|
| Hou25 — Ranking and Selection of Bias Words (Interspeech 2025) | 10.21437/Interspeech.2025-646 | scorer TTS-audio↔speech có train, pool cho sẵn ≤2000, Whisper-turbo; không có code (kiểm 03/10). **Giải thích + mã giả:** `hou25-ranking-selection.md`, PDF: `hou25_interspeech.pdf` |
| PTC-Bias | 2609.28727 | training-free, phoneme posterior, định vị span + post-correction; có code; báo Qwen3-ASR trên IS21. **Đọc code:** `2609.28727-ptc-bias.md` |

## Benchmark / nền
| Paper | ID | Ghi chú |
|---|---|---|
| Le et al. 2021 — deep biasing, IS21 | — | nguồn protocol + scorer (`fbai-speech/is21_deep_bias`) |
| RAG-based context discovery for ASR (Findings EMNLP 2025) | 2509.19567 | retrieve từ vựng 466k từ transcript **segment trước**, ASR frozen black-box, có so CB-LLM; chỉ WER, long-form; đe doạ novelty "tự khám phá context" |
| UserLibri (Interspeech 2022) | 2207.00706 | văn bản phần còn lại của sách cho LibriSpeech test (đã lọc câu test) → nguồn "context thật" |

**Bản đồ gap đầy đủ (03/10):** `gap-map-2026-10-03.md`. Có thêm Suh24, RECAST, Gong24, LOGIC, Novitasari, Huber & Waibel, COALA, BR-ASR.

## Baseline training-free
| Paper | ID |
|---|---|
| TurboBias | 2508.07014 |
| CTC-WS | 2406.07096 |
| WCTC-Biasing | 2506.01263 |
| Zero-shot trie decoding, Whisper frozen | 2508.17796 |
| Apple phonetic retrieval | 2409.15353 |
