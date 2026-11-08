# Hiểu hệ thống từ A–Z: dữ liệu → scorer → ASR → các con số

Đọc theo đúng đường đi của **một câu nói**. Ví dụ xuyên suốt: câu `533-131562-0001` trong test-other.

## 0. Bức tranh tổng

```
                 ┌──────────────── SCORER (scorer/) ────────────────┐
audio .flac ───► │ Whisper encoder (đóng băng) → frame audio         │
list 2000 từ ──► │ g2p → âm vị → so audio với từng từ → điểm         │ ──► rank: 2000 từ xếp theo điểm
                 └───────────────────────────────────────────────────┘          │
                                                                                ├──► (a) đo CHỌN TỪ: Recall_B#K, @99, H#K
                                                                                ▼
                                          giữ từ có p ≥ 0.5 (≈3 từ/câu) ─► Qwen3-ASR (prompt) ─► văn bản
                                                                                                      │
                                                                                (b) đo ASR: WER, B-WER ◄┘
```

Có **2 loại số**:
- **(a)** chỉ đo scorer: từ đúng có lên đầu danh sách không.
- **(b)** đo cả hệ thống: ASR cuối cùng chép từ hiếm đúng không.

## 1. Dữ liệu (`bench/`)
| File | Nội dung | Ví dụ |
|---|---|---|
| `protocol/test-other/ref.tsv` | uid · câu đúng · **từ đúng** = từ hiếm của câu | `533-131562-0001` · "the keys of your cabinet desk drawers and whatever…" · `["cabinet", "drawers"]` |
| `data/is21/common_words_5k.txt` | 5.000 từ thông dụng | từ hiếm = từ **không** nằm trong file này |
| `data/is21/all_rare_words.txt` | kho 209k từ hiếm (lấy từ dữ liệu train) | nguồn rút từ nhiễu |
| `protocol/test-other/ctx_N2000.json` | list của từng câu = từ đúng + ~2000 từ nhiễu ngẫu nhiên trong kho | `["a'thing", "aaronic", …, "cabinet", …, "drawers", …]` (xếp a→z) |

Các list do IS21 (Le et al. 2021) công bố sẵn, nhóm không tự tạo → so được với mọi paper dùng cùng list.

## 2. Scorer: từ audio + 1 từ → 1 điểm
### 2.1 Phía audio (giống nhau ở mọi kiến trúc)
`scorer/common.py` → `Encoder`

```
wav 16 kHz  →  log-mel (128 × 3000, Whisper luôn pad tới 30 s)
            →  encoder Whisper-large-v3 (ĐÓNG BĂNG, không train)
            →  (1500 × 1280), cắt theo độ dài thật: câu 7 s ≈ 350 frame, mỗi frame 20 ms
```

⚠ Câu dài hơn 30 s bị cắt mất đuôi (test-other có 7 câu như vậy). Đây là một lỗi đang cần sửa.

### 2.2 Phía từ (âm vị)
`scorer/data/g2p.py` (thư viện g2p_en: tra từ điển CMUdict, không có thì đoán) → `work/phones.tsv`

```
cabinet  →  K AE B AH N AH T     →  id: [20, 2, 7, 3, 23, 3, 31]   (39 âm vị ARPAbet, 0 = pad)
```

⚠ Từ lạ g2p hay đoán sai. Ví dụ mảnh từ *n't* (trong "could n't") bị đọc thành chữ cái `EH N T AY`.

### 2.3 Ba cách so (`scorer/models.py`)
**pmaxsim** (2.72M tham số train), nhanh:
```
audio: (T × 1280) → Linear → 256 → 1 lớp Transformer → chuẩn hoá        = A (T × 256)
từ:    id âm vị → Embedding + vị trí → 2 lớp Transformer → chuẩn hoá    = P (7 × 256)
S = P · Aᵀ   (7 × T)   = độ giống (cosine) của mỗi âm vị với mỗi frame
mỗi âm vị lấy frame giống nhất (max theo T) → trung bình 7 âm vị = m
logit = a·m + c       (a, c học được; xác suất = sigmoid(logit))
```
Phía từ không phụ thuộc câu, nên tính trước 1 lần cho cả kho 209k từ. Không xét thứ tự âm: *stone* ≈ *notes*.

**pxattn** (4.96M), kỹ hơn:
```
âm vị P (7 × 256) làm "câu hỏi"
→ 2 lớp TransformerDecoder: (i) các âm vị nhìn nhau, (ii) mỗi âm vị nhìn vào toàn bộ audio A (cross-attention)
→ gộp 7 vector: [trung bình ; max]  (512)  →  MLP  →  logit
```

**hou25** (tái hiện Hou25, 1.95M): từ được biểu diễn bằng **audio TTS** (edge-tts đọc từ đó), cũng qua encoder Whisper.
```
cross-attention 2 chiều → 4 ma trận cosine (frame câu × frame từ) → CNN 4 lớp → trung bình → logit
```
CNN nhìn được "đường chéo" trên ma trận, tức thứ tự âm (ta diễn giải).

### 2.4 Train (`scorer/training/train.py`)
Mỗi câu train:
- **từ đúng** = từ hiếm của câu (tối đa 8);
- **từ sai** = 4 × số từ đúng, trong đó 40% gần chính tả (`near.json`) và 60% ngẫu nhiên trong kho 209k.

Loss = BCE từng cặp (câu, từ): đúng → 1, sai → 0. AdamW, lr 2e-4, 8 câu/bước.

| Bản | Dữ liệu | Số bước |
|---|---|---|
| 100 h | train-clean-100 (28.5k câu, giọng đọc rõ) | 5 epoch = 17.8k bước |
| 960 h | + clean-360 + other-500 (281k câu, có giọng khó) | 1 epoch = 35.2k bước; bản "cùng bước" = bước 17.836 |

Chọn checkpoint theo dev (dev-other, list 2000 từ thật).

### 2.5 Chấm list test (`scorer/inference/rank.py`)
Với mỗi câu: encoder chạy 1 lần → chấm cả 2000 từ → xếp giảm dần.

Đầu ra:
- `work/out/<model>/<split>/rank_N2000.json`: `{uid: [[từ, logit], …]}`, đủ 2000 từ → dùng cho số (a).
- `bench/protocol/<split>/ctx_<model>_N2000_p0.5.json`: chỉ giữ từ có sigmoid ≥ 0.5 (tối đa 50), xếp **tăng dần** (Hou25 Bảng 3: từ điểm cao đặt cuối) → đưa vào ASR.

Ví dụ: `533-131562-0001` → top: `cabinet 0.97, drawers 0.95, cabinets 0.41, …` → list đưa ASR: `["drawers", "cabinet"]` (số minh hoạ).

## 3. ASR (`bench/run_qwen3asr.py`)
Qwen3-ASR-1.7B, list đưa vào **system prompt** dạng `"w1, w2, w3"` (như prompt "naive" của Hou25). Không train gì.

Ra `bench/out/<split>/qwen3asr_<ctx>.raw.tsv`.

## 4. Các con số tính thế nào
### 4.1 Số (b): WER / B-WER (`bench/score.py` → `score_is21_official.py`, script chính thức của IS21)
1. Chuẩn hoá cả câu đúng và câu ASR bằng Whisper normalizer (`--whisper-norm`).
2. Căn chỉnh 2 câu với chi phí thay thế = 4, thêm/xoá = 3 (theo script IS21).
3. Đếm lỗi theo từng loại từ:
   - từ của câu đúng thuộc **danh sách từ đúng** → tính vào **B-WER**, còn lại vào U-WER;
   - từ ASR **chèn thêm**: tính vào B-WER nếu nó là một từ đúng, ngược lại vào U-WER. Chèn một từ nhiễu trong list sẽ rơi vào U-WER.

```
B-WER = (thay + xoá + chèn trên từ đúng) / tổng số lần từ đúng xuất hiện trong câu đúng
```

Ví dụ: ref "…your **cabinet** desk **drawers**…", ASR "…your **cabin** desk **drawers**…" → 1 lỗi / 2 từ đúng.

Số **4.78 / 9.71** = B-WER trên toàn bộ test-clean (2620 câu) / test-other (2939 câu) khi đưa list `ctx_pxattn_N2000_p0.5` vào Qwen (`scorer/results.md` §4).

### 4.2 Số (a): chất lượng chọn từ (`bench/eval_retrieval.py`, định nghĩa theo BR-ASR 2505.19179 / PTC 2609.28727)
"Từ đúng" = từ đúng của câu **và** có trong list. Test-other có 5.248 lượt.

| Metric | Cách tính | Ví dụ |
|---|---|---|
| **Recall_B#K** ↑ | % từ đúng nằm trong top-K của câu mình | 5.248 từ đúng, 5.166 lọt top-50 → #50 = 98.4 |
| **Recall_B@99** ↓ | gộp điểm của **mọi** từ, mọi câu; hạ dần **một ngưỡng chung** tới khi 99% từ đúng có điểm ≥ ngưỡng; đếm trung bình mỗi câu có bao nhiêu từ ≥ ngưỡng | ngưỡng −2.29 → trung bình 53.8 từ/câu ≥ ngưỡng → @99 = 53.8 |
| **K99** | như trên nhưng K cố định cho mọi câu (không cần điểm) | |
| **Recall_H#K** ↓ | H = từ **nhiễu trong list** nghe gần giống từ đúng (khoảng cách âm vị ≤ 2); % từ H lọt top-K | *cabinet* `K AE B AH N AH T` vs *cabinets* (+S) → thuộc H |

Vì sao @99 nhạy: 1% từ đúng tệ nhất (~52 từ) có điểm rất thấp → ngưỡng chung phải hạ rất sâu → mọi câu bị lọt thêm nhiều từ nhiễu. Phân tích: `scorer/runs/2026-10-18/diag_at99.py`.

## 5. Mỗi con số trong brief lấy từ đâu
| Con số | Lệnh / file |
|---|---|
| B-WER Qwen none / oracle / N100–2000 | `run_qwen3asr.py` + `score.py --whisper-norm` → `bench/results.md` |
| 4.78 / 9.71, top-50 5.71 / 11.57 | như trên, với `ctx_pxattn_N2000_p0.5` / `ctx_pxattn_N2000` → `scorer/results.md` §4 |
| Recall_B#50, @99, H#50 của nhóm, Hou25 | `rank.py` → `eval_retrieval.py --phones` → `scorer/runs/2026-10-18/log_night_metrics.txt`, `scorer/results.md` §8–9 |
| PTC 99.3 / 16.9 / 22.0, BR-ASR | chép từ paper: PTC 2609.28727 Bảng 3; BR-ASR 2505.19179 Bảng 2 |
| 77–85% từ bị rớt có điểm < 0.3 | `scorer/runs/2026-10-18/diag_rank.py` → `scorer/DEEPDIVE.md` §3 |
| "một nửa lỗi: có trong list mà vẫn sai" | `score.py --ctx` (phân rã lỗi) → `scorer/results.md` §6 |
| 54 từ tệ nhất, @99 ≈ 20 khi bỏ từ lỗi | `diag_at99.py` → `log_night_metrics.txt` (mô phỏng, không báo cáo) |

## 6. Tự chạy lại một con số (ví dụ Recall của pmaxsim 960 h)
Trên máy GPU, trong `scorer/`:
```bash
python -m inference.rank --ckpt work/ckpt/pmaxsim_960/step17836.pt --name pmaxsim_960s --split test-other --libri /work/libri/LibriSpeech --lists N2000
```
Rồi trong `bench/`:
```bash
python eval_retrieval.py --ref protocol/test-other/ref.tsv --scores ../scorer/work/out/pmaxsim_960s/test-other/rank_N2000.json --ctx protocol/test-other/ctx_N2000.json --phones ../scorer/work/phones.tsv
```
