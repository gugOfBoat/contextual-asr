# Chương 9 — Thực hành: dựng baseline và chạy thí nghiệm đầu tiên

> **Mục tiêu chương:** trong vòng một tuần, có **4 baseline chạy được** và **một bộ metric tin cậy được**. Sau đó mọi ý tưởng mới đều có chỗ để so.

---

## 9.1 Toolkit nào có sẵn cái gì

| Toolkit | Kiến trúc | Biasing có sẵn | Ghi chú |
|---|---|---|---|
| **NeMo** (NVIDIA) | CTC, RNN-T/TDT, Canary | **CTC-WS**, GPU phrase-boosting (TurboBias) | Nhanh nhất để có baseline ký hiệu mạnh |
| **icefall / k2** | Zipformer CTC & RNN-T | **context graph** (trie + failure arc) | Code sạch, dễ đọc để học cơ chế |
| **WeNet** | Conformer CTC/AED | context graph, hotword | Thiên về triển khai sản phẩm |
| **FunASR** (Alibaba) | Paraformer | **SeACo-Paraformer hotword** | Dùng ngay được, mạnh với tiếng Trung; đáng thử với tiếng Việt |
| **ESPnet** | mọi thứ | recipe contextual/biasing, OWSM | Học thuật, nhiều biến thể |
| **HuggingFace** | Whisper, wav2vec2 | `initial_prompt` (Whisper), `pyctcdecode` hotwords (CTC) | Nhanh nhất để thử ý tưởng |
| **pyctcdecode** | mọi model CTC | `hotwords` + `hotword_weight` | Thư viện nhỏ, đọc source trong 30 phút |

**Đề xuất cho bạn**: bắt đầu bằng HuggingFace (nhanh, quen) + NeMo (baseline ký hiệu mạnh). icefall để **đọc code** hiểu context graph. FunASR nếu muốn thử nhánh non-autoregressive.

---

## 9.2 Bốn baseline dựng được trong tuần đầu

### Baseline 0 — không ngữ cảnh (bắt buộc)

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device=0)
hyp = asr("a.wav", generate_kwargs={"language": "vi", "task": "transcribe"})["text"]
```

### Baseline 1 — Whisper + prompt (họ 5, mức 1)

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

proc  = WhisperProcessor.from_pretrained("vinai/PhoWhisper-small")
model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-small").cuda().eval()

bias  = ["paracetamol", "amoxicillin", "ibuprofen"]
prompt_ids = proc.get_prompt_ids(", ".join(bias), return_tensors="pt").cuda()

feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
with torch.no_grad():
    ids = model.generate(feats, prompt_ids=prompt_ids, language="vi", task="transcribe",
                         num_beams=5)
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

**Việc phải làm ngay với baseline này** (chứ không chỉ chạy cho có): quét $N \in \{10, 50, 100, 500, 1000\}$ và vẽ WER theo $N$. Bạn sẽ **tự tay tái hiện được hiện tượng sụp đổ** ở báo cáo `01` §1.5. Một biểu đồ tự chạy đáng giá hơn mười trích dẫn.

### Baseline 2 — CTC + hotword boosting (họ 1)

```python
# pip install pyctcdecode kenlm
import numpy as np, torch
from transformers import AutoProcessor, AutoModelForCTC
from pyctcdecode import build_ctcdecoder

name  = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"      # ví dụ model CTC tiếng Việt
proc  = AutoProcessor.from_pretrained(name)
model = AutoModelForCTC.from_pretrained(name).eval()

vocab  = [k for k, _ in sorted(proc.tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
labels = [t.replace("|", " ") if t == "|" else t for t in vocab]
decoder = build_ctcdecoder(labels)          # thêm kenlm_model_path=... nếu có LM n-gram

inp = proc(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inp).logits[0].cpu().numpy()

# ★ đây chính là BIAS_SCORE trong pseudo-code chương 5 §5.3
text = decoder.decode(logits,
                      hotwords=["paracetamol", "amoxicillin"],
                      hotword_weight=10.0)
print(text)
```

**Việc phải làm**: quét `hotword_weight` ∈ {0, 2, 5, 10, 20} → tính B-WER/U-WER mỗi mức → **vẽ đường cong đánh đổi** (chương 8 §8.4). Đây là hình quan trọng nhất trong phần thực nghiệm của bạn.

### Baseline 3 — NeMo CTC-WS / GPU boosting (họ 1, bản mạnh)

NeMo có sẵn script cho context-biasing bằng CTC word spotter. Ý tưởng dùng:

```bash
# tinh thần (đối chiếu README của phiên bản NeMo bạn cài, API có đổi theo bản)
python .../speech_to_text_eval.py \
    pretrained_name=<model_ctc> \
    dataset_manifest=test_manifest.json \
    ctc_decoding.strategy=... \
    ctc_decoding.beam.beam_size=... \
    context_biasing.apply=true \
    context_biasing.context_file=bias_list.txt \
    context_biasing.score_threshold=...
```

Đây là baseline **phải có** nếu bạn định đề xuất phương pháp neural — vì nó rẻ, mạnh, scale tốt (chương 7 §7.2c).

---

## 9.3 Dựng bias list theo chuẩn để so được với thế giới

Theo Le et al. (báo cáo `01` §2.1):

```python
import json, random
from collections import Counter

def build_bias_lists(train_texts, test_items, N=1000, n_common=5000, seed=0):
    """test_items: [{'id':..., 'text':...}]  ->  {utt_id: [cụm bias]}"""
    rng  = random.Random(seed)
    freq = Counter(w for t in train_texts for w in t.split())
    common = {w for w, _ in freq.most_common(n_common)}
    rare   = [w for w in freq if w not in common]                 # kho distractor

    out = {}
    for it in test_items:
        pos = sorted({w for w in it["text"].split() if w not in common})
        pool = [w for w in rare if w not in set(pos)]
        distr = rng.sample(pool, max(0, N - len(pos)))
        lst = pos + distr
        rng.shuffle(lst)
        out[it["id"]] = lst
    return out

json.dump(build_bias_lists(train_texts, test_items, N=1000),
          open("bias_per_utt.json", "w", encoding="utf-8"), ensure_ascii=False)
```

**Bốn quyết định phải ghi rõ trong luận văn** (chương 8 §8.9): đơn vị từ/cụm, ngưỡng rare, cách lấy distractor (ngẫu nhiên hay khó), dạng chuẩn của từ mượn.

---

## 9.4 Tính metric

```bash
python kb-contextual-asr/tools/bwer.py \
    --ref ref.tsv --hyp hyp.tsv --bias-json bias_per_utt.json
```

Đầu ra (đã kiểm chứng khớp với ví dụ tính tay ở chương 8 §8.2):

```
WER          83.33
B-WER       100.00
U-WER        80.00
N            6
N_biased     1
N_unbiased   5
S 1   D 0   I 4   I_biased 0
```

Chạy đúng script này cho **mọi** cấu hình, đừng dùng jiwer cho cái này và script khác cho cái kia — căn chỉnh khác nhau sẽ làm số không so được (chương 8 §8.1, bẫy 2).

---

## 9.5 Lịch tuần đầu (đề xuất, 1 GPU)

| Ngày | Việc | Đầu ra kiểm chứng được |
|---|---|---|
| 1 | Chuẩn bị dữ liệu: chọn tập test, chuẩn hoá NFC, viết `ref.tsv`. Chạy `probe()` tokenizer (chương 4 §4.6) | bảng fertility của 20 thuật ngữ |
| 2 | Baseline 0 + tính WER. Dựng bias list N=100/500/1000/2000 | 1 con số WER + 4 file bias |
| 3 | Baseline 1 (prompt), quét $N$ | **biểu đồ WER theo $N$** — tự tái hiện hiện tượng sụp đổ |
| 4 | Baseline 2 (CTC + hotword), quét $\alpha$ | **đường cong B-WER ↔ U-WER** |
| 5 | Thí nghiệm chẩn đoán: no-context / **oracle context** / adversarial | 3 con số → biết có "context-utilization gap" không |
| 6 | Thí nghiệm chẩn đoán phát âm: bias list dạng chính tả vs dạng đọc Việt hoá | chênh lệch B-WER giữa hai dạng |
| 7 | Viết lại thành 1 trang: bảng số + 2 biểu đồ + 5 nhận xét | tài liệu để trao đổi với thầy |

Ngày 5 và 6 mới là phần có giá trị khoa học — hai ngày đó có thể quyết định hướng đề tài. Đừng dồn hết thời gian vào việc chạy cho ra model tốt.

---

## 9.6 Checklist debug: "biasing chạy mà số không đổi"

Theo thứ tự, dừng ở chỗ đầu tiên sai:

1. **Bias list có thật sự tới được decoder không?** In ra bên trong hàm decode, đừng tin cấu hình.
2. **Tokenizer khớp không?** In token của một cụm bias và token của chính cụm đó trong reference. Phải giống hệt. (Chương 4 §4.4 — nguyên nhân số 1.)
3. **Chuẩn hoá khớp không?** NFC? chữ thường? dấu câu? Kiểm cả 3 nguồn: ref, hyp, bias list.
4. **Trọng số có đủ lớn không?** Thử $\alpha$ cực lớn (ví dụ 50). Nếu **vẫn** không đổi → cơ chế không được gọi. Nếu transcript loạn hết → cơ chế hoạt động, chỉ là chỉnh sai mức.
5. **Model có sinh ứng viên gần đúng không?** In n-best. Nếu không giả thuyết nào gần cụm đích → vấn đề nằm ở **tầng âm học/phát âm**, không phải ở biasing (nguyên nhân gốc 2, chương 6 §6.2).
6. **B-WER có được tính trên đúng tập từ không?** In `N_biased`. Nếu bằng 0 → bias list không khớp reference (thường do BOM/NFC/chữ hoa).

> Kinh nghiệm: hơn nửa số ca "biasing không có tác dụng" dừng ở bước 2 hoặc bước 6. Cả hai đều **không báo lỗi**.

---

## 9.7 Ghi chép thí nghiệm — làm ngay từ ngày 1

Mỗi lần chạy, lưu một dòng JSON: `{model, checkpoint, tập test, hàm chuẩn hoá, N, cách lấy distractor, α, beam, seed, WER, B-WER, U-WER, entity-P/R, RTF}`.

Lý do rất thực tế: đến lúc viết luận văn, câu hỏi "con số 4.2 này chạy với cấu hình nào?" mà không trả lời được thì phải chạy lại tất cả. Một file `runs.jsonl` tốn 10 phút để thiết lập.

---

## Bài tập tự kiểm tra

1. Chạy Baseline 1 với $N$ = 10, 100, 500, 1000 trên 200 câu. Vẽ WER theo $N$. Điểm gãy nằm ở đâu với model tiếng Việt của bạn?
2. Với Baseline 2, tìm $\alpha$ nhỏ nhất sao cho U-WER tăng không quá 0.5 điểm tuyệt đối. B-WER tại đó là bao nhiêu?
3. Chạy oracle context (bias list chỉ gồm thực thể đúng). So với no-context. Khoảng cách đó là **trần trên** của bạn — nó có đủ lớn để đề tài có ý nghĩa không?
4. Lấy 20 câu bị sai thực thể, phân loại theo 4 nguyên nhân gốc ở chương 6 §6.2. Nguyên nhân nào chiếm đa số? Đó chính là hướng bạn nên đi.

---

*Chương tiếp → [`10-thuat-ngu-va-doc-them.md`](10-thuat-ngu-va-doc-them.md): bảng tra ký hiệu, ôn toán tối thiểu, từ điển thuật ngữ và danh sách đọc theo thứ tự.*
