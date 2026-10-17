# Dữ liệu cho benchmark

## IS21 bias list (Le et al. 2021) — `is21/`
Nguồn: `github.com/facebookresearch/fbai-speech`, thư mục `is21_deep_bias/` (đã kiểm md5 trùng với bản dùng trong các số liệu).

| File | Trong git? | Nguồn |
|---|---|---|
| `all_rare_words.txt` (209.291 từ), `common_words_5k.txt` | ✅ | `is21_deep_bias/words/` |
| `test-{clean,other}.biasing_{100,500,1000,2000}.tsv` (3–67 MB/file) | ❌ tải lại | `is21_deep_bias/ref/` |

```bash
bash bench/data/download_is21.sh
```

## LibriSpeech (audio) — không có trong git
openslr.org/12: `test-clean`, `test-other` (đánh giá), `train-clean-100`, `dev-clean`, `dev-other` (train/dev scorer).
Giải nén sao cho có `<LIBRI>/LibriSpeech/<split>/<spk>/<chap>/<uid>.flac`.

## Protocol — tạo lại từ 2 nguồn trên
```bash
cd bench && python build_is21.py      # → protocol/<split>/{ref.tsv, audio.tsv, ctx_*.json}
```
`build_is21.py` đặt đường dẫn LibriSpeech ở biến `LIBRI` (mặc định `../datasets/original/LibriSpeech`).
`ref.tsv`, `ctx_none.json`, `ctx_oracle.json` có sẵn trong git; các `ctx_N*.json` (tới 66 MB) và list do scorer lọc (`ctx_<model>_*.json`) thì tạo lại.
List do scorer lọc tạo lại bằng `scorer/inference/rank.py` (cần GPU) — xác suất top-50 đã lưu sẵn ở `scorer/runs/2026-10-16/out/`.
