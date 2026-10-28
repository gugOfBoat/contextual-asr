"""Tải LibriSpeech train-clean-360 / train-other-500 từ mirror HuggingFace (openslr/librispeech_asr, parquet)
rồi ghi lại đúng cấu trúc gốc <split>/<spk>/<chap>/<uid>.flac + <spk>-<chap>.trans.txt.
openslr.org chỉ ~0.2 MB/s từ máy GPU, HF ~15 MB/s. Byte audio trong parquet là file flac gốc → không giải mã lại.

  python -m data.fetch_libri_hf --libri /work/libri/LibriSpeech --splits train-clean-360 train-other-500
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--libri", required=True)
ap.add_argument("--splits", nargs="+", default=["train-clean-360", "train-other-500"])
ap.add_argument("--cache", default="/work/hf_libri")
a = ap.parse_args()

for split in a.splits:
    sub = "all/" + split.replace("-", ".", 1).replace("-", ".")           # train-clean-360 → all/train.clean.360
    root = snapshot_download("openslr/librispeech_asr", repo_type="dataset", allow_patterns=[f"{sub}/*.parquet"],
                             local_dir=a.cache, max_workers=8)
    trans = defaultdict(list)
    n = 0
    for f in sorted((Path(root) / sub).glob("*.parquet")):
        for row in pq.read_table(f, columns=["id", "text", "audio"]).to_pylist():
            spk, chap, _ = row["id"].split("-")
            d = Path(a.libri) / split / spk / chap
            d.mkdir(parents=True, exist_ok=True)
            audio = row["audio"]["bytes"]
            assert audio[:4] == b"fLaC", row["id"]
            (d / f"{row['id']}.flac").write_bytes(audio)
            trans[d / f"{spk}-{chap}.trans.txt"].append(f"{row['id']} {row['text'].upper()}")
            n += 1
        f.unlink()                                                        # xoá parquet ngay để đỡ tốn đĩa
    for p, lines in trans.items():
        p.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"{split}: {n} câu", flush=True)
