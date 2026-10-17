"""Tái hiện dòng "Whisper-turbo" và "+ prompt" trong Bảng 1 của Hou25 (Interspeech 2025).

  python run_whisper.py --ctx none N100 --subset protocol/test-clean/subset100.txt --tag _sub100

Prompt "naive" theo Hou25 §2.1: các từ nối bằng ", ".
Lưu ý: Whisper giới hạn prompt ~224 token; list dài bị transformers cắt (giữ phần CUỐI).
"""
import argparse, json, time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from textnorm import norm

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-clean")
ap.add_argument("--ctx", nargs="+", default=["none"])
ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
ap.add_argument("--subset")
ap.add_argument("--tag", default="")
ap.add_argument("--fw", action="store_true", help="dùng faster-whisper int8 (CPU, ít RAM) thay cho HF")
a = ap.parse_args()

proto = ROOT / "protocol" / a.split
audio = [l.rstrip("\n").split("\t") for l in open(proto / "audio.tsv", encoding="utf-8")]
if a.subset:
    keep = set(open(a.subset).read().split())
    audio = [x for x in audio if x[0] in keep]
out = ROOT / "out" / a.split
out.mkdir(parents=True, exist_ok=True)

if a.fw:
    from faster_whisper import WhisperModel
    fw = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8")
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()

for c in a.ctx:
    ctx = json.load(open(proto / f"ctx_{c}.json", encoding="utf-8"))
    t0 = time.time()
    fn = out / f"whisper_{c}{a.tag}.tsv"
    done = {l.split("\t")[0] for l in open(fn, encoding="utf-8")} if fn.exists() else set()
    with open(fn, "a", encoding="utf-8", buffering=1) as f, \
         open(out / f"whisper_{c}{a.tag}.raw.tsv", "a", encoding="utf-8", buffering=1) as fr:
        for uid, path in audio:
            if uid in done:  # chạy tiếp nếu lần trước bị dừng giữa chừng
                continue
            wav, sr = sf.read(path, dtype="float32")
            if a.fw:  # greedy, không điều kiện câu trước — gần với generate() mặc định của HF
                segs, _ = fw.transcribe(wav, language="en", beam_size=1, without_timestamps=True,
                                        condition_on_previous_text=False,
                                        initial_prompt=", ".join(ctx[uid]) or None)
                text = " ".join(s.text for s in segs)
                f.write(f"{uid}\t{norm(text)}\n")
                fr.write(f"{uid}\t{text.strip()}\n")
                continue
            feats = proc(wav, sampling_rate=sr, return_tensors="pt").input_features.to(dev, dtype)
            kw = dict(language="en", task="transcribe")
            if ctx[uid]:  # giống openai-whisper: chỉ giữ 223 token CUỐI của prompt
                p = proc.get_prompt_ids(", ".join(ctx[uid]))
                kw["prompt_ids"] = torch.tensor([p[0], *p[1:][-223:]]).to(dev)
            with torch.no_grad():
                ids = model.generate(feats, **kw)[0].tolist()
            sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            if sot in ids:  # bỏ phần prompt trả kèm ở đầu
                ids = ids[ids.index(sot):]
            text = proc.decode(ids, skip_special_tokens=True)
            f.write(f"{uid}\t{norm(text)}\n")
            fr.write(f"{uid}\t{text.strip()}\n")
    print(f"xong {c}: {len(audio)} câu, {time.time() - t0:.0f}s")
