"""Whisper greedy decode có boost bias word, gate theo vị trí (thí nghiệm trần C3, GOAL.md mục 6).

Mỗi bước decode biết decoder "đang nghe" frame nào: cộng attention của các alignment head
(generation_config.alignment_heads), lọc trung vị rộng 7, lấy argmax (Simul-Whisper, arXiv 2406.10052 §2.1).
Một bias word chỉ được cộng điểm khi thời điểm đó nằm trong cửa sổ của nó.

  python gate_boost.py --ctx oracle --boost 0                    # không prompt, không boost
  python gate_boost.py --ctx oracle --prompt                     # chỉ prompt (cách Hou25)
  python gate_boost.py --ctx oracle --boost 3 --gate none        # boost mọi lúc
  python gate_boost.py --ctx oracle --boost 3 --gate span --margin 0.1
  python gate_boost.py --ctx oracle --boost 3 --gate center --margin 0.3

Vị trí: --spans = align.tsv (vị trí hoàn hảo từ align_words.py); sau này thay bằng vị trí do scorer dự đoán.
Ra: out/<split>/whisper_<ctx><tag>.tsv (+ .raw.tsv, + .gate.tsv ghi mỗi lần boost), chấm bằng score.py.
"""
import argparse, json, os, time, zlib
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from textnorm import norm

ROOT = Path(__file__).parent
FR = 0.02  # encoder Whisper: 20 ms / frame

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test-other")
ap.add_argument("--ctx", default="oracle")
ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
ap.add_argument("--subset")
ap.add_argument("--limit", type=int)
ap.add_argument("--prompt", action="store_true", help="đưa list vào prompt, nối bằng ', ' như Hou25")
ap.add_argument("--boost", type=float, default=0.0, help="cộng vào log-prob token của bias word; 0 = tắt")
ap.add_argument("--gate", default="span", choices=["none", "span", "center"])
ap.add_argument("--margin", type=float, default=0.1, help="nới cửa sổ mỗi bên (s); với center là nửa bề rộng")
ap.add_argument("--spans", help="tsv: uid, i, từ, bắt đầu, kết thúc (mặc định protocol/<split>/align.tsv)")
ap.add_argument("--once", action="store_true",
                help="B3: chỉ boost token ĐẦU của từ khi số lần từ đã được viết < số lần nó có trong câu (số span; gate none = 1)")
ap.add_argument("--times", action="store_true", help="ghi giây decoder đang nghe cho mỗi token ra .time.tsv (để sửa sau theo span)")
ap.add_argument("--boost_ctx", help="list riêng cho boost (mặc định = --ctx): vd prompt top-50 + boost chỉ vài từ chắc chắn")
ap.add_argument("--topk", type=int, default=0,
                help="chỉ boost token ĐẦU của từ khi nó đã nằm trong top-k của model ở bước đó (0 = tắt): tránh đè lên từ thường")
ap.add_argument("--fallback", action="store_true", help="câu bị lặp chữ thì decode lại không context (đếm số câu)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

proto = ROOT / "protocol" / a.split
audio = [l.rstrip("\n").split("\t") for l in open(proto / "audio.tsv", encoding="utf-8")]
if a.subset:
    keep = set(open(a.subset).read().split())
    audio = [x for x in audio if x[0] in keep]
audio = audio[:a.limit] if a.limit else audio
ctx = json.load(open(proto / f"ctx_{a.ctx}.json", encoding="utf-8"))
bctx = json.load(open(proto / f"ctx_{a.boost_ctx}.json", encoding="utf-8")) if a.boost_ctx else ctx

spans = defaultdict(list)  # (uid, từ) -> các khoảng (bắt đầu, kết thúc); từ nói 2 lần thì có 2 khoảng
if a.boost and a.gate != "none":
    for l in open(a.spans or proto / "align.tsv", encoding="utf-8"):
        uid, _, w, b, e = l.rstrip("\n").split("\t")[:5]
        b, e = float(b), float(e)
        if a.gate == "center":
            b = e = (b + e) / 2
        spans[uid, w].append((b - a.margin, e + a.margin))

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32
proc = WhisperProcessor.from_pretrained(a.model)
tok = proc.tokenizer
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=dtype, attn_implementation="eager").to(dev).eval()
heads = model.generation_config.alignment_heads
eot = tok.convert_tokens_to_ids("<|endoftext|>")
start = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
ban = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=dev)
ban[eot + 1:] = True  # mọi token đặc biệt trừ <|endoftext|>
ban[list(model.generation_config.suppress_tokens or [])] = True

if not a.tag:
    a.tag = ("_prompt" if a.prompt else "") + (f"_b{a.boost:g}_{a.gate}" + (f"{a.margin:g}" if a.gate != "none" else "") if a.boost else "")
    a.tag += "_once" if a.once and a.boost else ""
    a.tag = a.tag or "_plain"
out = ROOT / "out" / a.split
out.mkdir(parents=True, exist_ok=True)
fn = out / f"whisper_{a.ctx}{a.tag}.tsv"
done = {l.split("\t")[0] for l in open(fn, encoding="utf-8")} if fn.exists() else set()


def variants(w):  # các cách viết token hoá của một bias word: " cabinet", " Cabinet"
    return {tuple(tok.encode(" " + v, add_special_tokens=False)) for v in (w, w.capitalize())}


t0, nfb = time.time(), 0
with open(fn, "a", encoding="utf-8", buffering=1) as f, \
     open(out / f"whisper_{a.ctx}{a.tag}.raw.tsv", "a", encoding="utf-8", buffering=1) as fr, \
     open(out / f"whisper_{a.ctx}{a.tag}.gate.tsv", "a", encoding="utf-8", buffering=1) as fg, \
     open(out / f"whisper_{a.ctx}{a.tag}.time.tsv" if a.times else os.devnull, "a", encoding="utf-8", buffering=1) as ft:
    for uid, path in audio:
        if uid in done:
            continue
        wav, sr = sf.read(path, dtype="float32")
        n = min(1500, int(len(wav) / sr / FR) + 1)  # số frame có audio thật
        feats = proc(wav, sampling_rate=sr, return_tensors="pt").input_features.to(dev, dtype)
        words = ctx[uid]
        for attempt in (0, 1):  # --fallback: chữ lặp (tỉ lệ nén > 2.4, như openai-whisper) → decode lại KHÔNG context
            use = attempt == 0
            prefix = list(start)
            if use and a.prompt and words:  # giống openai-whisper: chỉ giữ 223 token cuối của prompt
                p = proc.get_prompt_ids(", ".join(words))
                prefix = [int(p[0]), *map(int, p[1:][-223:])] + prefix
            bw = bctx[uid]
            seqs = [(w, s) for w in bw for s in variants(w)] if a.boost and use else []
            cap = {w: (len(spans[uid, w]) if a.gate != "none" else 1) for w in bw}  # B3: số lần được viết
            wrote = defaultdict(int)
            gen, pkv, inp = [], None, torch.tensor([prefix], device=dev)
            with torch.no_grad():
                enc = model.model.encoder(feats).last_hidden_state
                for _ in range(224):
                    o = model.model.decoder(input_ids=inp, encoder_hidden_states=enc, past_key_values=pkv,
                                            use_cache=True, output_attentions=True)
                    pkv = o.past_key_values
                    lp = F.log_softmax(model.proj_out(o.last_hidden_state[0, -1]).float(), -1).masked_fill(ban, -1e9)
                    if seqs or a.times:
                        att = sum(o.cross_attentions[l][0, h, -1, :n].float() for l, h in heads)
                        att = F.pad(att[None, None], (3, 3), mode="replicate").unfold(-1, 7, 1).median(-1).values[0, 0]
                        t = att.argmax().item() * FR  # decoder đang nghe giây thứ t
                    if seqs:
                        bonus = {}
                        for w, s in seqs:
                            if a.gate != "none" and not any(b <= t <= e for b, e in spans[uid, w]):
                                continue
                            for k in range(len(s)):  # k = số token của từ đã sinh ngay trước đó
                                if k == 0 and a.once and wrote[w] >= cap[w]:
                                    continue
                                if k == 0 or gen[-k:] == list(s[:k]):
                                    bonus[s[k]] = w
                        if a.topk and bonus:
                            top = set(lp.topk(a.topk).indices.tolist())
                            first = {s_[0] for _, s_ in seqs}
                            bonus = {v: w for v, w in bonus.items() if v in top or (v not in first) or (gen and any(gen[-k:] == list(s_[:k]) and s_[k] == v for _, s_ in seqs for k in range(1, len(s_))))}
                        for v, w in bonus.items():
                            lp[v] += a.boost
                    nxt = int(lp.argmax())
                    if seqs and nxt in bonus:
                        fg.write(f"{uid}\t{len(gen)}\t{t:.2f}\t{bonus[nxt]}\t{tok.decode([nxt])}\n")
                    if nxt == eot:
                        break
                    if a.times:
                        ft.write(f"{uid}\t{len(gen)}\t{t:.2f}\t{tok.decode([nxt])}\n")
                    gen.append(nxt)
                    for w, s in seqs:  # đếm từ đã viết xong (kể cả khi model tự viết, không nhờ boost)
                        if gen[-len(s):] == list(s):
                            wrote[w] += 1
                    inp = torch.tensor([[nxt]], device=dev)
            text = tok.decode(gen, skip_special_tokens=True)
            if not a.fallback or attempt == 1 or len(text.encode()) <= 2.4 * len(zlib.compress(text.encode())):
                break
            nfb += 1
        f.write(f"{uid}\t{norm(text)}\n")
        fr.write(f"{uid}\t{text.strip()}\n")
print(f"xong {len(audio)} câu, {nfb} câu decode lại vì lặp, {time.time() - t0:.0f}s -> {fn}")
