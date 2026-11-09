"""A1b: đọc vệt chéo trực tiếp trên S1 (như Hoàng, F14 / Table 9), không qua CNN.

  s1     S1 = cos(proj(speech), proj(TTS)) của checkpoint Hou25 (lớp chiếu đã train)
  s1raw  cos(feature Whisper thô của speech, của TTS), không có tham số train nào
Mỗi từ: diag[t] = trung bình_j S[t + j, j]; điểm = max_t, vị trí = argmax_t (frame đầu).

  cd /work/code/scorer && python analysis/a1_s1.py --split test-other --libri /work/libri/LibriSpeech

Ra work/out/a1/<split>/: rank_s1_N2000.json, rank_s1raw_N2000.json,
  loc_s1_N2000.json {uid: {từ: [frame đầu s1, frame đầu s1raw, số frame TTS]}} (top --keep của mỗi cách + từ đúng)
"""
import argparse, json, sys, time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, WORK, BiasStore, Encoder, libri_path  # noqa: E402
from models import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(WORK / "ckpt/hou25/best.pt"))
ap.add_argument("--split", default="test-other")
ap.add_argument("--libri", required=True)
ap.add_argument("--list", default="N2000")
ap.add_argument("--keep", type=int, default=200)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
m = build(ck["arch"])
m.load_state_dict(ck["state"])
proj = m.proj.cuda().half().eval()


def diag(S, lb):
    """S (K, Ts, Tb), cột pad = 0 → (điểm max, frame đầu argmax)."""
    K, Ts, Tb = S.shape
    d = torch.zeros(K, Ts, device=S.device)
    for j in range(min(Tb, Ts)):
        d[:, :Ts - j] += S[:, j:, j].float()
    d = d / lb[:, None].float()
    return d.max(1)


@torch.no_grad()
def score(fs, fb, lb):
    valid = (torch.arange(fb.shape[1], device=fb.device)[None] < lb[:, None]).half()   # (K, Tb)
    out = []
    for s, b in [(proj(fs), proj(fb)), (fs, fb)]:
        S = F.normalize(s, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)          # (K, Ts, Tb)
        out.append(diag(S * valid[:, None, :], lb))
    return torch.stack([out[0][0], out[1][0]], 1), torch.stack([out[0][1], out[1][1]], 1)


proto = BENCH / "protocol" / a.split
ctx = json.load(open(proto / f"ctx_{a.list}.json", encoding="utf-8"))
orc = json.load(open(proto / "ctx_oracle.json", encoding="utf-8"))
uids = list(json.load(open(proto / "ctx_none.json", encoding="utf-8")))[:a.limit]
mf = WORK / "out" / "a1" / a.split / f"rank_mean_{a.list}.json"   # để có vị trí S1 cho top-50 của Hou25 gốc
top_mean = {u: {w for w, _ in ws[:50]} for u, ws in json.load(open(mf, encoding="utf-8")).items()} if mf.exists() else {}
store = BiasStore()
enc = Encoder()
V = ["s1", "s1raw"]
rank = {v: {} for v in V}
loc = {}
t0 = time.time()
for i in range(0, len(uids), a.bs):
    bu = uids[i:i + a.bs]
    for uid, fs in zip(bu, enc([sf.read(libri_path(a.libri, u), dtype="float32")[0] for u in bu])):
        words = sorted(w for w in set(ctx[uid]) if w in store)
        sc, tt, L = [], [], []
        for k in range(0, len(words), a.chunk):
            fb, lb = store.get(words[k:k + a.chunk])
            s, t = score(fs, fb.cuda(), lb.cuda())
            sc.append(s.cpu())
            tt.append(t.cpu())
            L += lb.tolist()
        sc, tt = torch.cat(sc), torch.cat(tt)
        for j, v in enumerate(V):
            o = sc[:, j].argsort(descending=True)[:a.keep].tolist()
            rank[v][uid] = [[words[q], round(sc[q, j].item(), 4)] for q in o]
        want = {w for v in V for w, _ in rank[v][uid]} | set(orc.get(uid, [])) | top_mean.get(uid, set())
        loc[uid] = {w: [tt[q, 0].item(), tt[q, 1].item(), L[q]] for q, w in enumerate(words) if w in want}
    print(f"{i + len(bu)}/{len(uids)} {(time.time() - t0) / (i + len(bu)):.2f}s/câu", flush=True)

out = WORK / "out" / "a1" / a.split
out.mkdir(parents=True, exist_ok=True)
for v in V:
    json.dump(rank[v], open(out / f"rank_{v}_{a.list}.json", "w", encoding="utf-8"))
json.dump(loc, open(out / f"loc_s1_{a.list}.json", "w", encoding="utf-8"))
print("xong", out)
