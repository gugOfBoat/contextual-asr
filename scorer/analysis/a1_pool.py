"""A1: cùng checkpoint Hou25, chỉ đổi cách gộp bản đồ CNN thành 1 điểm (PLAN.md P1, câu hỏi Q2).

Hou25 gộp bằng trung bình rồi mới qua lớp linear. Vì linear sau trung bình = trung bình của linear từng ô,
ta tính được bản đồ điểm từng ô cam (Ts × Tb) (class activation map) rồi đọc nó theo nhiều cách:
  mean  trung bình cả bản đồ                = đúng như Hou25 (để kiểm tra tái hiện)
  max   ô cao nhất
  diag  vệt chéo dốc 1 bắt đầu ở frame t: trung bình cam[t + j, j], lấy t tốt nhất → cũng ra vị trí
  box   cửa sổ rộng bằng độ dài từ (mọi cột): trung bình cam[t:t + Tb, :], lấy t tốt nhất → ra vị trí
Không train lại.

  cd /work/code/scorer && python analysis/a1_pool.py --split test-other --libri /work/libri/LibriSpeech

Ra work/out/a1/<split>/:
  rank_<v>_N2000.json  {uid: [[từ, điểm], ...]} giảm dần, giữ --keep từ đầu
  loc_N2000.json       {uid: {từ: [frame đầu diag, frame đầu box, số frame TTS]}} cho top --keep của diag/box + từ đúng
"""
import argparse, json, sys, time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, WORK, BiasStore, Encoder, libri_path  # noqa: E402
from models import build, pad_mask  # noqa: E402

V = ["mean", "max", "diag", "box"]
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(WORK / "ckpt/hou25/best.pt"))
ap.add_argument("--split", default="test-other")
ap.add_argument("--libri", required=True)
ap.add_argument("--list", default="N2000")
ap.add_argument("--keep", type=int, default=200)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
m = build(ck["arch"])
m.load_state_dict(ck["state"])
m = m.cuda().eval()
w_out, b_out = m.out.weight[0], m.out.bias[0]


@torch.no_grad()
def maps(fs, fb, lb):
    """Như Hou25.forward nhưng trả về điểm theo 4 cách gộp (K, 4) và frame đầu của vệt (K, 2)."""
    K, Tb, _ = fb.shape
    mask = pad_mask(lb, Tb)
    s = m.proj(fs).unsqueeze(0).expand(K, -1, -1)
    b = m.proj(fb)
    sb, _ = m.ca_sb(s, b, b, key_padding_mask=mask, need_weights=False)
    bs, _ = m.ca_bs(b, s, s, need_weights=False)
    cos = lambda x, y: torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(y, dim=-1).transpose(-1, -2)
    S = torch.stack([cos(s, b), cos(s, bs), cos(sb, b), cos(sb, bs)], 1)
    valid = (~mask)[:, None, None, :].to(S.dtype)
    h = m.cnn(S * valid) * valid                                            # (K, 256, Ts, Tb)
    cam = torch.einsum("kcij,c->kij", h, w_out.to(h.dtype)).float()         # (K, Ts, Tb), = 0 ở cột pad
    Ts = cam.shape[1]
    lbf = lb.float()
    mean = cam.sum((1, 2)) / (lbf * Ts)
    mx = cam.masked_fill(mask[:, None, :], -1e9).amax((1, 2))
    diag = torch.zeros(K, Ts, device=cam.device)                            # diag[t] = Σ_j cam[t + j, j]
    for j in range(min(Tb, Ts)):
        diag[:, :Ts - j] += cam[:, j:, j]
    diag = diag / lbf[:, None]
    col = cam.sum(2)                                                        # (K, Ts): tổng mọi cột ở mỗi frame
    cs = torch.nn.functional.pad(col.cumsum(1), (1, 0))
    end = (torch.arange(Ts, device=cam.device)[None] + lb[:, None]).clamp(max=Ts)
    box = (cs.gather(1, end) - cs[:, :Ts]) / (lbf[:, None] ** 2)            # trung bình trên lb frame × lb cột
    d, td = diag.max(1)
    x, tx = box.max(1)
    score = torch.stack([mean, mx, d, x], 1) + b_out.float()
    return score, torch.stack([td, tx], 1)


proto = BENCH / "protocol" / a.split
ctx = json.load(open(proto / f"ctx_{a.list}.json", encoding="utf-8"))
orc = json.load(open(proto / "ctx_oracle.json", encoding="utf-8"))
uids = list(json.load(open(proto / "ctx_none.json", encoding="utf-8")))[:a.limit]
store = BiasStore()
enc = Encoder()
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, t = maps(fs, fb.cuda(), lb.cuda())
            sc.append(s.float().cpu())
            tt.append(t.cpu())
            L += lb.tolist()
        sc, tt = torch.cat(sc), torch.cat(tt)
        for j, v in enumerate(V):
            o = sc[:, j].argsort(descending=True)[:a.keep].tolist()
            rank[v][uid] = [[words[q], round(sc[q, j].item(), 3)] for q in o]
        want = {w for v in ("diag", "box") for w, _ in rank[v][uid]} | set(orc.get(uid, []))
        loc[uid] = {w: [tt[q, 0].item(), tt[q, 1].item(), L[q]] for q, w in enumerate(words) if w in want}
    print(f"{i + len(bu)}/{len(uids)} {(time.time() - t0) / (i + len(bu)):.2f}s/câu", flush=True)

out = WORK / "out" / "a1" / a.split
out.mkdir(parents=True, exist_ok=True)
for v in V:
    json.dump(rank[v], open(out / f"rank_{v}_{a.list}.json", "w", encoding="utf-8"))
json.dump(loc, open(out / f"loc_{a.list}.json", "w", encoding="utf-8"))
print("xong", out)
