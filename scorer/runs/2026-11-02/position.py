"""Scorer có biết từ hiếm nằm Ở ĐÂU trong câu không, và hai phía của scorer (audio / âm vị) có "nói cùng một thứ tiếng" không.

Câu hỏi:
  Q1 vị trí   — ma trận (đơn vị của từ × frame audio) của từng scorer có đỉnh đúng chỗ từ được nói không?
                pxattn: trọng số cross-attention (âm vị hỏi vào audio), lớp 1 và 2, trung bình các head
                        + cosine thô giữa đầu ra encoder âm vị và audio (2 không gian này KHÔNG được train để so bằng cosine)
                pmaxsim: cosine âm vị × frame (đúng thứ model dùng để chấm: max theo thời gian rồi trung bình)
                hou25:  S1 = cosine frame TTS × frame câu nói (cùng là audio, cùng encoder Whisper) — mốc so sánh
  Q2 thứ tự âm vị — trong một từ, âm vị thứ j có đỉnh ở đúng đoạn âm vị j của audio (theo MFA) không, các đỉnh có đi tăng dần không
  Q3 nhận dạng ngược — mỗi đoạn âm vị của audio (MFA) khớp nhất với hàng âm vị nào của từ; ma trận 39×39 loại âm vị audio × loại âm vị chữ
  Q4 phía audio có mang âm vị không — probe tuyến tính (39 âm vị/frame, chia người nói) trên feature Whisper thô và
                trên nhánh audio của từng scorer, so với chiếu ngẫu nhiên cùng số chiều
  Q5 vị trí có thêm thông tin "có/không" không — AUC từ đúng vs từ nhiễu: xác suất của scorer, độ cao đỉnh

Mốc thời gian thật: căn chỉnh MFA của LibriSpeech (Lugosch et al. 2019, zenodo 2619474) — có cả từ và âm vị ARPAbet.
Chạy trên CPU (máy GPU bận), tập con ngẫu nhiên các câu có từ hiếm.

  python position.py feats --split test-clean --n 250        # encoder Whisper-large-v3 → /work/pos/feats/<uid>.npy
  python position.py run   --split test-clean                 # → /work/pos/<split>_rows.jsonl, _phone.jsonl, _mat.npz, _probe.json
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

SC = Path("/work/code/scorer")
sys.path.insert(0, str(SC))
from common import PHONES, N_PHONE, BiasStore, PhoneStore, libri_path, BENCH  # noqa: E402
from models import build, pad_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["feats", "run"])
ap.add_argument("--split", required=True)
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--n_neg", type=int, default=20, help="số từ nhiễu (lấy từ list N100) mỗi câu")
ap.add_argument("--libri", default="/work/libri/LibriSpeech")
ap.add_argument("--align", default="/work/align")
ap.add_argument("--out", default="/work/pos")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--limit", type=int)
a = ap.parse_args()
torch.set_num_threads(a.threads)
OUT = Path(a.out)
(OUT / "feats").mkdir(parents=True, exist_ok=True)
FPS, TOL = 50, 0.2
PID = {p: i + 1 for i, p in enumerate(PHONES)}


def read_tg(path):
    """TextGrid (Praat, dạng dài) → {tên tier: [(text, xmin, xmax)]}."""
    tiers, name, x0, x1 = {}, None, 0.0, 0.0
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l.startswith("name ="):
            name = l.split("=", 1)[1].strip().strip('"')
            tiers[name] = []
        elif l.startswith("xmin ="):
            x0 = float(l.split("=")[1])
        elif l.startswith("xmax ="):
            x1 = float(l.split("=")[1])
        elif l.startswith("text =") and name:
            tiers[name].append((l.split("=", 1)[1].strip().strip('"').lower(), x0, x1))
    return tiers


def find_tg(uid):
    spk, chap, _ = uid.split("-")
    for p in Path(a.align).glob(f"**/{spk}/{chap}/{uid}.TextGrid"):
        return p
    return None


def utts():
    ref = {}
    for l in open(BENCH / f"protocol/{a.split}/ref.tsv", encoding="utf-8"):
        u, t, r = l.rstrip("\n").split("\t")
        if json.loads(r):
            ref[u] = (t.split(), json.loads(r))
    us = sorted(u for u in ref if find_tg(u))
    random.Random(0).shuffle(us)
    return [(u, *ref[u]) for u in us[:a.n]]


# ---------------------------------------------------------------- stage 1: feature encoder Whisper-large-v3 (CPU, fp32)
if a.stage == "feats":
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3").get_encoder().eval()
    todo = [u for u, _, _ in utts() if not (OUT / "feats" / f"{u}.npy").exists()]
    print(f"{a.split}: encode {len(todo)} câu", flush=True)
    for i, u in enumerate(todo):
        w = sf.read(libri_path(a.libri, u), dtype="float32")[0]
        with torch.no_grad():
            h = enc(fe(w, sampling_rate=16000, return_tensors="pt").input_features).last_hidden_state[0]
        np.save(OUT / "feats" / f"{u}.npy", h[:max(1, min(1500, -(-len(w) * FPS // 16000)))].numpy().astype(np.float16))
        if i % 20 == 0:
            print(i, u, flush=True)
    sys.exit()


# ---------------------------------------------------------------- stage 2: bản đồ của từng scorer
# Bản torch của image NVIDIA (2.14 alpha) trên CPU: kernel SDPA đa luồng cho logit cross-attention SAI và đổi theo số luồng
# (cùng 1 câu: 3.55 với 1–4 luồng = tính tay, 3.36 với 8, 3.39 với 32). Ép backend MATH thì khớp tính tay ở mọi số luồng.
# Feature Whisper không bị ảnh hưởng (so sdpa/eager, 4/8 luồng: lệch ≤ 0.008 = làm tròn fp16).
from torch.nn.attention import sdpa_kernel, SDPBackend  # noqa: E402
_MATH = sdpa_kernel(SDPBackend.MATH)  # giữ tham chiếu: object bị thu gom thì context tự thoát
_MATH.__enter__()
def load(name):
    ck = torch.load(SC / f"work/ckpt/{name}/best.pt", map_location="cpu")
    m = build(ck["arch"], N_PHONE)
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def maps_pxattn(m, fs, ids, lb):
    """→ logit (K,), {tên: (K, P, Ts)}. Chạy lại từng lớp TransformerDecoderLayer để lấy trọng số cross-attention."""
    K, P = ids.shape
    mask = pad_mask(lb, P)
    s = m.ts(m.ls(fs).unsqueeze(0)).expand(K, -1, -1)
    h = m.pe(ids, lb)
    out = {"cos_raw": F.normalize(h, dim=-1) @ F.normalize(s, dim=-1).transpose(1, 2)}
    for i, L in enumerate(m.xa):
        x = L.norm1(h + L._sa_block(h, None, mask))
        att, w = L.multihead_attn(x, s, s, need_weights=True, average_attn_weights=True)
        x = L.norm2(x + att)
        h = L.norm3(x + L._ff_block(x))
        out[f"att{i + 1}"] = w
    valid = (~mask).unsqueeze(-1).to(h.dtype)
    mean = (h * valid).sum(1) / lb[:, None]
    mx = h.masked_fill(mask.unsqueeze(-1), -1e4).amax(1)
    z = m.out(torch.cat([mean, mx], -1)).squeeze(-1)
    assert torch.allclose(z, m(fs, ids, lb), atol=1e-3), "chạy lại lớp không khớp forward"
    return z, out


@torch.no_grad()
def maps_pmaxsim(m, fs, ids, lb):
    es, eb = m.enc_speech(fs), m.enc_bias(ids, lb)
    return m.score(es, eb, lb), {"cos": eb @ es.T}


@torch.no_grad()
def maps_hou25(m, fs, fb, lb):
    s = m.proj(fs)
    b = m.proj(fb.float())
    S1 = F.normalize(b, dim=-1) @ F.normalize(s, dim=-1).T                # (K, Tb, Ts): hàng = frame TTS
    return m(fs, fb.float(), lb), {"S1": S1}


def one_by_one(fn, m, fs, X, lb):
    """Chấm từng ứng viên, không pad (đơn giản, khỏi lo mask; K ~ 23 nên vẫn nhanh)."""
    zs, maps = [], {}
    for i in range(len(X)):
        z, mp = fn(m, fs, X[i:i + 1, :lb[i]], lb[i:i + 1])
        zs.append(z)
        for k, v in mp.items():
            maps.setdefault(k, []).append(v[0])
    R = int(lb.max())
    pad = lambda v: F.pad(v, (0, 0, 0, R - v.shape[0]))
    return torch.cat(zs), {k: torch.stack([pad(v) for v in vs]) for k, vs in maps.items()}


def speech_repr(name, m, fs):
    """Nhánh audio của scorer, trước khi gặp phía từ (cho probe Q4)."""
    with torch.no_grad():
        if name == "pxattn":
            return m.ts(m.ls(fs).unsqueeze(0))[0]
        if name == "pmaxsim":
            return m.enc_speech(fs)
        return m.proj(fs)


def streak(M, slopes):
    """M (R, Ts): hàng r ứng với frame (t0 + r·slope). → (tâm dự đoán theo frame, độ cao đường chéo tốt nhất)."""
    R, Ts = M.shape
    best = (Ts / 2, -1e9)
    j = np.arange(R)
    for sl in slopes:
        dj = np.rint(j * sl).astype(int)
        span = dj[-1] + 1
        if span >= Ts:
            continue
        t0 = np.arange(Ts - span + 1)
        d = M[j[None, :], t0[:, None] + dj[None, :]].mean(1)
        k = int(d.argmax())
        if d[k] > best[1]:
            best = (k + span / 2, float(d[k]))
    return best


SLOPES = {"phone": (2, 3, 4, 5, 6, 7, 8), "tts": (0.67, 1.0, 1.5)}       # frame audio / đơn vị của từ


def strip(p):
    return p.rstrip("012").upper()


rng = np.random.default_rng(0)
models = {n: load(n) for n in ["pxattn", "pmaxsim", "hou25"]}
pstore, bstore = PhoneStore(SC / "work"), BiasStore(SC / "work")
ctx = json.load(open(BENCH / f"protocol/{a.split}/ctx_N100.json", encoding="utf-8"))
rows, prow = [], []
mat = {}                                                                 # (model, map) → (40, 40) tổng, số đếm
probe_X, probe_y, probe_spk = {k: [] for k in ["whisper", "pxattn", "pmaxsim", "hou25"]}, [], []
ex, exmeta = {}, {}                                                      # ví dụ để vẽ (+ đoạn âm vị MFA của từ)
U = utts()[:a.limit]
for ui, (u, ref, rare) in enumerate(U):
    fp = OUT / "feats" / f"{u}.npy"
    if not fp.exists():
        continue
    fs = torch.from_numpy(np.load(fp)).float()
    Ts = len(fs)
    tg = read_tg(find_tg(u))
    W = [(w, x0, x1) for w, x0, x1 in tg["words"] if w]
    PH = [(strip(p), x0, x1) for p, x0, x1 in tg["phones"] if p and strip(p) in PID]
    occ = {}
    for w, x0, x1 in W:
        occ.setdefault(w, []).append((x0, x1))
    # nhãn âm vị từng frame (cho probe)
    lab = np.zeros(Ts, int)
    for p, x0, x1 in PH:
        lab[int(x0 * FPS):int(np.ceil(x1 * FPS))] = PID[p]
    keep = lab > 0
    probe_y.append(lab[keep])
    probe_spk.append(np.full(keep.sum(), int(u.split("-")[0])))
    probe_X["whisper"].append(fs[keep].numpy().astype(np.float16))
    for n, m in models.items():
        probe_X[n].append(speech_repr(n, m, fs)[keep].numpy().astype(np.float16))

    true = [w for w in dict.fromkeys(rare) if w in occ and w in pstore and w in bstore]
    negs = [w for w in ctx[u] if w not in ref and w in pstore and w in bstore]
    negs = list(rng.choice(negs, min(a.n_neg, len(negs)), replace=False))
    cands = true + negs
    if not true:
        continue
    ids, lb = pstore.get(cands)
    fb, lbb = bstore.get(cands)
    res = {"pxattn": one_by_one(maps_pxattn, models["pxattn"], fs, ids, lb),
           "pmaxsim": one_by_one(maps_pmaxsim, models["pmaxsim"], fs, ids, lb),
           "hou25": one_by_one(maps_hou25, models["hou25"], fs, fb, lbb)}
    dur = Ts / FPS
    rand_t = rng.uniform(0, dur, len(cands))
    for mn, (z, mp) in res.items():
        p = torch.sigmoid(z).numpy()
        unit = "tts" if mn == "hou25" else "phone"
        R = (lbb if unit == "tts" else lb).numpy()
        for kn, M in mp.items():
            M = M.numpy()
            for ci, w in enumerate(cands):
                Mi = M[ci, :R[ci], :Ts]
                prof = Mi.mean(0)
                t_prof = (prof.argmax() + 0.5) / FPS
                t_med = (np.median(Mi.argmax(1)) + 0.5) / FPS
                c, pk = streak(Mi, SLOPES[unit])
                r = {"split": a.split, "uid": u, "w": w, "true": ci < len(true), "model": mn, "map": kn,
                     "p": float(p[ci]), "peak_prof": float(prof.max()), "peak_streak": pk, "R": int(R[ci]), "dur": dur}
                if ci < len(true):
                    spans = occ[w]
                    cen = np.array([(x0 + x1) / 2 for x0, x1 in spans])
                    for est, t in (("prof", t_prof), ("med", t_med), ("streak", (c + 0.5) / FPS),
                                   ("centre", dur / 2), ("random", rand_t[ci])):
                        r[f"err_{est}"] = float(np.abs(cen - t).min())
                        r[f"hit_{est}"] = bool(any(x0 - TOL <= t <= x1 + TOL for x0, x1 in spans))
                    x0, x1 = spans[0]
                    f0, f1 = int(x0 * FPS), min(Ts, int(np.ceil(x1 * FPS)))
                    inside = np.zeros(Ts, bool)
                    inside[f0:f1] = True
                    r["mass_in"] = float(Mi[:, inside].sum(1).mean() / max(Mi.sum(1).mean(), 1e-9)) if kn.startswith("att") else None
                    r["chance_in"] = float(inside.mean())
                    r["mean_in"], r["mean_out"] = float(Mi[:, inside].mean()), float(Mi[:, ~inside].mean())
                    r["n_occ"] = len(spans)
                    # Q2/Q3: chỉ với scorer âm vị và lần xuất hiện đầu, khi âm vị MFA của từ trùng âm vị g2p
                    if unit == "phone":
                        seg = [(q, y0, y1) for q, y0, y1 in PH if y0 >= x0 - 1e-3 and y1 <= x1 + 1e-3]
                        g2p = [PHONES[i - 1] for i in ids[ci, :R[ci]].tolist()]
                        same = [q for q, _, _ in seg] == g2p
                        r["g2p_match"] = same
                        if same:
                            pk_f = Mi.argmax(1)                                   # đỉnh từng âm vị
                            sf_ = [(int(y0 * FPS), max(int(y0 * FPS) + 1, int(np.ceil(y1 * FPS)))) for _, y0, y1 in seg]
                            seg_of = lambda t: next((k for k, (s0, s1) in enumerate(sf_) if s0 <= t < s1), -1)
                            for j in range(len(g2p)):
                                s0, s1 = sf_[j]
                                kk = seg_of(int(pk_f[j]))
                                prow.append({"model": mn, "map": kn, "uid": u, "w": w, "j": j, "P": len(g2p), "ph": g2p[j],
                                             "own": bool(s0 - 1 <= pk_f[j] < s1 + 1), "in_word": bool(f0 - 10 <= pk_f[j] < f1 + 10),
                                             "idx_err": abs(kk - j) if kk >= 0 else None, "peak_f": int(pk_f[j]),
                                             "seg_dur": (s1 - s0) / FPS, "dur": dur})
                            # Q3: đoạn âm vị audio k → hàng âm vị của từ có giá trị trung bình cao nhất
                            segmean = np.stack([Mi[:, s0:s1].mean(1) for s0, s1 in sf_])  # (seg, P)
                            r["seg_id_acc"] = float(np.mean(segmean.argmax(1) == np.arange(len(g2p))))
                            r["seg_id_chance"] = 1 / len(g2p)
                            if len(g2p) >= 3:
                                r["spearman"] = float(np.corrcoef(np.argsort(np.argsort(pk_f)), np.arange(len(g2p)))[0, 1])
                                r["monotone"] = bool(np.all(np.diff(pk_f) >= 0))
                            key = f"{mn}/{kn}"
                            S, Cn = mat.setdefault(key, (np.zeros((N_PHONE, N_PHONE)), np.zeros((N_PHONE, N_PHONE))))
                            for k, (q, _, _) in enumerate(seg):
                                for jj, g in enumerate(g2p):
                                    S[PID[q], PID[g]] += segmean[k, jj]
                                    Cn[PID[q], PID[g]] += 1
                    if ci == 0 and ui < 8:
                        ex[f"{u}|{w}|{mn}|{kn}|{f0}|{f1}"] = Mi.astype(np.float16)
                        exmeta[f"{u}|{w}"] = [(q, round(y0 * FPS), round(y1 * FPS)) for q, y0, y1 in PH
                                              if y0 >= x0 - 1e-3 and y1 <= x1 + 1e-3]
                rows.append(r)
    if ui % 10 == 0:
        print(f"{ui}/{len(U)} {u}", flush=True)

with open(OUT / f"{a.split}_rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(OUT / f"{a.split}_phone.jsonl", "w") as f:
    for r in prow:
        f.write(json.dumps(r) + "\n")
np.savez(OUT / f"{a.split}_mat.npz", **{k.replace("/", "__") + "__sum": v[0] for k, v in mat.items()},
         **{k.replace("/", "__") + "__cnt": v[1] for k, v in mat.items()})
np.savez(OUT / f"{a.split}_ex.npz", **{k: v for k, v in ex.items()})
json.dump(exmeta, open(OUT / f"{a.split}_exmeta.json", "w"))

# ---------------------------------------------------------------- Q4 probe tuyến tính 39 âm vị / frame, chia người nói 50/50
y = np.concatenate(probe_y) - 1
spk = np.concatenate(probe_spk)
sp = np.unique(spk)
test_spk = set(np.random.default_rng(0).permutation(sp)[: len(sp) // 2])
te = np.array([s in test_spk for s in spk])
probe = {}
feats = dict(probe_X)
Xw = np.concatenate(feats["whisper"]).astype(np.float32)
g = torch.Generator().manual_seed(0)
feats["random256"] = [Xw @ (torch.randn(1280, 256, generator=g).numpy() / 1280 ** 0.5)]
for name, X in feats.items():
    X = torch.from_numpy(np.concatenate(X).astype(np.float32))
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-5
    X = (X - mu) / sd
    yt = torch.from_numpy(y)
    lin = torch.nn.Linear(X.shape[1], len(PHONES))
    opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=1e-5)
    tr_idx = torch.from_numpy(np.where(~te)[0])
    for ep in range(15):
        for b in tr_idx[torch.randperm(len(tr_idx), generator=g)].split(2048):
            opt.zero_grad()
            F.cross_entropy(lin(X[b]), yt[b]).backward()
            opt.step()
    with torch.no_grad():
        pr = lin(X[torch.from_numpy(te)]).argmax(1)
    acc = float((pr == yt[torch.from_numpy(te)]).float().mean())
    probe[name] = {"dim": X.shape[1], "acc": acc, "n_train": int((~te).sum()), "n_test": int(te.sum())}
    print("probe", name, probe[name], flush=True)
probe["majority"] = float(np.mean(y[te] == np.bincount(y[~te]).argmax()))
json.dump(probe, open(OUT / f"{a.split}_probe.json", "w"), indent=1)
print("DONE", a.split)
_MATH.__exit__(None, None, None)
