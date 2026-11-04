"""Q4 bổ sung: probe âm vị trên chiếu NGẪU NHIÊN của feature Whisper, cùng số chiều với nhánh audio của scorer
(256 = pxattn/pmaxsim, 368 = hou25), 3 seed → ngưỡng nhiễu. Cùng câu (lấy từ <split>_rows.jsonl), cùng nhãn MFA, cùng cách chia người nói."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
PHONES = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH").split()
PID = {p: i for i, p in enumerate(PHONES)}
OUT, AL = Path("/work/pos"), Path("/work/align")


def read_tg(path):
    tiers, name, x0, x1 = {}, None, 0.0, 0.0
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l.startswith("name ="):
            name = l.split("=", 1)[1].strip().strip('"'); tiers[name] = []
        elif l.startswith("xmin ="):
            x0 = float(l.split("=")[1])
        elif l.startswith("xmax ="):
            x1 = float(l.split("=")[1])
        elif l.startswith("text =") and name:
            tiers[name].append((l.split("=", 1)[1].strip().strip('"').upper().rstrip("012"), x0, x1))
    return tiers


def probe(X, y, te, seed=0):
    X = torch.from_numpy(X)
    X = (X - X[~te].mean(0)) / (X[~te].std(0) + 1e-5)
    yt, g = torch.from_numpy(y), torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], 39)
    opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=1e-5)
    tr = torch.from_numpy(np.where(~te)[0])
    for _ in range(15):
        for b in tr[torch.randperm(len(tr), generator=g)].split(2048):
            opt.zero_grad(); F.cross_entropy(lin(X[b]), yt[b]).backward(); opt.step()
    with torch.no_grad():
        return float((lin(X[torch.from_numpy(te)]).argmax(1) == yt[torch.from_numpy(te)]).float().mean())


res = {}
for s in ["test-clean", "test-other"]:
    uids = sorted({json.loads(l)["uid"] for l in open(OUT / f"{s}_rows.jsonl")})
    Xs, ys, spk = [], [], []
    for u in uids:
        fs = np.load(OUT / "feats" / f"{u}.npy").astype(np.float32)
        a, b, _ = u.split("-")
        tg = read_tg(AL / s / a / b / f"{u}.TextGrid")
        lab = np.full(len(fs), -1)
        for p, x0, x1 in tg["phones"]:
            if p in PID:
                lab[int(x0 * 50):int(np.ceil(x1 * 50))] = PID[p]
        k = lab >= 0
        Xs.append(fs[k]); ys.append(lab[k]); spk.append(np.full(k.sum(), int(a)))
    X, y, spk = np.concatenate(Xs), np.concatenate(ys), np.concatenate(spk)
    sp = np.unique(spk)
    test_spk = set(np.random.default_rng(0).permutation(sp)[: len(sp) // 2])   # giống position.py
    te = np.array([v in test_spk for v in spk])
    r = {"whisper_seed": [probe(X, y, te, sd) for sd in range(3)]}
    for d in (256, 368):
        r[f"random{d}"] = []
        for sd in range(3):
            W = torch.randn(1280, d, generator=torch.Generator().manual_seed(100 + sd)).numpy() / 1280 ** 0.5
            r[f"random{d}"].append(probe(X @ W, y, te, sd))
    res[s] = r
    print(s, {k: [round(100 * v, 1) for v in vs] for k, vs in r.items()}, flush=True)
json.dump(res, open(OUT / "probe_rand.json", "w"), indent=1)
