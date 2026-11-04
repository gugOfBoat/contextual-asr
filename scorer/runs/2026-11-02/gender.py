"""Ước lượng giới tính người nói bằng F0 trung vị (không có SPEAKERS.TXT trên máy): >165 Hz → nữ.
→ /work/pos/gender.json {spk: [f0, "F"/"M"]} cho các người nói trong <split>_rows.jsonl."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, "/work/code/scorer")
from common import libri_path


def f0_track(w, sr=16000, win=640, hop=160):
    """F0 theo tự tương quan, chỉ frame đủ năng lượng và tuần hoàn rõ."""
    out = []
    lo, hi = sr // 400, sr // 70
    e_thr = 0.1 * np.sqrt(np.mean(w ** 2)) * win ** 0.5
    for i in range(0, len(w) - win, hop):
        x = w[i:i + win] - w[i:i + win].mean()
        if np.linalg.norm(x) < e_thr:
            continue
        ac = np.correlate(x, x, "full")[win - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5 * ac[0]:
            out.append(sr / k)
    return np.array(out)

spk_utts = defaultdict(list)
for s in ["test-clean", "test-other"]:
    for u in sorted({json.loads(l)["uid"] for l in open(f"/work/pos/{s}_rows.jsonl")}):
        spk_utts[u.split("-")[0]].append(u)
out = {}
for spk, us in spk_utts.items():
    f0 = []
    for u in us[:5]:
        f0.append(f0_track(sf.read(libri_path("/work/libri/LibriSpeech", u), dtype="float32")[0]))
    m = float(np.median(np.concatenate(f0)))
    out[spk] = [round(m, 1), "F" if m > 165 else "M"]
json.dump(out, open("/work/pos/gender.json", "w"))
print(len(out), sorted(v[0] for v in out.values()))
