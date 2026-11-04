"""Giới tính người nói (F0 trung vị, ngưỡng 150 Hz) × scorer: xác suất từ đúng, AUC đúng/nhiễu, định vị.
Giọng TTS của hou25 = en-US-AriaNeural (nữ). Phía từ của pxattn/pmaxsim là âm vị → không có giọng.
  python gender_eval.py out > gender.md"""
import json, sys
import numpy as np
D = sys.argv[1]
g = {k: ("F" if v[0] > 150 else "M") for k, v in json.load(open(f"{D}/gender.json")).items()}


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_diff(a, b, f, n=1000, seed=0):
    """CI 95% của f(a) − f(b), lấy mẫu lại theo người nói (a, b: dict spk → mảng)."""
    rng = np.random.default_rng(seed)
    ka, kb = list(a), list(b)
    out = []
    for _ in range(n):
        sa = np.concatenate([a[k] for k in rng.choice(ka, len(ka))])
        sb = np.concatenate([b[k] for k in rng.choice(kb, len(kb))])
        out.append(f(sa) - f(sb))
    return np.percentile(out, [2.5, 97.5])


print(f"Người nói: {sum(v == 'F' for v in g.values())} nữ / {sum(v == 'M' for v in g.values())} nam (ước lượng F0)\n")
for s in ["test-clean", "test-other"]:
    R = [json.loads(l) for l in open(f"{D}/{s}_rows.jsonl")]
    print(f"## {s}\n")
    print("| scorer | p TB của từ đúng: nữ / nam [CI 95% của nữ − nam] | AUC đúng/nhiễu: nữ / nam | định vị streak trúng span: nữ / nam (%) |")
    print("| --- | --- | --- | --- |")
    for mn, kn in [("hou25", "S1"), ("pxattn", "att1"), ("pmaxsim", "cos")]:
        G = [r for r in R if (r["model"], r["map"]) == (mn, kn)]
        by = {sx: [r for r in G if g[r["uid"].split("-")[0]] == sx] for sx in "FM"}
        tp = {sx: {} for sx in "FM"}
        for sx in "FM":
            for r in by[sx]:
                if r["true"]:
                    tp[sx].setdefault(r["uid"].split("-")[0], []).append(r["p"])
            tp[sx] = {k: np.array(v) for k, v in tp[sx].items()}
        lo, hi = boot_diff(tp["F"], tp["M"], np.mean)
        mp = {sx: np.mean(np.concatenate(list(tp[sx].values()))) for sx in "FM"}
        au = {sx: auc(np.array([r["p"] for r in by[sx] if r["true"]]), np.array([r["p"] for r in by[sx] if not r["true"]])) for sx in "FM"}
        hit = {sx: 100 * np.mean([r["hit_streak"] for r in by[sx] if r["true"]]) for sx in "FM"}
        print(f"| {mn} | {mp['F']:.3f} / {mp['M']:.3f} [{lo:+.3f}, {hi:+.3f}] | {au['F']:.3f} / {au['M']:.3f} | {hit['F']:.1f} / {hit['M']:.1f} |")
    print()
