"""Gom output của position.py → bảng markdown (stdout) + hình trong imgs/.

  python summarize.py <thư mục chứa *_rows.jsonl ...> <thư mục imgs>
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, IMG = Path(sys.argv[1]), Path(sys.argv[2])
IMG.mkdir(parents=True, exist_ok=True)
SPLITS = ["test-clean", "test-other"]
MAPS = [("hou25", "S1", "hou25 · cosine TTS×audio"), ("pmaxsim", "cos", "pmaxsim · cosine âm vị×audio"),
        ("pxattn", "att1", "pxattn · attention lớp 1"), ("pxattn", "att2", "pxattn · attention lớp 2"),
        ("pxattn", "cos_raw", "pxattn · cosine thô (không train)")]
PHONES = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH").split()


def load(name):
    return {s: [json.loads(l) for l in open(D / f"{s}_{name}.jsonl")] for s in SPLITS if (D / f"{s}_{name}.jsonl").exists()}


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rows, prow = load("rows"), load("phone")
md = lambda *c: print("| " + " | ".join(str(x) for x in c) + " |")
pct = lambda x: f"{100 * np.mean(x):.1f}"

for s, R in rows.items():
    T = [r for r in R if r["true"]]
    n_utt = len({r["uid"] for r in T})
    print(f"\n## {s}: {n_utt} câu, {len({(r['uid'], r['w']) for r in T})} từ hiếm đúng\n")
    print("### Q1. Định vị từ (|err| = khoảng cách tới tâm từ theo MFA; trúng = trong [đầu−0.2s, cuối+0.2s])\n")
    md("bản đồ", "ước lượng", "median \\|err\\| (s)", "trong 0.5 s (%)", "trúng span (%)", "mass/mean trong span", "chance")
    md(*["---"] * 7)
    for mn, kn, lab in MAPS:
        G = [r for r in T if r["model"] == mn and r["map"] == kn]
        if not G:
            continue
        for est in ("prof", "med", "streak"):
            e = np.array([r[f"err_{est}"] for r in G])
            inside = (f"{np.mean([r['mass_in'] for r in G]):.2f}" if kn.startswith("att")
                      else f"{np.mean([r['mean_in'] for r in G]):.3f} vs {np.mean([r['mean_out'] for r in G]):.3f}")
            md(lab, est, f"{np.median(e):.2f}", pct(e <= 0.5), pct([r[f'hit_{est}'] for r in G]),
               inside if est == "prof" else "", f"{np.mean([r['chance_in'] for r in G]):.2f}" if est == "prof" else "")
    G = [r for r in T if r["model"] == "pmaxsim" and r["map"] == "cos"]
    for est, lab in (("centre", "mốc: giữa câu"), ("random", "mốc: ngẫu nhiên")):
        e = np.array([r[f"err_{est}"] for r in G])
        md(lab, "", f"{np.median(e):.2f}", pct(e <= 0.5), pct([r[f'hit_{est}'] for r in G]), "", "")

    print("\n### Q5. Từ đúng vs từ nhiễu (AUC)\n")
    md("scorer", "xác suất của scorer", "đỉnh profile", "đỉnh đường chéo")
    md(*["---"] * 4)
    for mn, kn, lab in MAPS:
        G = [r for r in R if r["model"] == mn and r["map"] == kn]
        if not G:
            continue
        pos = lambda k: np.array([r[k] for r in G if r["true"]])
        neg = lambda k: np.array([r[k] for r in G if not r["true"]])
        md(lab, f"{auc(pos('p'), neg('p')):.3f}", f"{auc(pos('peak_prof'), neg('peak_prof')):.3f}",
           f"{auc(pos('peak_streak'), neg('peak_streak')):.3f}")

    print("\n### Q2–Q3. Bên trong từ: âm vị j có đỉnh ở đúng đoạn âm vị j không (từ có âm vị MFA = g2p)\n")
    P = prow.get(s, [])
    gm = [r["g2p_match"] for r in T if r["model"] == "pmaxsim" and r["map"] == "cos"]
    print(f"Âm vị g2p trùng hoàn toàn âm vị MFA: {pct(gm)}% số từ.\n")
    md("bản đồ", "đỉnh trong đoạn âm vị của chính nó (%)", "đỉnh trong từ ±0.2 s (%)", "lệch chỉ số âm vị (TB)",
       "đơn điệu (%)", "Spearman TB", "đoạn audio → đúng hàng âm vị (%)", "chance (%)")
    md(*["---"] * 8)
    for mn, kn, lab in MAPS[1:]:
        G = [r for r in P if r["model"] == mn and r["map"] == kn]
        W = [r for r in T if r["model"] == mn and r["map"] == kn and "seg_id_acc" in r]
        if not G:
            continue
        ie = [r["idx_err"] for r in G if r["idx_err"] is not None]
        sp = [r["spearman"] for r in W if "spearman" in r and not np.isnan(r["spearman"])]
        md(lab, pct([r["own"] for r in G]), pct([r["in_word"] for r in G]), f"{np.mean(ie):.2f}" if ie else "–",
           pct([r["monotone"] for r in W if "monotone" in r]), f"{np.mean(sp):.2f}",
           pct([r["seg_id_acc"] for r in W]), pct([r["seg_id_chance"] for r in W]))
    if P:
        g = [r for r in P if r["model"] == "pmaxsim"]
        if "dur" in g[0]:
            print(f"\nMốc may rủi cho cột 1: đỉnh rơi ngẫu nhiên trong câu thì trúng đoạn âm vị của chính nó "
                  f"{100 * np.mean([r['seg_dur'] / r['dur'] for r in g]):.1f}% số lần.")
        print(f"\nThời lượng TB một đoạn âm vị (MFA) trong từ hiếm: {np.mean([r['seg_dur'] for r in g]):.3f} s "
              f"= {np.mean([r['seg_dur'] for r in g]) * 50:.1f} frame.")

# ---- Q4 probe
print("\n### Q4. Probe tuyến tính 39 âm vị / frame (chia người nói 50/50, độ chính xác trên người nói chưa thấy)\n")
md("feature", "số chiều", *SPLITS)
md(*["---"] * (2 + len(SPLITS)))
pr = {s: json.load(open(D / f"{s}_probe.json")) for s in SPLITS if (D / f"{s}_probe.json").exists()}
for k in ["whisper", "random256", "hou25", "pxattn", "pmaxsim"]:
    md(k, pr[SPLITS[0]][k]["dim"], *[f"{100 * pr[s][k]['acc']:.1f}" for s in pr])
md("đoán lớp nhiều nhất", "", *[f"{100 * pr[s]['majority']:.1f}" for s in pr])

# ---- Q3 ma trận 39×39 (gộp 2 tập)
fig, axs = plt.subplots(1, 4, figsize=(22, 5.8))
print("\n### Q3b. Ma trận loại âm vị audio (hàng) × loại âm vị của từ (cột), gộp 2 tập\n")
md("bản đồ", "hàng có đường chéo lớn nhất (%)", "hạng TB của đường chéo (1 = tốt nhất)", "số hàng")
md(*["---"] * 4)
for ax, (mn, kn, lab) in zip(axs, MAPS[1:]):
    S = sum(np.load(D / f"{s}_mat.npz")[f"{mn}__{kn}__sum"] for s in pr)[1:, 1:]
    C = sum(np.load(D / f"{s}_mat.npz")[f"{mn}__{kn}__cnt"] for s in pr)[1:, 1:]
    M = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    ok = (np.diag(C) >= 5) & ((C >= 1).sum(1) >= 10)
    Z = (M - np.nanmean(M, 1, keepdims=True)) / (np.nanstd(M, 1, keepdims=True) + 1e-9)
    rank = [1 + np.sum(Z[i][~np.isnan(Z[i])] > Z[i, i]) for i in np.where(ok)[0]]
    best = [np.nanargmax(Z[i]) == i for i in np.where(ok)[0]]
    md(lab, pct(best), f"{np.mean(rank):.1f}", int(ok.sum()))
    ax.imshow(Z, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_title(lab + "\n(z-score theo hàng)", fontsize=10)
    ax.set_xticks(range(39), PHONES, fontsize=5, rotation=90)
    ax.set_yticks(range(39), PHONES, fontsize=5)
    ax.set_xlabel("âm vị của từ (g2p)")
    ax.set_ylabel("âm vị đoạn audio (MFA)")
plt.tight_layout()
plt.savefig(IMG / "phone_matrix.png", dpi=150)

# ---- hình ví dụ: mỗi từ 1 hình, đủ các bản đồ, cắt quanh từ, vạch trắng = ranh giới âm vị MFA
ORDER = [("hou25", "S1"), ("pmaxsim", "cos"), ("pxattn", "att1"), ("pxattn", "att2"), ("pxattn", "cos_raw")]
for s in SPLITS:
    if not (D / f"{s}_exmeta.json").exists():
        continue
    ex, meta = np.load(D / f"{s}_ex.npz"), json.load(open(D / f"{s}_exmeta.json"))
    ok = [(uw, segs) for uw, segs in meta.items()                         # chỉ từ có âm vị MFA = g2p (hàng ↔ đoạn 1-1)
          if ex[next(k for k in ex.keys() if k.startswith(f"{uw}|pmaxsim|cos|"))].shape[0] == len(segs)]
    for n_ex, (uw, segs) in enumerate(ok[:4]):
        u, w = uw.split("|")
        fig, axs = plt.subplots(len(ORDER), 1, figsize=(10, 9))
        for ax, (mn, kn) in zip(axs, ORDER):
            k = next(k for k in ex.keys() if k.startswith(f"{uw}|{mn}|{kn}|"))
            f0, f1 = map(int, k.split("|")[4:6])
            M = ex[k].astype(float)
            a0, a1 = max(0, f0 - 40), min(M.shape[1], f1 + 40)
            ax.imshow(M[:, a0:a1], aspect="auto", cmap="viridis", interpolation="nearest", extent=(a0 - .5, a1 - .5, M.shape[0] - .5, -.5))
            for _, y0, y1 in segs:
                ax.axvline(y0 - .5, color="w", lw=.6, ls="--")
            ax.axvline(segs[-1][2] - .5, color="w", lw=.6, ls="--")
            if mn != "hou25":
                ax.set_yticks(range(len(segs)), [q for q, _, _ in segs], fontsize=7)
                for q, y0, y1 in segs:
                    ax.text((y0 + y1) / 2 - .5, -0.7, q, color="k", fontsize=6, ha="center", va="bottom")
            else:
                ax.set_ylabel("frame TTS", fontsize=7)
            ax.set_title({"S1": "hou25 · cosine (frame TTS × frame câu nói, cùng encoder Whisper)",
                          "cos": "pmaxsim · cosine (âm vị × frame câu nói) — đúng thứ model dùng",
                          "att1": "pxattn · trọng số cross-attention lớp 1 (âm vị hỏi vào audio)",
                          "att2": "pxattn · trọng số cross-attention lớp 2",
                          "cos_raw": "pxattn · cosine thô (embedding âm vị × audio, KHÔNG được train để so)"}[kn], fontsize=8, pad=10)
        axs[-1].set_xlabel("frame câu nói (20 ms); vạch trắng = ranh giới âm vị theo MFA")
        fig.suptitle(f"{s} {u}: từ '{w}'", fontsize=10)
        plt.tight_layout()
        plt.savefig(IMG / f"example_{s}_{n_ex}.png", dpi=140)
        plt.close(fig)
