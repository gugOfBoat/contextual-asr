"""Hình cho slide gặp thầy 20/11. Số lấy từ scorer/analysis/REPORT-1115.md (các bảng đã đo).
  python make_figs.py   → *.pdf cùng thư mục
"""
import json, math, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent
R = D.parents[2]
PRI, ACC, GRAY, RED, GREEN = "#111B88", "#F2A900", "#8A94A6", "#A23B3B", "#26734D"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

# 1. Sạch@K (test-other, N2000) — REPORT mục 3, 5, 7c
sysn = ["Hou25\n(tái hiện)", "Hou25\nđọc vệt chéo", "pmaxsim\n(âm vị)", "Hai nhánh\n+ dạy vị trí"]
vals = {1: [72.6, 82.9, 87.5, 89.3], 5: [87.1, 95.3, 96.2, 97.5], 10: [90.9, 96.7, 97.4, 98.6]}
cols = [GRAY, "#5B6BC0", PRI, ACC]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
w = 0.2
for i, (k, v) in enumerate(vals.items()):
    for j, x in enumerate(v):
        b = ax.bar(i + (j - 1.5) * w, x, w * 0.92, color=cols[j], label=sysn[j].replace("\n", " ") if i == 0 else None)
        ax.text(i + (j - 1.5) * w, x + 0.6, f"{x:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["Sạch@1", "Sạch@5", "Sạch@10"])
ax.set_ylim(65, 102); ax.set_ylabel("% từ đúng")
ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); fig.savefig(D / "sach_k.pdf"); plt.close(fig)

# 2. Báo nhầm theo độ dài câu — REPORT mục 3, 5 (eval_e §2)
bk = ["< 5 s\n(1522 câu)", "5–10 s\n(963)", "10–20 s\n(414)", "≥ 20 s\n(40)"]
fa = {"Hou25 (tái hiện)": ([1.4, 2.5, 6.1, 12.4], GRAY), "Hou25 đọc vệt chéo": ([0.9, 0.9, 1.1, 1.9], "#5B6BC0"),
      "pmaxsim": ([0.3, 0.6, 1.0, 1.3], PRI), "Hai nhánh + dạy vị trí": ([0.3, 0.5, 0.7, 0.9], ACC)}
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for n, (v, c) in fa.items():
    ax.plot(range(4), v, "o-", color=c, lw=2.2, label=n)
    ax.text(3.08, v[-1], f"{v[-1]}", color=c, va="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(bk, fontsize=9)
ax.set_ylabel("từ nhiễu có p ≥ 0.5 / câu"); ax.set_xlim(-0.2, 3.5)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(D / "bao_nham.pdf"); plt.close(fig)

# 2b. Như trên nhưng chỉ Hou25 (cho phần 1, khi chưa giới thiệu các scorer khác)
v = fa["Hou25 (tái hiện)"][0]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.bar(range(4), v, 0.6, color=GRAY)
for i, x in enumerate(v):
    ax.text(i, x + 0.25, f"{x}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(bk, fontsize=9)
ax.set_ylabel("từ nhiễu có p ≥ 0.5 / câu"); ax.set_ylim(0, 14.5)
ax.set_title("Hou25 (tái hiện): câu càng dài, càng nhiều từ nhiễu bị chấm cao", fontsize=10)
fig.tight_layout(); fig.savefig(D / "bao_nham_hou25.pdf"); plt.close(fig)

# 3. Tranh nhau tại chỗ: 2 câu thật (pdual2_dl, test-other, p ≥ 0.3)
O = R / "scorer/runs/2026-11-14/out/pdual2_dl/test-other"
prob = json.load(open(O / "prob_N2000.json", encoding="utf-8"))
span = json.load(open(O / "span_N2000.json", encoding="utf-8"))
ref = {l.split("\t")[0]: (l.split("\t")[1], set(json.loads(l.rstrip("\n").split("\t")[2])))
       for l in open(R / "bench/protocol/test-other/ref.tsv", encoding="utf-8")}


def nms(order, sp):
    keep = []
    for w in order:
        b, e = sp[w]
        if any(min(e, sp[x][1]) - max(b, sp[x][0]) > 0
               and (min(e, sp[x][1]) - max(b, sp[x][0])) / (max(e, sp[x][1]) - min(b, sp[x][0])) > 0.3 for x in keep):
            continue
        keep.append(w)
    return keep


ex = [("7018-75789-0016", "Câu A: tranh nhau loại hết nhiễu"), ("4198-61336-0012", "Câu B: nhiễu nghe giống một từ thường trong câu")]
fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6))
for ax, (u, title) in zip(axes, ex):
    text, gold = ref[u]
    ws = [(w, p) for w, p in prob[u] if p >= 0.3 and w in span[u]]
    kept = set(nms([w for w, _ in ws], span[u]))
    for i, (w, p) in enumerate(ws):
        b, e = span[u][w]
        g, k = w in gold, w in kept
        c = GREEN if g and k else (RED if k else "#C9CED8")
        ax.barh(i, e - b, left=b, height=0.7, color=c, edgecolor="black" if k else "none", lw=0.6)
        ax.text(e + 0.06, i, f"{w}  p={p:.2f}" + ("" if k else "  (bị loại)"), va="center", fontsize=9.5,
                color="black" if k else "#6B7280", fontweight="bold" if k else "normal")
    ax.invert_yaxis(); ax.set_yticks([])
    xm = max(span[u][w][1] for w, _ in ws)
    ax.set_xlim(0, xm + 3.4)
    ax.set_title(title + "\n" + textwrap.fill("“" + text + "”", 82), fontsize=8.5, loc="left")
    ax.set_xlabel("giây", fontsize=8.5)
fig.text(0.01, 0.01, "xanh = từ đúng được giữ · đỏ = nhiễu được giữ · xám = bị loại vì trùng chỗ",
         ha="left", fontsize=8.5, color="#374151")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D / "tranh_nhau.pdf"); plt.close(fig)

# 4. Theo cỡ list (pdual2_dl, test-other; đẩy có vị trí, không prompt) — REPORT mục 7c
N = [100, 500, 1000, 2000]
s1 = [98.2, 94.1, 92.6, 89.0]; prec = [97.8, 89.2, 81.4, 69.9]; bw = [7.14, 7.37, 7.47, 7.65]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.0))
a1.plot(N, s1, "o-", color=ACC, lw=2.2, label="Sạch@1"); a1.plot(N, prec, "s--", color=PRI, lw=2, label="precision (p ≥ 0.3)")
a1.set_xscale("log"); a1.set_xticks(N); a1.set_xticklabels(N); a1.set_ylabel("%"); a1.set_xlabel("số từ trong list")
a1.legend(fontsize=9, frameon=False); a1.set_title("Scorer", fontsize=10)
a2.plot(N, bw, "o-", color=ACC, lw=2.2, label="scorer + đẩy có vị trí")
a2.axhline(9.69, color=GRAY, ls=":", lw=2); a2.text(110, 9.5, "Hou25 prompt, N2000: 9.69", color="#4B5563", fontsize=8.5)
a2.axhline(17.92, color=RED, ls=":", lw=1.5); a2.text(110, 17.4, "không context: 17.92", color=RED, fontsize=8.5)
a2.set_xscale("log"); a2.set_xticks(N); a2.set_xticklabels(N); a2.set_ylim(5, 19); a2.set_ylabel("B-WER (%)"); a2.set_xlabel("số từ trong list")
a2.set_title("Whisper-large-v3-turbo, test-other", fontsize=10)
fig.tight_layout(); fig.savefig(D / "theo_N.pdf"); plt.close(fig)
print("xong")
