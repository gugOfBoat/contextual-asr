"""Hình mổ xẻ scorer âm vị cho slide 20/11 (SỐ GIẢ, chỉ minh hoạ cách chấm; số đo thật ghi trên slide).
  python make_moxe.py   → moxe_*.pdf cùng thư mục
Cách tính giống models.py: MaxSim = trung bình max từng hàng; MonoSim = quy hoạch động, bước 1..G khung.
Bảng giống scorer/analysis/fig_moxe.py (cùng seed) để hình khớp tài liệu MO-XE-RECALL1.md."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans"})
rng = np.random.default_rng(3)
T, G = 24, 3
ORG, PATH = "#d97706", "#111B88"


def maxsim(M):
    return M.max(1).mean(), M.argmax(1)


def mono(M):
    P, _ = M.shape
    Dm, B = M.copy(), np.zeros_like(M, dtype=int)
    for p in range(1, P):
        for t in range(T):
            c = [(Dm[p - 1, t - g], t - g) for g in range(1, G + 1) if t - g >= 0]
            v, j = max(c) if c else (-1e4, 0)
            Dm[p, t], B[p, t] = M[p, t] + v, j
    t = int(Dm[-1].argmax())
    path = [t]
    for p in range(P - 1, 0, -1):
        t = B[p, t]
        path.append(t)
    return Dm[-1].max() / P, path[::-1]


def table(ph, bright):
    M = rng.uniform(0.0, 0.3, (len(ph), T))
    for p, t, v in bright:
        M[p, t] = v
    return M


good_ph = "B AA R T L IY".split()
good = table(good_ph, [(0, 6, .9), (1, 8, .85), (2, 10, .8), (3, 12, .9), (4, 14, .85), (5, 16, .9),
                       (1, 7, .6), (3, 13, .6), (5, 17, .55)])
bad_ph = "G EH T OW".split()
bad = table(bad_ph, [(0, 20, .8), (1, 3, .75), (2, 12, .9), (3, 7, .7)])


def draw(ax, M, ph, title, circles=True, path=None, grey=None, xlabel=True):
    ax.imshow(M, cmap="Greens", vmin=-0.2, vmax=1, aspect="auto")
    if grey is not None:
        for t in range(T):
            if grey[t]:
                ax.add_patch(plt.Rectangle((t - .5, -.5), 1, len(ph), color="#bbbbbb", alpha=.9, lw=0))
    ax.set_yticks(range(len(ph)), ph, fontsize=11)
    ax.set_xticks(range(0, T, 2), [str(t) for t in range(0, T, 2)], fontsize=8)
    if xlabel:
        ax.set_xlabel("khung audio (1 khung = 20 ms)", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    if circles:
        for p, t in enumerate(M.argmax(1)):
            ax.plot(t, p, "o", mfc="none", mec=ORG, ms=13, mew=2.2)
    if path is not None:
        ax.plot(path, range(len(ph)), "-s", color=PATH, ms=6, lw=2.2)


# 1. Tự do và đúng thứ tự trên từ đúng / từ nhiễu
fig, axs = plt.subplots(2, 2, figsize=(12.5, 5.6))
for r, (M, ph, nm) in enumerate([(good, good_ph, "bartley (ĐÚNG)"), (bad, bad_ph, "ghetto (NHIỄU)")]):
    s1, _ = maxsim(M)
    s2, path = mono(M)
    draw(axs[r, 0], M, ph, f"{nm}, khớp tự do = {s1:.2f}", xlabel=r == 1)
    draw(axs[r, 1], M, ph, f"{nm}, khớp đúng thứ tự = {s2:.2f}", circles=False, path=path, xlabel=r == 1)
fig.tight_layout()
fig.savefig(D / "moxe_tudo_thutu.pdf")
plt.close(fig)
print("tự do đúng/nhiễu", round(maxsim(good)[0], 2), round(maxsim(bad)[0], 2),
      "thứ tự đúng/nhiễu", round(mono(good)[0], 2), round(mono(bad)[0], 2), "đường", mono(good)[1])

# 2. Dạy vị trí: ô sáng ngoài đoạn MFA bị gán −1
fake = good.copy()
fake[3, 21] = .95
out = np.ones(T, bool)
out[5:18] = False
fig, axs = plt.subplots(1, 2, figsize=(12.5, 2.7))
draw(axs[0], fake, good_ph, "Không dạy vị trí: T lấy ô sáng ở khung 21 (chữ t của từ sau) vẫn ăn điểm")
draw(axs[1], np.where(out[None], -1.0, fake), good_ph, "Dạy vị trí: ngoài đoạn MFA ±0.1 s gán −1 (xám)", grey=out)
fig.tight_layout()
fig.savefig(D / "moxe_day_vitri.pdf")
plt.close(fig)

# 3. Vị trí: đường đi → khung → giây → so với MFA
_, path = mono(good)
b, e = path[0], path[-1] + 1
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4.4), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
draw(a0, good, good_ph, "Đường đi của nhánh đúng thứ tự (lần ngược từ ô cuối)", circles=False, path=path, xlabel=False)
for t in (b, e):
    a0.axvline(t - .5, color=PATH, ls="--", lw=1.2)
a1.set_ylim(0, 2)
a1.set_yticks([0.5, 1.5], ["scorer", "MFA"], fontsize=10)
mb, me = 5, 18          # MFA: 0.10–0.36 s
a1.barh(1.5, me - mb, left=mb - .5, height=.6, color="#cbd5e1")
a1.barh(0.5, e - b, left=b - .5, height=.6, color="#86c49a")
for y, (x0, x1) in [(1.5, (mb, me)), (0.5, (b, e))]:
    c = (x0 + x1) / 2 - .5
    a1.plot([c, c], [y - .3, y + .3], color="black", lw=2)
    a1.text(x1 - .3, y, f"  {x0 * .02:.2f}–{x1 * .02:.2f} s, tâm {(x0 + x1) / 2 * .02:.2f} s", va="center", fontsize=10)
a1.set_xlim(-.5, T - .5)
a1.set_xticks(range(0, T, 2), [f"{t}\n{t * .02:.2f}s" for t in range(0, T, 2)], fontsize=7.5)
a1.set_xlabel("khung / giây", fontsize=9)
for s in ("top", "right"):
    a1.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(D / "moxe_vitri.pdf")
plt.close(fig)
print("đoạn scorer", b, e, "khung")
