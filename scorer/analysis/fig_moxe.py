"""Hình minh hoạ cho MO-XE-RECALL1.md (số GIẢ, chỉ để giải thích cách chấm; chạy CPU).
  python scorer/analysis/fig_moxe.py  → scorer/analysis/imgs_moxe/*.png
Cách tính MaxSim / MonoSim giống hệt models.py (MaxSim.score, MonoSim._mono), gap thu nhỏ = 3 cho bảng nhỏ."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "imgs_moxe"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(3)
T, G = 24, 3


def maxsim(M):
    return M.max(1).mean(), M.argmax(1)


def mono(M):
    """D[p,t] = M[p,t] + max_{1<=g<=G} D[p-1,t-g]; điểm = max_t D[P-1,t] / P; lần ngược ra đường đi."""
    P, T_ = M.shape
    D, B = M.copy(), np.zeros_like(M, dtype=int)
    for p in range(1, P):
        for t in range(T_):
            c = [(D[p - 1, t - g], t - g) for g in range(1, G + 1) if t - g >= 0]
            v, j = max(c) if c else (-1e4, 0)
            D[p, t], B[p, t] = M[p, t] + v, j
    t = int(D[-1].argmax())
    path = [t]
    for p in range(P - 1, 0, -1):
        t = B[p, t]
        path.append(t)
    return D[-1].max() / P, path[::-1]


def table(ph, bright):
    M = rng.uniform(0.0, 0.3, (len(ph), T))
    for p, t, v in bright:
        M[p, t] = v
    return M


# từ đúng "bartley" nói ở frame 6–16; từ nhiễu "ghetto": vài ô sáng rải rác, SAI thứ tự
good_ph = "B AA R T L IY".split()
good = table(good_ph, [(0, 6, .9), (1, 8, .85), (2, 10, .8), (3, 12, .9), (4, 14, .85), (5, 16, .9),
                       (1, 7, .6), (3, 13, .6), (5, 17, .55)])
bad_ph = "G EH T OW".split()
bad = table(bad_ph, [(0, 20, .8), (1, 3, .75), (2, 12, .9), (3, 7, .7)])


def draw(ax, M, ph, title, mark_max=True, path=None, allow=None):
    ax.imshow(M, cmap="Greens", vmin=-0.2, vmax=1, aspect="auto")
    if allow is not None:
        for t in range(T):
            if not allow[t]:
                ax.add_patch(plt.Rectangle((t - .5, -.5), 1, len(ph), color="#bbbbbb", alpha=.85, lw=0))
    ax.set_yticks(range(len(ph)), ph, fontsize=10)
    ax.set_xticks(range(0, T, 2), [str(t) for t in range(0, T, 2)], fontsize=7)
    ax.set_xlabel("frame audio (1 frame = 20 ms)", fontsize=9)
    ax.set_title(title, fontsize=10)
    if mark_max:
        for p, t in enumerate(M.argmax(1)):
            ax.plot(t, p, "o", mfc="none", mec="#d97706", ms=13, mew=2)
    if path is not None:
        ax.plot(path, range(len(ph)), "-s", color="#1c3f6e", ms=6, lw=2)


# Hình 1: MaxSim vs MonoSim trên từ đúng và từ nhiễu
fig, axs = plt.subplots(2, 2, figsize=(13, 6.2))
for row, (M, ph, name) in enumerate([(good, good_ph, "bartley (từ ĐÚNG)"), (bad, bad_ph, "ghetto (từ NHIỄU)")]):
    s1, _ = maxsim(M)
    s2, path = mono(M)
    draw(axs[row, 0], M, ph, f"{name} — khớp TỰ DO (MaxSim): mỗi hàng lấy ô sáng nhất  →  {s1:.2f}")
    draw(axs[row, 1], M, ph, f"{name} — khớp ĐÚNG THỨ TỰ (MonoSim), bước 1–{G} frame  →  {s2:.2f}",
         mark_max=False, path=path)
fig.suptitle("Bảng giống nhau âm vị × frame (số giả). Vòng cam = ô MaxSim chọn; đường xanh = đường MonoSim lần ngược", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "1_maxsim_vs_mono.png", dpi=130)

# Hình 2: dạy vị trí — từ đúng chỉ được khớp trong span MFA ± 0.1 s
allow = np.zeros(T, bool)
allow[5:18] = True                        # MFA nói "bartley" ở frame 6–16 (± 5 frame = 0.1 s, thu nhỏ còn ±1 cho hình)
fake = good.copy()
fake[3, 21] = .95                         # một ô sáng ở chỗ khác (vd chữ "t" của từ sau)
fig, axs = plt.subplots(1, 2, figsize=(13, 2.9))
draw(axs[0], fake, good_ph, "Không dạy vị trí: âm vị T được lấy ô sáng ở frame 21 (sai chỗ) vẫn được điểm")
masked = np.where(allow[None], fake, -1.0)
draw(axs[1], masked, good_ph, "Dạy vị trí: ngoài span MFA ± 0.1 s gán −1 (xám) → chỉ khớp được đúng chỗ", allow=allow)
fig.tight_layout()
fig.savefig(OUT / "2_day_vi_tri.png", dpi=130)

# Hình 3: Hou25 — gộp trung bình vs vệt chéo trên bản đồ điểm sau CNN
Ts, Tb = 40, 8
cam = rng.normal(0, .25, (Ts, Tb))
for j in range(Tb):
    cam[14 + j, j] += 1.6                 # vệt chéo = từ đúng nói ở frame 14–21
for t in (3, 9, 27, 33, 36):              # rải rác vài ô hơi sáng (nhiễu)
    cam[t, rng.integers(Tb)] += 1.0
d = np.array([sum(cam[t + j, j] for j in range(Tb)) / Tb for t in range(Ts - Tb + 1)])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.3), gridspec_kw={"width_ratios": [2.2, 1, 1.6]})
axs[0].imshow(cam.T, cmap="Greens", aspect="auto")
axs[0].set_xlabel("frame câu nói"); axs[0].set_ylabel("frame audio TTS")
axs[0].set_title("Bản đồ điểm từng ô sau CNN (Hou25)", fontsize=10)
axs[0].plot(range(14, 22), range(Tb), color="#1c3f6e", lw=2)
axs[1].bar([0], [cam.mean()], color="#cbd5e1")
axs[1].set_xticks([0], ["trung bình\ncả bản đồ"])
axs[1].set_title(f"Hou25 gốc: {cam.mean():.2f}\n(vệt bị pha loãng bởi {Ts * Tb - Tb} ô nền)", fontsize=10)
axs[1].set_ylim(0, 1.8)
axs[2].plot(d, color="#13a548")
axs[2].set_xlabel("frame bắt đầu vệt t0")
axs[2].set_title(f"Vệt chéo: max theo t0 = {d.max():.2f} tại t0 = {d.argmax()}\n→ vừa ra điểm, vừa ra vị trí", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "3_hou25_mean_vs_diag.png", dpi=130)
print("ok", sorted(p.name for p in OUT.iterdir()))
print("maxsim good/bad", maxsim(good)[0], maxsim(bad)[0], "mono good/bad", mono(good)[0], mono(bad)[0])
