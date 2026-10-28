"""So 2 output ASR trên cùng tập: ΔB-WER + khoảng tin cậy 95% (paired bootstrap theo câu, 1.000 lần) [B]
và G = % khoảng trống lấp được giữa "cả list" và "oracle" [C].

  python compare.py --ref protocol/test-clean/ref.tsv --whisper-norm \
      --a out/test-clean/qwen3asr_pxattn_N2000_p0.5.tsv --b out/test-clean/qwen3asr_pxattn_N2000_top50.tsv \
      --full out/test-clean/qwen3asr_N2000.tsv --oracle out/test-clean/qwen3asr_oracle.tsv

Δ = A − B; khoảng tin cậy không chứa 0 → khác biệt có ý nghĩa. G = (B_full − B_x) / (B_full − B_oracle).
"""
import argparse, random

from score import rate, utt_errors

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--full")
ap.add_argument("--oracle")
ap.add_argument("--whisper-norm", action="store_true")
ap.add_argument("--key", default="B-WER", choices=["WER", "U-WER", "B-WER"])
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()

E = {k: utt_errors(a.ref, p, a.whisper_norm) for k, p in [("A", a.a), ("B", a.b), ("full", a.full), ("oracle", a.oracle)] if p}
uids = sorted(set.intersection(*(set(e) for e in E.values())))
k = a.key
for name, e in E.items():
    print(f"{name:7s} {k} {rate(e, k, uids):.2f}")

rng = random.Random(0)
d0 = rate(E["A"], k, uids) - rate(E["B"], k, uids)
ds = sorted(rate(E["A"], k, s) - rate(E["B"], k, s) for s in ([rng.choice(uids) for _ in uids] for _ in range(a.n)))
lo, hi = ds[int(0.025 * a.n)], ds[int(0.975 * a.n) - 1]
print(f"Δ{k} (A − B) = {d0:+.2f}  [95% CI {lo:+.2f}, {hi:+.2f}]  " + ("có ý nghĩa" if lo > 0 or hi < 0 else "KHÔNG có ý nghĩa"))

if a.full and a.oracle:
    bf, bo = rate(E["full"], k, uids), rate(E["oracle"], k, uids)
    for name in ("A", "B"):
        print(f"G({name}) = {100 * (bf - rate(E[name], k, uids)) / (bf - bo):.1f}%")
