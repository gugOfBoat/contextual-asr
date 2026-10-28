"""Chấm WER / U-WER / B-WER bằng scorer IS21 chính thức, có tuỳ chọn chuẩn hoá Whisper.

  python score.py --ref protocol/test-clean/ref.tsv --hyp out/test-clean/whisper_none_sub100.raw.tsv --whisper-norm
  python score.py --ref ... --hyp out/test-clean/qwen3asr_pxattn_N2000_p0.5.tsv --whisper-norm \
      --ctx protocol/test-clean/ctx_pxattn_N2000_p0.5.json        # thêm phân rã lỗi từ hiếm

--whisper-norm: áp EnglishTextNormalizer của Whisper cho CẢ ref, hyp và từ hiếm
(honour->honor, saint/st, số...). Không bật thì giữ nguyên ref và chỉ dùng textnorm.norm cho hyp.
Chỉ chấm các câu có trong hyp (cho phép chạy trên tập con).
--ctx: list đã đưa cho ASR → mỗi lượt từ hiếm của ref được xếp vào 1 trong 3 ô [C]:
  đúng | sai, CÓ trong list (ASR không dùng được) | sai, KHÔNG có trong list (bước chọn từ bỏ sót)
"""
import argparse, json
from pathlib import Path

from score_is21_official import Code, EditDistance, WordError
from textnorm import norm

KEYS = ["WER", "U-WER", "B-WER"]


def normalizer(whisper_norm):
    if not whisper_norm:
        return lambda s: s
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    spell = json.load(open(Path(__file__).parent / "data" / "whisper_normalizer.json", encoding="utf-8"))
    return EnglishTextNormalizer(spell)


def utt_errors(ref_path, hyp_path, whisper_norm):
    """{uid: {"WER"/"U-WER"/"B-WER": [lỗi, số từ ref], "bias": [(từ hiếm của ref, chép đúng?)]}}"""
    f = normalizer(whisper_norm)
    hyps = {}
    for line in open(hyp_path, encoding="utf-8"):
        uid, _, text = line.rstrip("\n").partition("\t")
        hyps[uid] = f(norm(text)) if whisper_norm else norm(text)
    out = {}
    for line in open(ref_path, encoding="utf-8"):
        uid, text, rare = line.rstrip("\n").split("\t")
        if uid not in hyps:
            continue
        ref = f(text).split()
        bias = {w for r in json.loads(rare) for w in f(r).split()}
        hyp = hyps[uid].split()
        if not ref and not hyp:
            continue
        e = {k: [0, 0] for k in KEYS}
        e["bias"] = []
        res = EditDistance().align(ref, hyp)
        for code, ri, hi in zip(res.codes, res.refs, res.hyps):
            if code == Code.insertion:
                e["WER"][0] += 1
                e["B-WER" if hyp[hi] in bias else "U-WER"][0] += 1
                continue
            k = "B-WER" if ref[ri] in bias else "U-WER"
            for kk in ("WER", k):
                e[kk][1] += 1
                e[kk][0] += code != Code.match
            if k == "B-WER":
                e["bias"].append((ref[ri], code == Code.match))
        out[uid] = e
    return out


def rate(errs, key, uids=None):
    uids = errs if uids is None else uids
    n = sum(errs[u][key][1] for u in uids)
    return 100 * sum(errs[u][key][0] for u in uids) / n if n else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--whisper-norm", action="store_true")
    ap.add_argument("--ctx")
    a = ap.parse_args()

    errs = utt_errors(a.ref, a.hyp, a.whisper_norm)
    n_bias = sum(e["B-WER"][1] for e in errs.values())
    print(f"{len(errs)} câu | " + " | ".join(f"{k} {rate(errs, k):.2f}" for k in KEYS) + f" | N_bias {n_bias}")
    multi = [l.split("\t")[0] for l in open(a.ref, encoding="utf-8") if len(json.loads(l.rstrip("\n").split("\t")[2])) > 1]
    multi = [u for u in multi if u in errs]  # lát ablation [C]: câu có > 1 từ hiếm
    print(f"  câu >1 từ hiếm: {len(multi)} câu | " + " | ".join(f"{k} {rate(errs, k, multi):.2f}" for k in KEYS))

    if a.ctx:
        f = normalizer(a.whisper_norm)
        ctx = json.load(open(a.ctx, encoding="utf-8"))
        c = {"đúng": 0, "sai, có trong list (ASR)": 0, "sai, không có trong list (chọn từ)": 0, "  (đúng mà không có trong list)": 0}
        for uid, e in errs.items():
            given = {w for x in ctx.get(uid, []) for w in f(x).split()}
            for w, ok in e["bias"]:
                c["đúng" if ok else ("sai, có trong list (ASR)" if w in given else "sai, không có trong list (chọn từ)")] += 1
                c["  (đúng mà không có trong list)"] += ok and w not in given
        print("Phân rã lượt từ hiếm: " + " | ".join(f"{k} {v} ({100 * v / n_bias:.2f}%)" for k, v in c.items()))
