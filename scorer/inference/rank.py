"""Bước 5 — chấm list bias của test, xuất context cho ASR.

  # 1 tầng (Hou25): chấm toàn bộ list N
  python -m inference.rank --ckpt work/ckpt/hou25/best.pt --name hou25 --split test-clean --libri /data/LibriSpeech
  # cascade: maxsim lọc mỗi list còn 200 từ → hou25 xếp hạng lại; "all" = cả kho 209k, không cần list
  python -m inference.rank --pre work/ckpt/maxsim/best.pt --pre_k 200 --ckpt work/ckpt/hou25/best.pt \\
      --name cascade --lists N100 N2000 all ...

Đầu ra:
  work/out/<name>/<split>/rank_<L>.json        {uid: list xếp GIẢM dần} → bench/eval_retrieval.py
  bench/protocol/<split>/ctx_<name>_<L>.json   top-k (=50) xếp TĂNG dần, từ điểm cao đứng CUỐI (Hou25 §4.1.2, Bảng 3)
  work/out/<name>/<split>/time.json            thời gian chấm trung bình mỗi câu (ms), tách tầng 1 / tầng 2
  work/out/<name>/<split>/prob_<L>.json        {uid: [[từ, xác suất], ...]} top-k theo tầng cuối
  bench/protocol/<split>/ctx_<name>_<L>_p<thr>.json   lọc theo độ liên quan: chỉ giữ từ có xác suất ≥ thr
                                               (tối đa k) → list có thể RỖNG khi không từ nào đủ tin cậy
"""
import argparse, json, math, time

import soundfile as sf
import torch

from common import BENCH, IS21, N_PHONE, WORK, Encoder, libri_path, make_store
from models import build, hard_logit, present_logit, score_chunks

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--pre", help="ckpt model rẻ (maxsim) làm tầng lọc")
ap.add_argument("--pre_k", type=int, default=200)
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="test-clean")
ap.add_argument("--libri", required=True)
ap.add_argument("--lists", nargs="+", default=["N100", "N500", "N1000", "N2000"])
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--keep", type=int, default=1000, help="số từ lưu trong rank_all.json")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--limit", type=int, help="chỉ chấm n câu đầu (chạy thử)")
ap.add_argument("--thr", type=float, nargs="*", default=[0.5], help="ngưỡng xác suất cho list lọc theo độ liên quan")
ap.add_argument("--target", default="present", choices=["present", "hard"],
                help="hxattn: xếp hạng / lọc theo p(có trong câu) hay p(có VÀ Qwen chép sai)")
ap.add_argument("--hyp", help="bản chép lần 1 (mặc định bench/out/<split>/qwen3asr_none.tsv), cho hxattn")
a = ap.parse_args()


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = build(ck["arch"], N_PHONE)
    m.load_state_dict(ck["state"])
    return m.cuda().eval()


class Bank:
    """Chỉ cho model rẻ: phía bias không phụ thuộc câu → encode cả kho 1 lần, giữ trên GPU."""

    @torch.no_grad()
    def __init__(self, model, store, words, chunk=4096):
        self.pos = {w: i for i, w in enumerate(words)}
        E, L = [], []
        for i in range(0, len(words), chunk):
            fb, lb = store.get(words[i:i + chunk])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model.enc_bias(fb.cuda(), lb.cuda()).half()
            E += [e[j, :n] for j, n in enumerate(lb.tolist())]
            L += lb.tolist()
        self.E = torch.cat(E)
        self.len = torch.tensor(L, device="cuda")
        self.start = torch.cumsum(self.len, 0) - self.len

    @torch.no_grad()
    def score(self, model, fs, words, chunk=16384):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            es = model.enc_speech(fs).half()
        ids = torch.tensor([self.pos[w] for w in words], device="cuda")
        out = []
        for i in range(0, len(ids), chunk):
            k = ids[i:i + chunk]
            lb = self.len[k]
            idx = (self.start[k, None] + torch.arange(int(lb.max()), device="cuda")).clamp(max=len(self.E) - 1)
            out.append(model.score(es, self.E[idx], lb).float())
        return torch.cat(out)


def scorer(model):
    store = make_store(model)
    bank = Bank(model, store, sorted(store.idx)) if model.cheap else None

    def f(fs, words, hyp_words):
        if bank:
            return bank.score(model, fs, words).tolist()
        fb, lb = store.get(words)
        hyp = store.seq(hyp_words).cuda() if getattr(model, "uses_hyp", False) else None
        z = score_chunks(model, fs, fb.cuda(), lb.cuda(), hyp=hyp)
        if z.dim() == 2:                                                  # hxattn: 3 lớp → 1 logit theo --target
            z = hard_logit(z) if a.target == "hard" else present_logit(z)
        return z.tolist()
    return f, store


def timed(f, *x):
    torch.cuda.synchronize()
    t = time.time()
    r = f(*x)
    torch.cuda.synchronize()
    return r, time.time() - t


if __name__ == "__main__":
    proto = BENCH / "protocol" / a.split
    ctx = {L: (None if L == "all" else json.load(open(proto / f"ctx_{L}.json", encoding="utf-8"))) for L in a.lists}
    uids = list(json.load(open(proto / "ctx_none.json", encoding="utf-8")))[:a.limit]
    hyp_file = a.hyp or BENCH / "out" / a.split / "qwen3asr_none.tsv"
    hyps = {l.split("\t")[0]: l.rstrip("\n").partition("\t")[2].split() for l in open(hyp_file, encoding="utf-8")}

    main = load(a.ckpt)
    pre = load(a.pre) if a.pre else None
    if "all" in a.lists and not (pre or main).cheap:
        raise SystemExit("list 'all' (209k từ) cần tầng đầu là model rẻ: dùng --pre maxsim hoặc --ckpt maxsim")
    f_main, st_main = scorer(main)
    f_pre, st_pre = scorer(pre) if pre else (None, st_main)
    store = {w for w in st_main.idx if w in st_pre}                     # từ chấm được ở cả 2 tầng
    rare = [w for w in open(IS21 / "all_rare_words.txt", encoding="utf-8").read().split() if w in store]
    enc = Encoder()

    rank = {L: {} for L in a.lists}
    prob = {L: {} for L in a.lists}
    t1 = t2 = 0.0
    miss = 0
    for i in range(0, len(uids), a.bs):
        bu = uids[i:i + a.bs]
        for uid, fs in zip(bu, enc([sf.read(libri_path(a.libri, u), dtype="float32")[0] for u in bu])):
            lists = {L: (rare if L == "all" else [w for w in ctx[L][uid] if w in store]) for L in a.lists}
            miss += sum(len(ctx[L][uid]) - len(lists[L]) for L in a.lists if L != "all")
            tail = {L: [] for L in a.lists}
            if f_pre:  # tầng 1: giữ pre_k từ điểm cao nhất của mỗi list, phần còn lại xếp sau theo điểm tầng 1
                u = sorted(set().union(*lists.values()))
                s, dt = timed(f_pre, fs, u, hyps[uid])
                t1 += dt
                s1 = dict(zip(u, s))
                for L, ws in lists.items():
                    ws = sorted(ws, key=s1.get, reverse=True)
                    lists[L], tail[L] = ws[:a.pre_k], ws[a.pre_k:]
            u = sorted(set().union(*lists.values()))
            s, dt = timed(f_main, fs, u, hyps[uid])
            t2 += dt
            s2 = dict(zip(u, s))
            for L, ws in lists.items():
                rank[L][uid] = (sorted(ws, key=s2.get, reverse=True) + tail[L])[:a.keep]
                prob[L][uid] = [[w, round(1 / (1 + math.exp(-s2[w])), 4)] for w in rank[L][uid][:a.k] if w in s2]
        print(f"{i + len(bu)}/{len(uids)}  tầng1 {1000 * t1 / (i + len(bu)):.0f}ms/câu  "
              f"tầng2 {1000 * t2 / (i + len(bu)):.0f}ms/câu", flush=True)

    out = WORK / "out" / a.name / a.split
    out.mkdir(parents=True, exist_ok=True)
    for L, r in rank.items():
        json.dump(r, open(out / f"rank_{L}.json", "w", encoding="utf-8"))
        json.dump({u: ws[:a.k][::-1] for u, ws in r.items()},
                  open(proto / f"ctx_{a.name}_{L}.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(prob[L], open(out / f"prob_{L}.json", "w", encoding="utf-8"))
        for thr in a.thr:
            keep = {u: [w for w, p in ps if p >= thr][::-1] for u, ps in prob[L].items()}
            json.dump(keep, open(proto / f"ctx_{a.name}_{L}_p{thr:g}.json", "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{L} p≥{thr:g}: trung bình {sum(map(len, keep.values())) / len(keep):.1f} từ/câu, "
                  f"{sum(not v for v in keep.values())} câu rỗng")
    json.dump({"pre_ms": 1000 * t1 / len(uids), "main_ms": 1000 * t2 / len(uids), "missing_feat": miss},
              open(out / "time.json", "w"))
    print(f"xong {a.split}: {len(uids)} câu, {miss} lượt từ thiếu feature (TTS lỗi)")
