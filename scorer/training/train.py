"""Bước 4 — train scorer.

  python -m training.train --arch hou25            # hou25 | nocross | maxsim | pmaxsim | pxattn | hxattn

Mỗi câu train-clean-100 (Hou25 §3.1, §3.3):
  pos = từ hiếm của câu (tối đa 8)
  neg = 4 × max(1, |pos|): 40% lấy từ near.json (gần theo edit distance), 60% ngẫu nhiên trong kho 209k
        tỉ lệ 1:4 paper không nêu [B]; không lấy từ nào có mặt trong câu
Loss: BCE trên từng cặp (câu, từ) [C — paper ghi "softmax" nhưng không nói softmax trên cái gì].
Dev: 500 câu dev-clean, list = từ hiếm + 100 từ ngẫu nhiên (như IS21 N=100) → Recall@10; lưu best.pt theo dev.
Encoder Whisper đóng băng, chạy trực tiếp trên câu nói (không cache để tiết kiệm đĩa); feature bias đọc từ bias.f16.

hxattn (A + B): nhãn 3 lớp từ work/pass1_<split>.jsonl (data/qwen_pass1.py):
  0 = negative, 1 = từ hiếm Qwen lần 1 chép đúng, 2 = từ hiếm Qwen chép sai ("khó")
  loss = CE 3 lớp (trọng số lớp khó x3 [B]) + loss theo list: softmax trên [ô "không chọn"] + mọi ứng viên,
         xác suất dồn vào các từ có mặt (hoặc vào ô "không chọn" nếu câu không có từ hiếm)        [C]
  dev: thêm Hard-R@5 = % từ khó nằm trong top-5 theo p(khó) → hxattn chọn best.pt theo chỉ số này

--list_neg n (model 1 logit, DEEPDIVE E3): thêm n từ ngẫu nhiên trong kho vào mỗi câu, và cộng loss theo list:
  softmax trên [ô "không có" logit 0] + mọi ứng viên, dồn xác suất vào các từ có mặt                 [C]
  → học "từ đúng phải hơn mọi từ nhiễu trong list", gần với bài toán 1 từ đúng / 2000 lúc test

Train nhiều model cùng lúc, dùng chung 1 lần chạy encoder (encoder là phần tốn nhất):
  python -m training.train --arch pmaxsim pxattn --train train960 --tag _960 --dev_split dev-other --libri $L \\
      --eval_every 4000 --epochs 2 --lr 2e-4
--dev_split: dev = list IS21 N2000 thật của split đó (bench/protocol/<split>/), Recall@10 trên cả 2000 từ    [C]
--eval_every: đánh giá + lưu ckpt step<k>.pt mỗi k bước → so được ở cùng số bước với bản train ít dữ liệu
"""
import argparse, json, random, time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import BENCH, IS21, N_PHONE, WORK, Encoder, libri_path, make_store
from models import build, hard_logit, present_logit, score_chunks

ap = argparse.ArgumentParser()
ap.add_argument("--arch", nargs="+", default=["hou25"], help="nhiều arch = train song song, chung encoder")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=5e-5)                    # Hou25 §3.3, list IS21
ap.add_argument("--bs", type=int, default=8, help="số câu mỗi bước")
ap.add_argument("--neg_ratio", type=int, default=4)
ap.add_argument("--max_pos", type=int, default=8)
ap.add_argument("--dev_n", type=int, default=500)
ap.add_argument("--limit", type=int, help="chỉ dùng n câu train (chạy thử)")
ap.add_argument("--loc", action="store_true",
                help="train có vị trí [C]: từ đúng chỉ được tính điểm trong span MFA ± 0.1 s (work/train_align.json); nhiễu: cả câu")
ap.add_argument("--cache", help="tên cache feature câu nói (data/cache_speech.py), vd train → work/speech_train.f16")
ap.add_argument("--balance", action="store_true",
                help="E3: BCE cân bằng — từ đúng nhân trọng số neg_ratio, để tổng đóng góp đúng = nhiễu khi neg_ratio lớn")
ap.add_argument("--chunk", type=int, default=0, help="forward/backward theo nhóm k từ mỗi câu (đỡ OOM khi neg_ratio lớn)")
ap.add_argument("--list_neg", type=int, default=0, help="số từ nhiễu ngẫu nhiên thêm mỗi câu + loss theo list")
ap.add_argument("--name", help="tên thư mục ckpt khi chỉ 1 arch (mặc định = arch + tag)")
ap.add_argument("--tag", default="", help="hậu tố thư mục ckpt, vd _960")
ap.add_argument("--train", default="train", help="work/<train>.jsonl")
ap.add_argument("--dev_split", help="vd dev-other: dev = list N2000 thật (cần --libri)")
ap.add_argument("--libri")
ap.add_argument("--eval_every", type=int, default=0, help="đánh giá + lưu ckpt mỗi k bước (0 = chỉ cuối epoch)")
a = ap.parse_args()

near = json.load(open(WORK / "near.json", encoding="utf-8"))
load = lambda name: [json.loads(l) for l in open(WORK / f"{name}.jsonl", encoding="utf-8")]
pass1 = {}                                                                # uid → {"hyp", "hard"} (chỉ cần cho hxattn)
for f in WORK.glob("pass1_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        x = json.loads(line)
        pass1[x["uid"]] = x


def sample(it, rng, n_rand=None):
    """n_rand=None: kiểu train (có negative khó); n_rand=k: kiểu dev/test IS21 (k từ ngẫu nhiên)."""
    pos = [w for w in it["pos"] if w in store]
    pos = rng.sample(pos, min(len(pos), a.max_pos))
    ban = set(it["words"])
    n = a.neg_ratio * max(1, len(pos)) if n_rand is None else n_rand
    neg = set()
    if n_rand is None:
        hard = sorted({w for p in pos for w in near.get(p, []) if w not in ban and w in store})
        neg |= set(rng.sample(hard, min(len(hard), round(0.4 * n))))
        n += a.list_neg
    while len(neg) < n:
        w = rng.choice(rare)
        if w not in ban:
            neg.add(w)
    return pos + sorted(neg), [1.0] * len(pos) + [0.0] * len(neg)


class DS(Dataset):
    def __init__(self, items, dev=False):
        self.items, self.dev = items, dev

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if "list" in it:                                                  # dev = list IS21 thật
            words = [w for w in it["list"] if w in store]
            y = [float(w in it["pos"]) for w in words]
        else:
            words, y = sample(it, random.Random(i) if self.dev else random, 100 if self.dev else None)
        if it.get("uid") in cache_idx:                                    # feature encoder đã lưu sẵn
            o, n_ = cache_idx[it["uid"]]
            wav = torch.from_numpy(np.array(cache_mm[o:o + n_]))
        else:
            wav, _ = sf.read(it["path"], dtype="float32")
        fb, lb = store.get(words)
        if a.loc and not self.dev:                                        # (K, T) True = frame được phép khớp
            T = len(wav) if torch.is_tensor(wav) else max(1, min(1500, -(-len(wav) * 50 // 16000)))
            allow = torch.ones(len(words), T, dtype=torch.bool)
            al = train_align.get(it["uid"], {})
            for k, (w, v) in enumerate(zip(words, y)):
                if v and w in al:
                    allow[k] = False
                    for b, e in al[w]:
                        allow[k, max(0, int(b * 50) - 5):int(e * 50) + 6] = True
                    if not allow[k].any():
                        allow[k] = True
            return wav, fb, lb, torch.tensor(y), allow
        if not getattr(models[0], "uses_hyp", False):
            return wav, fb, lb, torch.tensor(y), None
        hard = set(pass1[it["uid"]]["hard"])
        y3 = [0 if v == 0 else 2 if w in hard else 1 for w, v in zip(words, y)]
        return wav, fb, lb, torch.tensor(y3), store.seq(pass1[it["uid"]]["hyp"].split())


def loader(items, dev):
    return DataLoader(DS(items, dev), batch_size=a.bs, shuffle=not dev, num_workers=8, collate_fn=list)


def feats(enc, xs):
    """Câu đã cache (tensor 2 chiều) → đưa thẳng lên GPU; câu chưa cache (wav) → chạy encoder."""
    raw = [i for i, x in enumerate(xs) if not torch.is_tensor(x)]
    out = [x.cuda() if torch.is_tensor(x) else None for x in xs]
    if raw:
        for i, h in zip(raw, enc([xs[i] for i in raw])):
            out[i] = h
    return out


def forward(model, fs, fb, lb, hyp):
    if hyp is not None and hyp.dtype == torch.bool:                       # --loc: ô thứ 5 là mặt nạ vị trí
        return model(fs, fb, lb, allow=hyp.cuda()[:, :fs.shape[0]])
    return model(fs, fb, lb, hyp.cuda()) if hyp is not None else model(fs, fb, lb)


def loss_fn(model, z, y, n, B):
    if z.dim() == 1:                                                      # model 1 logit: BCE như cũ
        pw = torch.tensor(float(a.neg_ratio), device=z.device) if a.balance else None
        bce = F.binary_cross_entropy_with_logits(z, y.float(), pos_weight=pw, reduction="sum") / n
        if not a.list_neg:
            return bce
        lp = torch.cat([z.new_zeros(1), z]).log_softmax(0)               # ô 0 = "không có", logit cố định 0
        pos = (y > 0).nonzero().squeeze(1) + 1
        return bce + -(torch.logsumexp(lp[pos], 0) if len(pos) else lp[0]) / B
    ce = F.cross_entropy(z, y, weight=torch.tensor([1.0, 1.0, 3.0], device=z.device), reduction="sum") / n
    lp = torch.cat([model.null.view(1).float(), present_logit(z)]).log_softmax(0)
    pos = (y > 0).nonzero().squeeze(1) + 1
    lst = -(torch.logsumexp(lp[pos], 0) if len(pos) else lp[0])
    return ce + lst / B


@torch.no_grad()
def evaluate(model, enc, dl):
    """→ (Recall@10 theo điểm "có mặt", Hard-R@5 theo điểm "khó"; -1 nếu model không có lớp khó, Recall@50)."""
    model.eval()
    hit = tot = hh = ht = h50 = c1 = 0
    for batch in dl:
        for fs, (_, fb, lb, y, hyp) in zip(feats(enc, [b[0] for b in batch]), batch):
            z = score_chunks(model, fs, fb.cuda(), lb.cuda(), hyp=None if hyp is None else hyp.cuda()).cpu()
            p = present_logit(z) if z.dim() == 2 else z
            if (y > 0).any():
                hit += int((y[p.topk(min(10, len(p))).indices] > 0).sum())
                h50 += int((y[p.topk(min(50, len(p))).indices] > 0).sum())
                c1 += int((p[y > 0] > p[y == 0].max()).sum()) if (y == 0).any() else int((y > 0).sum())   # Sạch@1
                tot += int((y > 0).sum())
            if z.dim() == 2 and (y == 2).any():
                hh += int((y[hard_logit(z).topk(5).indices] == 2).sum())
                ht += int((y == 2).sum())
    model.train()
    return 100 * hit / tot, (100 * hh / ht if ht else -1.0), 100 * h50 / tot, 100 * c1 / tot


def dev_items():
    if not a.dev_split:
        dev = load("dev")
        return random.Random(0).sample(dev, min(a.dev_n, len(dev)))
    proto = BENCH / "protocol" / a.dev_split
    ctx = json.load(open(proto / "ctx_N2000.json", encoding="utf-8"))
    items = []
    for line in open(proto / "ref.tsv", encoding="utf-8"):
        uid, _, rare_ = line.rstrip("\n").split("\t")
        items.append({"uid": uid, "path": str(libri_path(a.libri, uid)), "list": ctx[uid], "pos": set(json.loads(rare_))})
    return random.Random(0).sample(items, min(a.dev_n, len(items)))


if __name__ == "__main__":
    torch.manual_seed(0)
    cache_idx, cache_mm = {}, None
    if a.cache:
        for line in open(WORK / f"speech_{a.cache}.idx", encoding="utf-8"):
            u, o, n_ = line.rstrip("\n").split("\t")
            cache_idx[u] = (int(o), int(n_))
        cache_mm = np.memmap(WORK / f"speech_{a.cache}.f16", dtype=np.float16, mode="r").reshape(-1, 1280)
    train = load(a.train)[:a.limit]
    train_align = json.load(open(WORK / "train_align.json")) if a.loc else {}
    dev = dev_items()
    models = [build(x, N_PHONE).cuda() for x in a.arch]
    outs = [WORK / "ckpt" / (a.name if a.name and len(a.arch) == 1 else x + a.tag) for x in a.arch]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    store = make_store(models[0])                                         # feature TTS hoặc âm vị, theo kiến trúc
    assert all(m.bias_type == models[0].bias_type for m in models), "các arch train chung phải cùng loại bias"
    rare = [w for w in open(IS21 / "all_rare_words.txt", encoding="utf-8").read().split() if w in store]
    enc = Encoder()
    opts = [torch.optim.AdamW(m.parameters(), lr=a.lr) for m in models]
    for x, m in zip(a.arch, models):
        print(f"{x}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M tham số", flush=True)
    print(f"{len(train)} câu train ({a.train}), {len(dev)} câu dev ({a.dev_split or 'dev-clean, list 100'})", flush=True)

    best, step, t0 = [-1.0] * len(models), 0, time.time()

    def checkpoint(ep):
        for i, (x, m, o) in enumerate(zip(a.arch, models, outs)):
            r10, hr5, r50, c1 = evaluate(m, enc, loader(dev, True))
            crit = hr5 if hr5 >= 0 else (c1 if a.dev_split else r10)      # hxattn: Hard-R@5; dev là list thật: Sạch@1
            print(f"== {x} ep {ep} step {step}: dev Recall@10 {r10:.2f}  Recall@50 {r50:.2f}  Sạch@1 {c1:.2f}  Hard-R@5 {hr5:.2f}  "
                  f"(best {max(best[i], crit):.2f})  {time.time() - t0:.0f}s", flush=True)
            ck = {"arch": x, "args": vars(a), "epoch": ep, "step": step, "dev_r10": r10, "dev_r50": r50, "dev_clean1": c1,
                  "dev_hard_r5": hr5, "state": m.state_dict()}
            torch.save(ck, o / "last.pt")
            if a.eval_every:
                torch.save(ck, o / f"step{step}.pt")
            if crit > best[i]:
                best[i] = crit
                torch.save(ck, o / "best.pt")

    for ep in range(a.epochs):
        run = [0.0] * len(models)
        for batch in loader(train, False):
            n = sum(len(b[3]) for b in batch)
            fss = feats(enc, [b[0] for b in batch])                       # 1 lần encoder (hoặc cache) cho mọi model
            for i, (m, opt) in enumerate(zip(models, opts)):
                opt.zero_grad()
                for fs, (_, fb, lb, y, hyp) in zip(fss, batch):
                    c = a.chunk if a.chunk and not a.list_neg and hyp is None else len(y)   # BCE cộng dồn được theo nhóm
                    for j in range(0, len(y), c):
                        with torch.autocast("cuda", dtype=torch.bfloat16):  # backward từng câu cho đỡ tốn bộ nhớ
                            z = forward(m, fs, fb[j:j + c].cuda(), lb[j:j + c].cuda(), hyp).float()
                        loss = loss_fn(m, z, y[j:j + c].cuda(), n, len(batch))
                        loss.backward()
                        run[i] += loss.item()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)       # [B]
                opt.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss " + " ".join(f"{x} {r / 100:.4f}" for x, r in zip(a.arch, run))
                      + f"  {time.time() - t0:.0f}s", flush=True)
                run = [0.0] * len(models)
            if a.eval_every and step % a.eval_every == 0:
                checkpoint(ep)
        checkpoint(ep)
