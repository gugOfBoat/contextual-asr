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
"""
import argparse, json, random, time

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import IS21, N_PHONE, WORK, Encoder, make_store
from models import build, hard_logit, present_logit, score_chunks

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="hou25")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=5e-5)                    # Hou25 §3.3, list IS21
ap.add_argument("--bs", type=int, default=8, help="số câu mỗi bước")
ap.add_argument("--neg_ratio", type=int, default=4)
ap.add_argument("--max_pos", type=int, default=8)
ap.add_argument("--dev_n", type=int, default=500)
ap.add_argument("--limit", type=int, help="chỉ dùng n câu train (chạy thử)")
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
        words, y = sample(it, random.Random(i) if self.dev else random, 100 if self.dev else None)
        wav, _ = sf.read(it["path"], dtype="float32")
        fb, lb = store.get(words)
        if not getattr(model, "uses_hyp", False):
            return wav, fb, lb, torch.tensor(y), None
        hard = set(pass1[it["uid"]]["hard"])
        y3 = [0 if v == 0 else 2 if w in hard else 1 for w, v in zip(words, y)]
        return wav, fb, lb, torch.tensor(y3), store.seq(pass1[it["uid"]]["hyp"].split())


def loader(items, dev):
    return DataLoader(DS(items, dev), batch_size=a.bs, shuffle=not dev, num_workers=8, collate_fn=list)


def forward(model, fs, fb, lb, hyp):
    return model(fs, fb, lb, hyp.cuda()) if hyp is not None else model(fs, fb, lb)


def loss_fn(model, z, y, n, B):
    if z.dim() == 1:                                                      # model 1 logit: BCE như cũ
        return F.binary_cross_entropy_with_logits(z, y.float(), reduction="sum") / n
    ce = F.cross_entropy(z, y, weight=torch.tensor([1.0, 1.0, 3.0], device=z.device), reduction="sum") / n
    lp = torch.cat([model.null.view(1).float(), present_logit(z)]).log_softmax(0)
    pos = (y > 0).nonzero().squeeze(1) + 1
    lst = -(torch.logsumexp(lp[pos], 0) if len(pos) else lp[0])
    return ce + lst / B


@torch.no_grad()
def evaluate(model, enc, dl):
    """→ (Recall@10 theo điểm "có mặt", Hard-R@5 theo điểm "khó"; -1 nếu model không có lớp khó)."""
    model.eval()
    hit = tot = hh = ht = 0
    for batch in dl:
        for fs, (_, fb, lb, y, hyp) in zip(enc([b[0] for b in batch]), batch):
            z = score_chunks(model, fs, fb.cuda(), lb.cuda(), hyp=None if hyp is None else hyp.cuda()).cpu()
            p = present_logit(z) if z.dim() == 2 else z
            if (y > 0).any():
                hit += int((y[p.topk(10).indices] > 0).sum())
                tot += int((y > 0).sum())
            if z.dim() == 2 and (y == 2).any():
                hh += int((y[hard_logit(z).topk(5).indices] == 2).sum())
                ht += int((y == 2).sum())
    model.train()
    return 100 * hit / tot, (100 * hh / ht if ht else -1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    out = WORK / "ckpt" / a.arch
    out.mkdir(parents=True, exist_ok=True)
    train = load("train")[:a.limit]
    dev = load("dev")
    dev = random.Random(0).sample(dev, min(a.dev_n, len(dev)))
    model = build(a.arch, N_PHONE).cuda()
    store = make_store(model)                                             # feature TTS hoặc âm vị, theo kiến trúc
    rare = [w for w in open(IS21 / "all_rare_words.txt", encoding="utf-8").read().split() if w in store]
    enc = Encoder()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    print(f"{a.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham số, "
          f"{len(train)} câu train, {len(dev)} câu dev", flush=True)

    best, step, t0 = -1.0, 0, time.time()
    for ep in range(a.epochs):
        run = 0.0
        for batch in loader(train, False):
            opt.zero_grad()
            n = sum(len(b[3]) for b in batch)
            for fs, (_, fb, lb, y, hyp) in zip(enc([b[0] for b in batch]), batch):
                with torch.autocast("cuda", dtype=torch.bfloat16):    # backward từng câu cho đỡ tốn bộ nhớ
                    z = forward(model, fs, fb.cuda(), lb.cuda(), hyp).float()
                loss = loss_fn(model, z, y.cuda(), n, len(batch))
                loss.backward()
                run += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # [B]
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss {run / 100:.4f}  {time.time() - t0:.0f}s", flush=True)
                run = 0.0
        r10, hr5 = evaluate(model, enc, loader(dev, True))
        crit = hr5 if hr5 >= 0 else r10                                   # hxattn chọn theo Hard-R@5
        print(f"== ep {ep}: dev Recall@10 {r10:.2f}  Hard-R@5 {hr5:.2f}  (best {max(best, crit):.2f})  "
              f"{time.time() - t0:.0f}s", flush=True)
        ck = {"arch": a.arch, "args": vars(a), "epoch": ep, "dev_r10": r10, "dev_hard_r5": hr5,
              "state": model.state_dict()}
        torch.save(ck, out / "last.pt")
        if crit > best:
            best = crit
            torch.save(ck, out / "best.pt")
