"""Bước 2 — đọc từng từ trong work/vocab.txt bằng edge-tts (Hou25 §3.3). Chạy lại thì bỏ qua từ đã có.

  python -m data.tts --conc 32

Giọng: paper không nêu → en-US-AriaNeural [B]. Từ lỗi sau 5 lần thử ghi vào work/tts_fail.txt.
"""
import argparse, asyncio, time

import edge_tts

from common import WORK, tts_path

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="en-US-AriaNeural")
ap.add_argument("--conc", type=int, default=32)
a = ap.parse_args()

words = [w for w in open(WORK / "vocab.txt", encoding="utf-8").read().split() if not tts_path(w).exists()]
fails, done, t0 = [], 0, time.time()


async def one(sem, w):
    global done
    out = tts_path(w)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for k in range(5):
            try:
                await edge_tts.Communicate(w, a.voice).save(str(out) + ".part")
                (out.parent / (out.name + ".part")).rename(out)
                break
            except Exception:
                await asyncio.sleep(2 ** k)
        else:
            fails.append(w)
    done += 1
    if done % 5000 == 0:
        print(f"{done}/{len(words)}  {done / (time.time() - t0):.1f} từ/s", flush=True)


async def main():
    sem = asyncio.Semaphore(a.conc)
    await asyncio.gather(*(one(sem, w) for w in words))

asyncio.run(main())
open(WORK / "tts_fail.txt", "w", encoding="utf-8").write("\n".join(fails))
print(f"xong {len(words)} từ, lỗi {len(fails)}")
