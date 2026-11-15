"""MFA (Lugosch 2019) của train-clean-100 → work/train_align.json {uid: {từ: [[bắt đầu, kết thúc], ...]}} (giây)."""
import json, re
from pathlib import Path
from common import WORK
iv = re.compile(r"xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = \"(.*?)\"")
out, miss = {}, 0
for l in open(WORK / "train.jsonl", encoding="utf-8"):
    it = json.loads(l); uid = it["uid"]; spk, chap, _ = uid.split("-")
    p = Path("/work/align/train-clean-100") / spk / chap / f"{uid}.TextGrid"
    if not p.exists():
        miss += 1; continue
    words = p.read_text(encoding="utf-8").split("name = \"phones\"")[0]
    d = {}
    for b, e, w in iv.findall(words):
        if w:
            d.setdefault(w.lower(), []).append([float(b), float(e)])
    out[uid] = d
json.dump(out, open(WORK / "train_align.json", "w"))
print(len(out), "câu, thiếu", miss)
