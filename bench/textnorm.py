import re


def norm(s):
    """Đưa hyp về format reference LibriSpeech: chữ thường, bỏ dấu câu, giữ dấu nháy đơn."""
    s = s.lower().replace("’", "'").replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return " ".join(w.strip("'") for w in s.split() if w.strip("'"))
