# Code bên thứ ba (không đưa vào git)

| Thư mục | Nguồn | Commit đã dùng | Dùng để |
|---|---|---|---|
| `PTC-Bias/` | github.com/aizhiqi-work/PTC-Bias | `156c2517e3136a41fd9b8a77bd7e00c92e4e392d` (clone 03/10/2026) | đối thủ gần nhất; giải thích code: `related-work/2609.28727-ptc-bias.md` |

```bash
cd bench/third_party
git clone https://github.com/aizhiqi-work/PTC-Bias.git
cd PTC-Bias && git checkout 156c2517e3136a41fd9b8a77bd7e00c92e4e392d
```
Repo có 2 model nhỏ (`models/`, ~80 MB). Checkpoint WavLM phoneme mà config trỏ tới **không có nguồn tải** (xem rủi ro trong note).
