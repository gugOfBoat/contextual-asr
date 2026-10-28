"""Chạy 1 script/module với tên tiến trình "qwen-3 asr" và giới hạn 90% bộ nhớ GPU (yêu cầu của user, 28/10).

  python /work/go.py run_whisper.py --ctx N100            # trong thư mục chứa script
  python /work/go.py -m inference.rank --ckpt ...
"""
import os, runpy, sys

import setproctitle, torch

setproctitle.setproctitle("qwen-3 asr")
torch.cuda.set_per_process_memory_fraction(0.9)
sys.path.insert(0, os.getcwd())
if sys.argv[1] == "-m":
    sys.argv = sys.argv[2:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
