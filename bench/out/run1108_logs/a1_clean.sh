while kill -0 535 2>/dev/null; do sleep 20; done
cd /work/code/scorer && python analysis/a1_pool.py --split test-clean --libri /work/libri/LibriSpeech > /work/run1108/a1_clean.log 2>&1
