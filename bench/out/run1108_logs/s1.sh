cd /work/code/scorer
python analysis/a1_s1.py --split test-other --libri /work/libri/LibriSpeech > /work/run1108/s1_other.log 2>&1
python analysis/a1_s1.py --split test-clean --libri /work/libri/LibriSpeech > /work/run1108/s1_clean.log 2>&1
