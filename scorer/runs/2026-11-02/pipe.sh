# Phân tích vị trí + feature của scorer (02/11). CPU only (4 card bận), container `model` --cpus 8.
set -x
cd /work/pos
for s in test-clean test-other; do python position.py feats --split $s --n 400; done
for s in test-clean test-other; do python position.py run --split $s --n 400; done
echo ALL_DONE
