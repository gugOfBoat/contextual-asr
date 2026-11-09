while pgrep -f "a1_clean.sh|a1_pool.py" >/dev/null; do sleep 20; done
for sp in test-other test-clean; do bash /work/run1108/e1_base.sh $sp > /work/run1108/e1_base_$sp.log 2>&1 & done
bash /work/run1108/sweep.sh A > /work/run1108/sweep_A.log 2>&1 &
bash /work/run1108/sweep.sh B > /work/run1108/sweep_B.log 2>&1 &
wait
