cd /work/code/bench
for L in N100 N500 N1000; do
  python gate_boost.py --split test-other --fallback --gate span --once --margin 0.05 --ctx pdual2dl_${L}_nms0.3 --boost 5 --spans protocol/test-other/span_pdual2dl_${L}.tsv --tag _gate_fb
done
touch /work/code/scorer/work/NSWEEP_DONE
