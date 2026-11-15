cd /work/code/bench
N=$1
for S in test-other test-clean; do
  F="--split $S --fallback --gate span --once --margin 0.05"; P=protocol/$S
  python gate_boost.py $F --ctx ${N}_N2000_nms0.3 --boost 5 --spans $P/span_${N}_N2000.tsv --tag _gate_fb
  python gate_boost.py $F --ctx ${N}_N2000 --prompt --boost_ctx ${N}_N2000_nms0.3 --boost 3 --spans $P/span_${N}_N2000.tsv --tag _prompt_gate_fb
  python gate_boost.py $F --ctx ${N}_missing_N2000_nms0.3 --boost 5 --spans $P/span_${N}_missing_N2000.tsv --tag _gate_fb
done
python gate_boost.py --split test-other --fallback --ctx ${N}_N2000_nms0.3 --boost 5 --gate none --once --tag _nogate_fb
python gate_boost.py --split test-other --fallback --ctx ${N}_N2000 --prompt --tag _prompt_fb
touch /work/code/scorer/work/ENSTEST_${N}_DONE
