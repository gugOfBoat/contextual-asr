cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback --gate span --once --margin 0.05 --spans protocol/dev-other/span_$1_N2000.tsv"
python gate_boost.py $D --ctx $1_N2000_nms0.3 --boost 5 --tag _gate_fb
python gate_boost.py $D --ctx $1_N2000_nms0.5 --boost 5 --tag _gate_fb
python gate_boost.py $D --ctx $1_N2000 --prompt --boost_ctx $1_N2000_nms0.3 --boost 3 --tag _prompt_gate_fb
touch /work/code/scorer/work/ENSDEV_$1_DONE
