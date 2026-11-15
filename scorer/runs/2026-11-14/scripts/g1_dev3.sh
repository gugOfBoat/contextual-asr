cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback --ctx pmaxsim_win_N2000_nms0.3 --boost 5 --gate span --once --spans protocol/dev-other/span_pmaxsim_win_N2000.tsv"
python gate_boost.py $D --margin 0 --tag _gate_m0_fb
python gate_boost.py $D --margin 0.1 --topk 20 --tag _gate_m0.1_k20_fb
python gate_boost.py $D --margin 0 --topk 20 --tag _gate_m0_k20_fb
python gate_boost.py $D --margin 0.1 --topk 5 --tag _gate_m0.1_k5_fb
touch /work/code/scorer/work/G1DEV3_DONE
