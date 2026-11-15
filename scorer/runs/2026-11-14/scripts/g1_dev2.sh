cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback"; SP=protocol/dev-other/span_pmaxsim_win_N2000.tsv
python gate_boost.py $D --ctx pmaxsim_win_N2000_nms0.5 --boost 5 --gate span --margin 0.1 --once --spans $SP --tag _gate_b5_fb
python gate_boost.py $D --ctx pmaxsim_win_N2000_nms0.3 --boost 5 --gate span --margin 0.1 --once --spans $SP --tag _gate_b5_fb
python gate_boost.py $D --ctx pmaxsim_win_N2000_nms0.5 --boost 4 --gate span --margin 0.1 --once --spans $SP --tag _gate_b4_fb
python gate_boost.py $D --ctx pmaxsim_win_N2000_nms0.5 --boost 5 --gate none --once --tag _nogate_b5_fb
touch /work/code/scorer/work/G1DEV2_DONE
