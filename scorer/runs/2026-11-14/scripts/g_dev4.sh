cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback --boost 5 --gate span --once --spans protocol/dev-other/span_pmaxsim_win_N2000.tsv"
python gate_boost.py $D --ctx pmaxsim_win_N2000 --prompt --boost_ctx pmaxsim_win_N2000_nms0.3 --margin 0 --tag _prompt_gate_m0_fb
python gate_boost.py $D --ctx pmaxsim_win_N2000 --prompt --boost_ctx pmaxsim_win_N2000_nms0.3 --margin 0.1 --tag _prompt_gate_m0.1_fb
python gate_boost.py $D --ctx oracle --margin 0 --tag _oracle_pspan_m0_fb
touch /work/code/scorer/work/GDEV4_DONE
