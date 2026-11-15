cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback --gate span --once --margin 0 --spans protocol/dev-other/span_pmaxsim_win_N2000.tsv --ctx pmaxsim_win_N2000 --prompt"
python gate_boost.py $D --boost_ctx pmaxsim_win_N2000_nms0.3 --boost 2 --tag _prompt_gate_b2_fb
python gate_boost.py $D --boost_ctx pmaxsim_win_N2000_nms0.3 --boost 3 --tag _prompt_gate_b3_fb
python gate_boost.py $D --boost_ctx pmaxsim_win_N2000_nms0.5 --boost 3 --tag _prompt_gate_b3_n5_fb
touch /work/code/scorer/work/GDEV5_DONE
