cd /work/code/bench
for S in test-other test-clean; do
  python gate_boost.py --split $S --fallback --gate span --once --margin 0 --spans protocol/$S/span_pmaxsim_win_N2000.tsv --ctx pmaxsim_win_N2000 --prompt --boost_ctx pmaxsim_win_N2000_nms0.3 --boost 3 --tag _prompt_gate_b3_fb
done
python gate_boost.py --split test-other --fallback --gate span --once --margin 0 --spans protocol/test-other/span_pmaxsim_win_missing.tsv --ctx pmaxsim_win_missing_N2000 --prompt --boost_ctx pmaxsim_win_missing_N2000_nms0.3 --boost 3 --tag _prompt_gate_b3_fb
python gate_boost.py --split test-other --fallback --ctx pmaxsim_win_missing_N2000 --prompt --tag _prompt_fb
touch /work/code/scorer/work/GTEST3_DONE
