cd /work/code/bench
for S in test-other test-clean; do
  F="--split $S --fallback"; SP=protocol/$S/span_pmaxsim_win_N2000.tsv; SM=protocol/$S/span_pmaxsim_win_missing.tsv
  [ -f $SM ] || python spans_from_rank.py /work/code/scorer/work/out/pmaxsim_win/$S/span_missing_N2000.json $SM
  python gate_boost.py $F --ctx pmaxsim_win_N2000_nms0.3 --boost 5 --gate span --margin 0 --once --spans $SP --tag _gate_m0_fb
  python gate_boost.py $F --ctx pmaxsim_win_missing_N2000_nms0.3 --boost 5 --gate span --margin 0 --once --spans $SM --tag _gate_m0_fb
  python gate_boost.py $F --ctx pmaxsim_win_N2000_nms0.3 --boost 5 --gate none --once --tag _nogate_b5_fb
done
python gate_boost.py --split test-other --fallback --ctx pmaxsim_win_missing_N2000_nms0.3 --boost 5 --gate none --once --tag _nogate_b5_fb
touch /work/code/scorer/work/GTEST2_DONE
