cd /work/code/bench
for S in test-other test-clean; do
  F="--split $S --fallback --gate span --once"; P=protocol/$S
  python gate_boost.py $F --ctx pmono_c4_N2000_nms0.3 --boost 5 --margin 0.05 --spans $P/span_pmono_c4_N2000.tsv --tag _gate_m0.05_fb
  python gate_boost.py $F --ctx pmono_c4_missing_N2000_nms0.3 --boost 5 --margin 0.05 --spans $P/span_pmono_c4_missing_N2000.tsv --tag _gate_m0.05_fb
  python gate_boost.py $F --ctx pmaxsim_c4_N2000_nms0.3 --boost 5 --margin 0.05 --spans $P/span_hyb_c4_N2000.tsv --tag _gate_hyb_fb
  python gate_boost.py $F --ctx pmono_c4_N2000 --prompt --boost_ctx pmono_c4_N2000_nms0.3 --boost 3 --margin 0.05 --spans $P/span_pmono_c4_N2000.tsv --tag _prompt_gate_fb
done
python gate_boost.py --split test-other --fallback --ctx pmono_c4_N2000_nms0.3 --boost 5 --gate none --once --tag _nogate_fb
touch /work/code/scorer/work/GTEST4_DONE
