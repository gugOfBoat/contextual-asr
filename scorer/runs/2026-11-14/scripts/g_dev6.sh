cd /work/code/bench
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback --gate span --once"; P=protocol/dev-other
python gate_boost.py $D --ctx pmaxsim_c4_N2000_nms0.3 --boost 5 --margin 0 --spans $P/span_pmaxsim_c4_N2000.tsv --tag _gate_m0_fb
python gate_boost.py $D --ctx pmaxsim_c4_N2000_nms0.3 --boost 5 --margin 0.05 --spans $P/span_hyb_c4_N2000.tsv --tag _gate_hyb_fb
python gate_boost.py $D --ctx pmono_c4_N2000_nms0.3 --boost 5 --margin 0.05 --spans $P/span_pmono_c4_N2000.tsv --tag _gate_m0.05_fb
python gate_boost.py $D --ctx pmaxsim_c4_N2000 --prompt --boost_ctx pmaxsim_c4_N2000_nms0.3 --boost 3 --margin 0.05 --spans $P/span_hyb_c4_N2000.tsv --tag _prompt_gate_hyb_fb
python gate_boost.py $D --ctx pmaxsim_c4_N2000 --prompt --tag _prompt_fb
python gate_boost.py $D --ctx oracle --boost 5 --margin 0.05 --spans $P/span_hyb_c4_N2000.tsv --tag _hybspan_fb
touch /work/code/scorer/work/GDEV6_DONE
