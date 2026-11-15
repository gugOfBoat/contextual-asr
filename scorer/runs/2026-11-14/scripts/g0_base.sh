# Mốc so sánh, mọi hệ có --fallback. Chạy trên GPU 0 cạnh E2.
cd /work/code/bench
for S in test-other test-clean; do
  F="--split $S --fallback"
  [ $S = test-clean ] && python gate_boost.py $F --ctx none --tag _fb
  python gate_boost.py $F --ctx hou25_win_N2000 --prompt --tag _prompt_fb
  python gate_boost.py $F --ctx oracle --boost 5 --gate span --margin 0.1 --once --tag _mfa_once_fb
  python gate_boost.py $F --ctx pmaxsim_win_N2000 --prompt --tag _prompt_fb
done
python gate_boost.py --split test-other --fallback --ctx hou25_win_missing_N2000 --prompt --tag _prompt_fb
touch /work/code/scorer/work/G0BASE_DONE
