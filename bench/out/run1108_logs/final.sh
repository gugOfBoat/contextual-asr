cd /work/code/bench
S="--split test-other --boost 5"
P=protocol/test-other/pred_a1s1.tsv
if [ $1 = A ]; then
python gate_boost.py $S --ctx oracle --gate none
python gate_boost.py $S --ctx oracle --gate span --margin 0.1 --tag _mfa
python gate_boost.py --split test-other --ctx a1mean_N2000 --prompt
else
python gate_boost.py $S --ctx oracle --gate span --margin 0.1 --spans $P --tag _s1
python gate_boost.py $S --ctx a1mean_N2000 --gate span --margin 0.1 --spans $P --tag _s1
python gate_boost.py $S --ctx a1mean_N2000 --gate none
fi
