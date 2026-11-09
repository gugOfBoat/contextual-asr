cd /work/code/bench
S="--split dev-other --ctx oracle --subset protocol/dev-other/sub400.txt"
if [ $1 = A ]; then
python gate_boost.py $S
python gate_boost.py $S --prompt
for b in 1 2 3 5; do python gate_boost.py $S --boost $b --gate none; done
python gate_boost.py $S --boost 3 --gate span --margin 0.05
python gate_boost.py $S --boost 5 --gate span --margin 0.05
else
for b in 2 3 5 8; do python gate_boost.py $S --boost $b --gate span --margin 0.1; done
python gate_boost.py $S --boost 3 --gate span --margin 0.2
python gate_boost.py $S --boost 5 --gate span --margin 0.2
fi
