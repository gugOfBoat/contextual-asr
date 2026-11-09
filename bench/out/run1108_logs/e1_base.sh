cd /work/code/bench
sp=$1
python gate_boost.py --split $sp --ctx oracle
python gate_boost.py --split $sp --ctx oracle --prompt
python gate_boost.py --split $sp --ctx hou25_N2000 --prompt
