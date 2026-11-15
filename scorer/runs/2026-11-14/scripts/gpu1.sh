cd /work/code/scorer
bash run_e.sh e3 > work/log_e3.txt 2>&1
cd /work/code/bench
for th in 0.6 0.7 0.8; do python postcorr.py --split test-other --hyp whisper_none_times --ctx hou25_win_N2000_p0.5 --spans protocol/test-other/span_hou25_win_N2000.tsv --phones /work/code/scorer/work/phones.tsv --theta $th --tag _t$th; done > /work/code/scorer/work/log_postcorr.txt 2>&1
touch /work/code/scorer/work/GPU1_DONE
