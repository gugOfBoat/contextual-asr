# B3 (chặn chèn lặp) + B2 (so các cách đưa từ vào, cùng list). Chạy trong container: bash b3b2.sh 2>&1 | tee b3b2.log
# Ước: dev ~15 phút, test-other ~2 giờ (Whisper-turbo, 1 card).
cd /work/code/bench
D="--split dev-other --ctx oracle --subset protocol/dev-other/sub400.txt"
# B3 trên dev: chọn λ cho bản --once (so với bản cũ b{3,5,8}_span0.1 đã có trong out/dev-other)
for b in 3 5 8; do python gate_boost.py $D --boost $b --gate span --margin 0.1 --once; done
python gate_boost.py --split dev-other --ctx none --subset protocol/dev-other/sub400.txt --times --tag _times
# B2 trên test-other, cùng list top50or = top-50 (a1mean) ∪ từ đúng; λ = 5 (đổi nếu dev chọn khác)
S="--split test-other"; P=protocol/test-other/pred_a1s1.tsv; B=5
python gate_boost.py $S --ctx none --times --tag _times                                       # gốc cho (b)
python gate_boost.py $S --ctx top50or_N2000 --prompt                                          # (a) prompt
python gate_boost.py $S --ctx top50or_N2000 --boost $B --gate none --once                     # (c) boost không gate
python gate_boost.py $S --ctx top50or_N2000 --boost $B --gate span --margin 0.1 --spans $P --once --tag _s1_once   # (d) gate, span S1
python gate_boost.py $S --ctx oracle --boost $B --gate span --margin 0.1 --once --tag _mfa_once                    # trần: từ đúng + span MFA
python gate_boost.py $S --ctx top50or_N2000 --prompt --boost $B --gate span --margin 0.1 --spans $P --once --tag _prompt_s1_once  # (e)
# (b) sửa sau: chạy CPU, nhanh
for th in 0.6 0.7 0.8; do python postcorr.py --split dev-other --hyp whisper_none_times --ctx oracle --theta $th --tag _t$th; done
for th in 0.6 0.7 0.8; do python postcorr.py $S --hyp whisper_none_times --ctx top50or_N2000 --spans $P --theta $th --tag _s1_t$th; done
python postcorr.py $S --hyp whisper_none_times --ctx oracle --theta 0.7 --tag _mfa_t0.7
