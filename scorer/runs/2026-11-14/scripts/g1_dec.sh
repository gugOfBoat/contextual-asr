# Scorer âm vị (pmaxsim, train clean-100) + vị trí → Whisper. Mọi hệ có --fallback.
cd /work/code/bench
O=/work/code/scorer/work/out
for S in dev-other test-other test-clean; do
  python spans_from_rank.py $O/pmaxsim_win/$S/span_N2000.json protocol/$S/span_pmaxsim_win_N2000.tsv
  python spans_from_rank.py $O/pmaxsim_win/$S/span_missing_N2000.json protocol/$S/span_pmaxsim_win_missing.tsv
done
# 1) chọn τ, λ trên dev-other (400 câu)
D="--split dev-other --subset protocol/dev-other/sub400.txt --fallback"
python gate_boost.py $D --ctx none --tag _fb
python gate_boost.py $D --ctx pmaxsim_win_N2000 --prompt --tag _prompt_fb
for t in 0.3 0.5 0.7; do for b in 3 5 8; do
  python gate_boost.py $D --ctx pmaxsim_win_N2000_p$t --boost $b --gate span --margin 0.1 --once --spans protocol/dev-other/span_pmaxsim_win_N2000.tsv --tag _gate_b${b}_fb
done; done
touch /work/code/scorer/work/G1DEV_DONE
