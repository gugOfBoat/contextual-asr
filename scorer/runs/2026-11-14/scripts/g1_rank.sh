cd /work/code/scorer
L=/work/libri/LibriSpeech
rk() { python -m inference.rank --libri $L --thr 0.3 0.5 0.7 0.9 --span --win "$@"; }
rk --split dev-other --lists N2000 missing_N2000 --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim_win
rk --split test-other --lists N2000 missing_N2000 hard_N2000 --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim_win
rk --split test-clean --lists N2000 missing_N2000 --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim_win
for S in dev-other test-other test-clean; do rk --split $S --lists N2000 missing_N2000 --ckpt work/ckpt/pmaxsim_960/best.pt --name pmaxsim960_win; done
rk --split test-other --lists hard_N2000 --ckpt work/ckpt/pmaxsim_960/best.pt --name pmaxsim960_hard
touch work/G1RANK_DONE
