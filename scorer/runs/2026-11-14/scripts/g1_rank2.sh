cd /work/code/scorer
L=/work/libri/LibriSpeech
rk() { python -m inference.rank --libri $L --thr 0.3 0.5 0.7 --span --win "$@"; }
for A in $@; do
  rk --split test-other --lists N2000 missing_N2000 hard_N2000 --ckpt work/ckpt/$A/best.pt --name $A
  rk --split dev-other --lists N2000 missing_N2000 --ckpt work/ckpt/$A/best.pt --name $A
  rk --split test-clean --lists N2000 missing_N2000 --ckpt work/ckpt/$A/best.pt --name $A
done
touch work/RANK2_$1_DONE
