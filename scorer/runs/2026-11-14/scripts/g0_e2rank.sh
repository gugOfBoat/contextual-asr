cd /work/code/scorer
for A in hou25_diag_e2 hou25_e2 hou25_topk_e2; do
  python -m inference.rank --libri /work/libri/LibriSpeech --thr 0.5 --span --win --split test-other --lists N2000 --limit 1000 --ckpt work/ckpt/$A/best.pt --name $A
done
touch work/E2RANK_DONE
