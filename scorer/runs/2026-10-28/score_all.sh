# Chấm mọi output Whisper của 28/10 (plain), ghi score_plain.txt. Chạy trong bench/: bash ../scorer/runs/2026-10-28/score_all.sh
for c in none N100 N500 N1000 N2000 oracle N100_spoken \
         pxattn_N100 pxattn_N500 pxattn_N1000 pxattn_N2000 pmaxsim_N100 pmaxsim_N500 pmaxsim_N1000 pmaxsim_N2000 \
         pxattn_N100_p0.5 pxattn_N500_p0.5 pxattn_N1000_p0.5 pxattn_N2000_p0.5 \
         pmaxsim_N100_p0.5 pmaxsim_N500_p0.5 pmaxsim_N1000_p0.5 pmaxsim_N2000_p0.5 \
         pxattn_N100_p0.5_spoken pmaxsim_N100_p0.5_spoken; do
  for s in test-clean test-other; do
    echo "$c $s $(PYTHONUTF8=1 python score.py --ref protocol/$s/ref.tsv --hyp out/$s/whisper_$c.tsv | tr '\n' ' ')"
  done
done > ../scorer/runs/2026-10-28/score_plain.txt
