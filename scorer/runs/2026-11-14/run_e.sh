# Roadmap scorer 14/11 (scorer/PLAN-2011.md): E1 cửa sổ trượt, E4 list nghe giống, E2 cách gộp, E3 nhiều nhiễu.
# Chạy trong container 1 GPU:  bash run_e.sh <phần>   (phần = smoke | e1e4 | adapter | e2 | e3)
cd /work/code/scorer
L=/work/libri/LibriSpeech
rk() { python -m inference.rank --libri $L --thr 0.5 --span "$@"; }   # --span: lưu vị trí cho Đúng@chỗ
set -x
case $1 in
smoke)  # ~15 phút: mọi đường code chạy được, không OOM
  rk --split test-other --lists N2000 --ckpt work/ckpt/hou25/best.pt --name smoke_win --win --limit 32
  python -m training.train --arch hou25 hou25_topk hou25_diag --epochs 1 --limit 200 --tag _smoke   # nhiều arch: --name bị bỏ qua, phải dùng --tag để KHÔNG ghi đè ckpt cũ
  python -m training.train --arch hou25 --name smoke_e3 --neg_ratio 16 --balance --chunk 48 --epochs 1 --limit 200
  cd /work/code/bench && python spans_from_rank.py ../scorer/work/out/smoke_win/test-other/span_N2000.json /tmp/sp.tsv     && python gate_boost.py --split test-other --ctx smoke_win_N2000_p0.5 --boost 5 --gate span --once --spans /tmp/sp.tsv --limit 8 --tag _smoke
  ;;
e1e4)   # ~5 giờ, không train
  rk --split test-other --lists N2000 missing_N2000 --ckpt work/ckpt/hou25/best.pt --name hou25_win --win   # missing: list toàn nhiễu, cho B2
  rk --split test-clean --lists N2000 --ckpt work/ckpt/hou25/best.pt --name hou25_win --win
  for S in test-other test-clean; do
    rk --split $S --lists N2000 --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim_win --win
  done
  rk --split test-other --lists hard_N2000 --ckpt work/ckpt/hou25/best.pt --name hou25_win --win
  rk --split test-other --lists hard_N2000 --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim_win --win
  ;;
adapter)  # ~3 giờ, sau e1e4. B3 chọn λ trên dev; B2: Whisper có cần vị trí không, cùng list thật (Hou25 tái hiện, --win)
  cd /work/code/bench
  O=/work/code/scorer/work/out/hou25_win/test-other; S="--split test-other"
  python spans_from_rank.py $O/span_N2000.json protocol/test-other/span_hou25_win_N2000.tsv
  python spans_from_rank.py $O/span_missing_N2000.json protocol/test-other/span_hou25_win_missing.tsv
  D="--split dev-other --ctx oracle --subset protocol/dev-other/sub400.txt"
  for b in 3 5 8; do python gate_boost.py $D --boost $b --gate span --margin 0.1 --once; done          # B3
  B=5   # đổi theo B3 nếu dev chọn khác
  python gate_boost.py $S --ctx none --times --tag _times                                                # gốc + giây từng token (cho sửa sau)
  python gate_boost.py $S --ctx hou25_win_N2000 --prompt                                                 # Hou25: top-50 vào prompt
  python gate_boost.py $S --ctx hou25_win_N2000_p0.5 --prompt                                            # chỉ từ p≥0.5 vào prompt
  python gate_boost.py $S --ctx hou25_win_N2000_p0.5 --boost $B --gate none --once                       # đẩy KHÔNG vị trí
  python gate_boost.py $S --ctx hou25_win_N2000_p0.5 --boost $B --gate span --margin 0.1 --once --spans protocol/test-other/span_hou25_win_N2000.tsv --tag _gate_once   # đẩy CÓ vị trí
  python gate_boost.py $S --ctx oracle --boost $B --gate span --margin 0.1 --once --tag _mfa_once        # trần: từ đúng + vị trí MFA
  # audio KHÔNG có từ nào trong list (missing_N2000): đếm số từ bị viết bừa
  python gate_boost.py $S --ctx hou25_win_missing_N2000 --prompt
  python gate_boost.py $S --ctx hou25_win_missing_N2000_p0.5 --boost $B --gate none --once
  python gate_boost.py $S --ctx hou25_win_missing_N2000_p0.5 --boost $B --gate span --margin 0.1 --once --spans protocol/test-other/span_hou25_win_missing.tsv --tag _gate_once
  for th in 0.6 0.7 0.8; do python postcorr.py $S --hyp whisper_none_times --ctx hou25_win_N2000_p0.5 --spans protocol/test-other/span_hou25_win_N2000.tsv --theta $th --tag _t$th; done   # sửa sau theo vị trí
  ;;
e2)     # ~12 giờ: 3 cách gộp train CHUNG một lần encoder, cùng seed / lr / dữ liệu / số epoch; mean train lại cho công bằng
  python -m training.train --arch hou25 hou25_topk hou25_diag --tag _e2 --epochs 3 --lr 5e-5 > work/log_train_e2.txt 2>&1
  for A in hou25 hou25_topk hou25_diag; do for S in test-other test-clean; do
    rk --split $S --lists N2000 --ckpt work/ckpt/${A}_e2/best.pt --name ${A}_e2 --win
  done; done
  ;;
e3)     # ~6 giờ: Hou25 mean, 16 nhiễu / từ đúng (thay vì 4), BCE cân bằng
  python -m training.train --arch hou25 --name hou25_neg16 --neg_ratio 16 --balance --chunk 48 --epochs 2 --lr 5e-5 \
      > work/log_train_e3.txt 2>&1
  for S in test-other test-clean; do rk --split $S --lists N2000 --ckpt work/ckpt/hou25_neg16/best.pt --name hou25_neg16 --win; done
  ;;
esac
# Đo (CPU, máy local sau khi kéo work/out/ về):
#   python analysis/eval_e.py --split test-other --base hou25_win hou25_win=<out>/hou25_win/test-other \
#       pmaxsim_win=<out>/pmaxsim_win/test-other hou25_e2=... hou25_topk_e2=... hou25_diag_e2=... hou25_neg16=...
#   E4: thêm --list hard_N2000
#   B2: python bench/e1_table.py --split test-other bench/out/test-other/whisper_*hou25_win*.tsv ... ; bench/misplace.py; bench/error_breakdown.py
