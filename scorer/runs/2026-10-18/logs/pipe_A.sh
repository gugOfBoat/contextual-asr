# GPU 2 (container model): E3 pxattn + loss theo list, rồi chấm N2000 có lưu điểm. Tự dừng sau 5h50.
cd /work/code/scorer
( sleep 21000; pkill -f "python -m" ) &
L=/work/libri/LibriSpeech
rk() { python -m inference.rank --libri $L --lists N2000 --thr 0.5 "$@"; }
set -x
python -m training.train --arch pxattn --name pxattn_list --list_neg 200 --epochs 4 --lr 2e-4 > work/log_train_pxattn_list.txt 2>&1
for S in test-other dev-other test-clean; do rk --split $S --ckpt work/ckpt/pxattn_list/best.pt --name pxattn_list; done
for S in test-other test-clean; do rk --split $S --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim; done
rk --split test-other --ckpt work/ckpt/pxattn/best.pt --name pxattn
touch work/A_DONE
