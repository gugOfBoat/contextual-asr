# GPU 0 (container model0): Hou25 tái hiện = TTS + encode + train hou25, rồi chấm N2000. Tự dừng sau 5h50.
cd /work/code/scorer
( sleep 21000; pkill -f "python -m" ) &
L=/work/libri/LibriSpeech
rk() { python -m inference.rank --libri $L --lists N2000 --thr 0.5 "$@"; }
set -x
python -m data.tts --conc 32 > work/log_tts.txt 2>&1
python -m data.encode_bias --bs 64 > work/log_encode_bias.txt 2>&1
python -m training.train --arch hou25 --epochs 3 --lr 5e-5 > work/log_train_hou25.txt 2>&1
for S in test-other dev-other test-clean; do rk --split $S --ckpt work/ckpt/hou25/best.pt --name hou25; done
touch work/B_DONE
