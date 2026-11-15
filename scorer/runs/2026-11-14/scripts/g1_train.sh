cd /work/code/scorer
while [ ! -f work/CACHE_DONE ]; do sleep 30; done
T="python -m training.train --arch pmaxsim pmono --cache train --dev_split dev-other --libri /work/libri/LibriSpeech --epochs 6 --lr 2e-4"
$T --tag _c4 > work/log_train_c4.txt 2>&1 &
$T --neg_ratio 16 --balance --tag _c16 > work/log_train_c16.txt 2>&1 &
wait
touch work/G1TRAIN_DONE
