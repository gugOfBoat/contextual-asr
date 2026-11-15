# chờ train xong rồi chấm: $1 = file DONE, còn lại = tên ckpt
cd /work/code/scorer
while [ ! -f work/$1 ]; do sleep 30; done
shift
bash g1_rank2.sh "$@"
