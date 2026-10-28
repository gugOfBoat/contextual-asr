# Bảng 1 + Bảng 2 kiểu Hou25 với scorer của nhóm, Whisper-large-v3-turbo, GPU 0, tối đa 12h (28/10).
# Chạy: timeout -s KILL 12h bash pipe_hou25tab.sh  (thứ tự = ưu tiên, bị cắt giờ thì mất phần cuối)
set -x
G="python /work/go.py"
L=/work/libri/LibriSpeech
S="test-clean test-other"

# 0. audio test (LibriSpeech trên server đã bị xoá)
[ -d $L/test-other ] || (cd /work/code/scorer && python -m data.fetch_libri_hf --libri $L --splits test-clean test-other)

# 1. chấm list N500/N1000 cho pmaxsim 100h (pxattn đã có đủ 4 mức N)
cd /work/code/scorer
for s in $S; do $G -m inference.rank --ckpt work/ckpt/pmaxsim/best.pt --name pmaxsim --split $s --libri $L --lists N100 N500 N1000 N2000; done

cd /work/code/bench
w() { for s in $S; do $G run_whisper.py --split $s "$@"; done; }
# 2. pxattn: top-50 (cách chọn của Hou25) và p≥0.5 (cách của nhóm)
w --ctx pxattn_N100 pxattn_N100_p0.5 pxattn_N2000 pxattn_N2000_p0.5
w --ctx pxattn_N500 pxattn_N500_p0.5 pxattn_N1000 pxattn_N1000_p0.5
# 3. baseline "+ prompt" còn thiếu (đã có none, N100, N2000)
w --ctx N500 N1000
# 4. Bảng 2: no distractor, spoken
w --ctx oracle
w --ctx N100 pxattn_N100_p0.5 --prompt spoken --tag _spoken
# 5. pmaxsim
w --ctx pmaxsim_N100 pmaxsim_N100_p0.5 pmaxsim_N2000 pmaxsim_N2000_p0.5
w --ctx pmaxsim_N500 pmaxsim_N500_p0.5 pmaxsim_N1000 pmaxsim_N1000_p0.5
w --ctx pmaxsim_N100_p0.5 --prompt spoken --tag _spoken
echo ALL_DONE
