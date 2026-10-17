# Bước tiếp theo: setup thực nghiệm chuẩn để so scorer với Hou25

> Mục đích: biết **scorer của nhóm tốt/kém hơn Hou25 ở đâu** trên **đúng setup của Hou25**, rồi mới quyết hướng cải thiện.
> Hiện mới so trên Qwen3-ASR, chưa so trên Whisper (ASR Hou25 dùng), và **bản Hou25 tái hiện chưa train** → chưa có so sánh công bằng.

## 1. Vì sao số hiện tại chưa đủ để kết luận
| Vấn đề | Hệ quả |
|---|---|
| Hou25 dùng **Whisper-large-v3-turbo**, nhóm mới đo trên **Qwen3-ASR** | Không đặt cạnh số trong paper được |
| Scorer Hou25 (TTS + CNN) có code (`scorer --arch hou25`) nhưng **chưa train** (feature TTS bị xoá 16/10) | Thiếu hàng "Hou25 tái hiện" |
| Chỉ chạy N = 2000 (và N100 cho phần lọc) | Hou25 báo đủ N = 100 / 500 / 1000 / 2000 |
| So "top-50" vs "lọc ngưỡng" mới trên 1 scorer | Chưa tách được: lợi ích do **scorer** hay do **cách lấy list** |
| Dev chọn checkpoint quá dễ (Recall@10 ≈ 99,7% từ epoch 0) | Không chọn được epoch / ngưỡng có ý nghĩa |
| PTC-Bias chỉ có số trong paper | Chưa chạy lại trên cùng output |

## 2. Setup chuẩn đề xuất
**Cố định (theo Hou25 [A]):** LibriSpeech test-clean / test-other, list IS21 N ∈ {100, 500, 1000, 2000}, train scorer trên train-clean-100,
prompt "naive" `w1, w2, ...`, top-k = 50 xếp tăng dần, metric WER / U-WER / B-WER bằng scorer IS21 (`bench/score_is21_official.py`).
Báo cả **plain** (để so với paper) và **wnorm** (số chính của nhóm).

**Lưới thí nghiệm** = ASR × scorer × cách lấy list × N:

| Trục | Giá trị |
|---|---|
| ASR | Whisper-large-v3-turbo (so với paper) · Qwen3-ASR-1.7B (mục tiêu của nhóm) |
| Scorer | không lọc (cả list) · **hou25 tái hiện** (TTS) · pxattn · pmaxsim · PTC-Bias retrieval (CTC-Filter) · oracle |
| Cách lấy list | top-50 tăng dần (như Hou25) · lọc ngưỡng p ≥ θ (θ chọn trên dev) |
| N | 100 · 500 · 1000 · 2000 (+ missing_N2000 để đo hại khi list sai) |

**Bảng đích** (mỗi ô: B-WER clean / other, đủ 4 cột N):
```
                           N=100   N=500   N=1000  N=2000   ms/câu
Whisper  không prompt
Whisper  cả list
Whisper  hou25 top-50      (paper: 5.4/10.0 · 5.4/10.7 · 5.6/11.3 · 5.8/11.7)
Whisper  hou25 tái hiện
Whisper  pxattn top-50 | pxattn ngưỡng
Qwen     (các hàng tương tự) + PTC-Bias (paper N2000: 4.10/7.76)
```
Whisper đã có sẵn 3 hàng nền (`bench/results.md` mục 2): không prompt 9.43/18.92, N100 8.04/13.98, N2000 9.15/18.69 (plain).

## 3. Việc cụ thể, theo thứ tự
1. **Dev khó hơn để chọn checkpoint và ngưỡng θ**: dev-clean + dev-other, list = từ đúng + 2000 từ ngẫu nhiên (giống test). Chọn θ trên dev, không nhìn test.
2. **Tái hiện scorer Hou25**: dựng lại TTS (`data/tts.py` ~80 phút + `data/encode_bias.py` ~25 phút, 21 GB), train `--arch hou25 --lr 5e-5`.
   Kiểm tra trên Whisper: có về gần số paper (5.4 / 5.8 ở N100 / N2000, test-clean) không.
3. **Chạy lưới** trên cả 2 ASR: `inference/rank.py` cho mọi scorer × N, rồi `bench/run_whisper.py` / `bench/run_qwen3asr.py`, `bench/score.py`.
   Ước tính: chấm list ~10–30 phút/scorer/tập; mỗi lượt ASR ~10 phút → ~60 lượt ≈ 10 giờ GPU.
4. **PTC-Bias**: chạy phần retrieval của họ (`bench/third_party/PTC-Bias`) trên cùng list để có hàng CTC-Filter thật (cần checkpoint phoneme — xem rủi ro trong `related-work/2609.28727-ptc-bias.md`).
5. Đo **Recall@K và thời gian** cho từng scorer (`bench/eval_retrieval.py`, `time.json`) để so chi phí.

## 4. Sau khi có bảng
- Nếu pxattn ≥ hou25 trên cả 2 ASR → đóng góp "âm vị thay TTS + lọc ngưỡng" đứng được; tiếp tục: ràng buộc khớp theo thứ tự (CTC) và cho các từ cạnh tranh nhau.
- Nếu hou25 tái hiện tốt hơn → giữ ý lọc ngưỡng, đổi backbone scorer.
- Song song (sau): sửa hxattn (che ngẫu nhiên hyp khi train để tránh học tắt); hướng "không có list" cần thêm nguồn ngữ cảnh (LLM).

## 5. Tài nguyên
- GPU: 1× RTX 4090 24 GB; máy GPU vẫn giữ LibriSpeech + checkpoint để chạy tiếp.
- Code chạy: `scorer/README.md` (scorer), `bench/README.md` (ASR + chấm điểm).
