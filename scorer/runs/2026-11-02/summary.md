
## test-clean: 395 câu, 1085 từ hiếm đúng

### Q1. Định vị từ (|err| = khoảng cách tới tâm từ theo MFA; trúng = trong [đầu−0.2s, cuối+0.2s])

| bản đồ | ước lượng | median \|err\| (s) | trong 0.5 s (%) | trúng span (%) | mass/mean trong span | chance |
| --- | --- | --- | --- | --- | --- | --- |
| hou25 · cosine TTS×audio | prof | 0.10 | 82.0 | 81.9 | 0.238 vs 0.119 | 0.07 |
| hou25 · cosine TTS×audio | med | 0.06 | 98.7 | 98.7 |  |  |
| hou25 · cosine TTS×audio | streak | 0.06 | 98.9 | 98.9 |  |  |
| pmaxsim · cosine âm vị×audio | prof | 0.05 | 99.8 | 99.7 | 0.125 vs -0.208 | 0.07 |
| pmaxsim · cosine âm vị×audio | med | 0.04 | 99.4 | 99.4 |  |  |
| pmaxsim · cosine âm vị×audio | streak | 0.04 | 99.9 | 99.8 |  |  |
| pxattn · attention lớp 1 | prof | 0.09 | 99.5 | 99.5 | 0.87 | 0.07 |
| pxattn · attention lớp 1 | med | 0.04 | 99.2 | 99.2 |  |  |
| pxattn · attention lớp 1 | streak | 0.03 | 99.5 | 99.5 |  |  |
| pxattn · attention lớp 2 | prof | 0.94 | 43.6 | 43.4 | 0.16 | 0.07 |
| pxattn · attention lớp 2 | med | 0.96 | 42.0 | 41.8 |  |  |
| pxattn · attention lớp 2 | streak | 0.45 | 50.8 | 49.9 |  |  |
| pxattn · cosine thô (không train) | prof | 2.55 | 9.4 | 8.9 | -0.015 vs -0.009 | 0.07 |
| pxattn · cosine thô (không train) | med | 2.20 | 11.6 | 11.2 |  |  |
| pxattn · cosine thô (không train) | streak | 2.54 | 9.7 | 8.8 |  |  |
| mốc: giữa câu |  | 1.95 | 14.3 | 14.6 |  |  |
| mốc: ngẫu nhiên |  | 2.52 | 11.6 | 10.2 |  |  |

### Q5. Từ đúng vs từ nhiễu (AUC)

| scorer | xác suất của scorer | đỉnh profile | đỉnh đường chéo |
| --- | --- | --- | --- |
| hou25 · cosine TTS×audio | 0.999 | 0.816 | 0.996 |
| pmaxsim · cosine âm vị×audio | 1.000 | 0.999 | 0.999 |
| pxattn · attention lớp 1 | 1.000 | 0.743 | 0.970 |
| pxattn · attention lớp 2 | 1.000 | 0.363 | 0.500 |
| pxattn · cosine thô (không train) | 1.000 | 0.557 | 0.534 |

### Q2–Q3. Bên trong từ: âm vị j có đỉnh ở đúng đoạn âm vị j không (từ có âm vị MFA = g2p)

Âm vị g2p trùng hoàn toàn âm vị MFA: 89.0% số từ.

| bản đồ | đỉnh trong đoạn âm vị của chính nó (%) | đỉnh trong từ ±0.2 s (%) | lệch chỉ số âm vị (TB) | đơn điệu (%) | Spearman TB | đoạn audio → đúng hàng âm vị (%) | chance (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pmaxsim · cosine âm vị×audio | 54.2 | 99.2 | 0.92 | 93.5 | 0.97 | 47.0 | 17.5 |
| pxattn · attention lớp 1 | 66.9 | 97.6 | 0.64 | 81.9 | 0.95 | 65.0 | 17.5 |
| pxattn · attention lớp 2 | 14.8 | 37.4 | 1.29 | 18.7 | 0.19 | 36.2 | 17.5 |
| pxattn · cosine thô (không train) | 1.3 | 10.9 | 2.85 | 3.3 | 0.02 | 15.7 | 17.5 |

Mốc may rủi cho cột 1: đỉnh rơi ngẫu nhiên trong câu thì trúng đoạn âm vị của chính nó 1.2% số lần.

Thời lượng TB một đoạn âm vị (MFA) trong từ hiếm: 0.099 s = 4.9 frame.

## test-other: 394 câu, 979 từ hiếm đúng

### Q1. Định vị từ (|err| = khoảng cách tới tâm từ theo MFA; trúng = trong [đầu−0.2s, cuối+0.2s])

| bản đồ | ước lượng | median \|err\| (s) | trong 0.5 s (%) | trúng span (%) | mass/mean trong span | chance |
| --- | --- | --- | --- | --- | --- | --- |
| hou25 · cosine TTS×audio | prof | 0.10 | 80.5 | 80.2 | 0.235 vs 0.124 | 0.08 |
| hou25 · cosine TTS×audio | med | 0.05 | 97.3 | 97.1 |  |  |
| hou25 · cosine TTS×audio | streak | 0.06 | 96.7 | 96.7 |  |  |
| pmaxsim · cosine âm vị×audio | prof | 0.05 | 99.1 | 99.1 | 0.140 vs -0.206 | 0.08 |
| pmaxsim · cosine âm vị×audio | med | 0.04 | 98.9 | 98.9 |  |  |
| pmaxsim · cosine âm vị×audio | streak | 0.03 | 99.3 | 99.3 |  |  |
| pxattn · attention lớp 1 | prof | 0.08 | 98.3 | 98.4 | 0.84 | 0.08 |
| pxattn · attention lớp 1 | med | 0.04 | 99.1 | 99.0 |  |  |
| pxattn · attention lớp 1 | streak | 0.03 | 99.2 | 99.2 |  |  |
| pxattn · attention lớp 2 | prof | 0.71 | 46.2 | 44.4 | 0.18 | 0.08 |
| pxattn · attention lớp 2 | med | 0.75 | 44.8 | 43.2 |  |  |
| pxattn · attention lớp 2 | streak | 0.38 | 51.7 | 50.6 |  |  |
| pxattn · cosine thô (không train) | prof | 1.96 | 13.2 | 11.5 | -0.019 vs -0.012 | 0.08 |
| pxattn · cosine thô (không train) | med | 1.74 | 17.3 | 15.2 |  |  |
| pxattn · cosine thô (không train) | streak | 2.05 | 14.2 | 12.3 |  |  |
| mốc: giữa câu |  | 1.54 | 17.7 | 16.6 |  |  |
| mốc: ngẫu nhiên |  | 1.95 | 15.5 | 12.4 |  |  |

### Q5. Từ đúng vs từ nhiễu (AUC)

| scorer | xác suất của scorer | đỉnh profile | đỉnh đường chéo |
| --- | --- | --- | --- |
| hou25 · cosine TTS×audio | 0.995 | 0.805 | 0.990 |
| pmaxsim · cosine âm vị×audio | 0.997 | 0.997 | 0.997 |
| pxattn · attention lớp 1 | 0.997 | 0.734 | 0.966 |
| pxattn · attention lớp 2 | 0.997 | 0.399 | 0.553 |
| pxattn · cosine thô (không train) | 0.997 | 0.537 | 0.532 |

### Q2–Q3. Bên trong từ: âm vị j có đỉnh ở đúng đoạn âm vị j không (từ có âm vị MFA = g2p)

Âm vị g2p trùng hoàn toàn âm vị MFA: 83.9% số từ.

| bản đồ | đỉnh trong đoạn âm vị của chính nó (%) | đỉnh trong từ ±0.2 s (%) | lệch chỉ số âm vị (TB) | đơn điệu (%) | Spearman TB | đoạn audio → đúng hàng âm vị (%) | chance (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pmaxsim · cosine âm vị×audio | 52.1 | 98.7 | 0.94 | 92.4 | 0.97 | 47.4 | 19.1 |
| pxattn · attention lớp 1 | 64.5 | 97.0 | 0.67 | 79.5 | 0.92 | 64.0 | 19.1 |
| pxattn · attention lớp 2 | 15.6 | 39.1 | 1.30 | 22.4 | 0.24 | 35.0 | 19.1 |
| pxattn · cosine thô (không train) | 1.8 | 13.3 | 2.60 | 3.5 | 0.01 | 16.8 | 19.1 |

Mốc may rủi cho cột 1: đỉnh rơi ngẫu nhiên trong câu thì trúng đoạn âm vị của chính nó 1.4% số lần.

Thời lượng TB một đoạn âm vị (MFA) trong từ hiếm: 0.096 s = 4.8 frame.

### Q4. Probe tuyến tính 39 âm vị / frame (chia người nói 50/50, độ chính xác trên người nói chưa thấy)

| feature | số chiều | test-clean | test-other |
| --- | --- | --- | --- |
| whisper | 1280 | 79.7 | 71.3 |
| random256 | 256 | 68.0 | 60.7 |
| hou25 | 368 | 72.3 | 64.8 |
| pxattn | 256 | 66.7 | 58.6 |
| pmaxsim | 256 | 67.8 | 59.9 |
| đoán lớp nhiều nhất |  | 7.9 | 7.8 |

### Q3b. Ma trận loại âm vị audio (hàng) × loại âm vị của từ (cột), gộp 2 tập

| bản đồ | hàng có đường chéo lớn nhất (%) | hạng TB của đường chéo (1 = tốt nhất) | số hàng |
| --- | --- | --- | --- |
| pmaxsim · cosine âm vị×audio | 5.1 | 4.6 | 39 |
| pxattn · attention lớp 1 | 69.2 | 1.5 | 39 |
| pxattn · attention lớp 2 | 5.1 | 5.3 | 39 |
| pxattn · cosine thô (không train) | 0.0 | 21.2 | 39 |
