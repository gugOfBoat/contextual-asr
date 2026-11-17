> Chạy 17/11 (CPU, local): `python analysis/nms_at_place.py test-other N2000 ...` trên output 14–15/11. Mục đích: so precision của các scorer ở **cùng ngưỡng / cùng mức giữ từ đúng**, thay cho so với Hou25 top-50. hou25 = hou25_win (Recall@1 72.6); hou25_e2 = bản trung bình train cùng lượt (73.3); hou25_diag = vệt chéo (82.9); pdual2_dl = scorer đề xuất một model (89.3). Số từ hiếm trung bình: 5248 lượt / 2939 câu = 1.79 từ/câu.

## test-other N2000: ngưỡng p ≥ τ, trước → sau khi tranh nhau tại chỗ

| hệ | τ | recall từ đúng % | nhiễu / câu | precision % |
|---|---|---|---|---|
| hou25 | 0.3 | 91.2 → 75.1 | 6.06 → 1.95 | 21.2 → 40.7 |
| hou25 | 0.5 | 87.0 → 73.0 | 2.60 → 1.06 | 37.4 → 55.2 |
| hou25 | 0.7 | 81.7 → 69.4 | 1.06 → 0.52 | 57.9 → 70.3 |
| hou25 | 0.9 | 65.4 → 57.4 | 0.20 → 0.13 | 85.1 → 89.1 |
| hou25_e2 | 0.3 | 90.4 → 82.0 | 4.53 → 1.56 | 26.3 → 48.4 |
| hou25_e2 | 0.5 | 86.6 → 79.6 | 1.91 → 0.86 | 44.7 → 62.2 |
| hou25_e2 | 0.7 | 81.1 → 75.4 | 0.80 → 0.44 | 64.4 → 75.2 |
| hou25_e2 | 0.9 | 66.0 → 62.6 | 0.19 → 0.14 | 85.8 → 89.0 |
| hou25_diag | 0.3 | 93.2 → 90.2 | 1.81 → 0.90 | 47.9 → 64.1 |
| hou25_diag | 0.5 | 90.7 → 88.3 | 0.94 → 0.59 | 63.4 → 72.9 |
| hou25_diag | 0.7 | 87.2 → 85.3 | 0.49 → 0.38 | 76.0 → 80.1 |
| hou25_diag | 0.9 | 74.5 → 73.5 | 0.17 → 0.15 | 88.6 → 89.5 |
| pdual2_dl | 0.3 | 96.3 → 95.0 | 1.23 → 0.73 | 58.4 → 69.9 |
| pdual2_dl | 0.5 | 93.3 → 92.4 | 0.40 → 0.27 | 80.8 → 85.9 |
| pdual2_dl | 0.7 | 85.1 → 84.5 | 0.13 → 0.10 | 91.9 → 93.5 |
| pdual2_dl | 0.9 | 47.4 → 47.2 | 0.01 → 0.01 | 98.3 → 98.5 |
