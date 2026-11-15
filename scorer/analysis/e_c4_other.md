## test-other, list N2000, 2939 câu

### 1. Sạch@K

| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với pmaxsim_c4 [CI 95%] |
|---|---|---|---|---|
| pmaxsim_win | 85.7 | 95.2 | 96.8 | -1.7 [-2.7; -0.8] |
| pmaxsim_c4 | 87.5 | 96.2 | 97.4 |  |
| pmono_c4 | 85.8 | 96.1 | 97.3 | -1.7 [-2.7; -0.7] |

### 2. Báo nhầm: từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu

| hệ | <5s (1522 câu) | 5-10s (963 câu) | 10-20s (414 câu) | ≥20s (40 câu) |
|---|---|---|---|---|
| pmaxsim_win | 0.5 | 0.8 | 1.2 | 1.6 |
| pmaxsim_c4 | 0.3 | 0.6 | 1.0 | 1.3 |
| pmono_c4 | 0.2 | 0.4 | 0.6 | 0.9 |

### 3. Sạch@1 theo việc Whisper không prompt nghe đúng / sai từ đó

| hệ | nghe sai | nghe đúng |
|---|---|---|
| pmaxsim_win | 61.3 (986) | 91.4 (4262) |
| pmaxsim_c4 | 64.5 (986) | 92.8 (4262) |
| pmono_c4 | 64.4 (986) | 90.8 (4262) |

### 4–5. Đúng@chỗ và định vị (hệ có span)

| hệ | Sạch@1 cả câu: 1 / 2-4 / ≥5 từ | **Đúng@chỗ**: 1 / 2-4 / ≥5 từ | nhiễu cùng chỗ TB | sai số tâm trung vị (s) | % tâm trong span ±0.1 s |
|---|---|---|---|---|---|
| pmaxsim_win | 87.2 / 86.2 / 84.2 | 92.6 / 94.1 / 96.0 | 12.5 | 0.05 | 98.8 |
| pmaxsim_c4 | 91.5 / 87.8 / 84.9 | 94.9 / 94.9 / 95.8 | 13.0 | 0.05 | 99.0 |
| pmono_c4 | 89.8 / 85.9 / 83.6 | 94.9 / 95.6 / 96.3 | 12.4 | 0.02 | 98.9 |

### 6. Từ đúng bắt đầu ≥ 30 s: % không lọt top-50

- pmaxsim_win: 0/6
- pmaxsim_c4: 0/6
- pmono_c4: 0/6

### 7. Nhóm lỗi + từ gần âm ở cùng mức recall 98%

| hệ | lượt bị vượt | đồng âm % | gần âm % | khác hẳn % | K98 (số nhiễu cần lấy) | nhiễu gần âm lọt / câu ở K98 |
|---|---|---|---|---|---|---|
| pmaxsim_win | 676 | 2.1 | 6.5 | 91.4 | 28 | 1.40 |
| pmaxsim_c4 | 589 | 2.4 | 7.3 | 90.3 | 20 | 1.19 |
| pmono_c4 | 684 | 1.9 | 6.3 | 91.8 | 19 | 1.38 |
