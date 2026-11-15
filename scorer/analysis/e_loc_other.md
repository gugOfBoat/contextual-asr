## test-other, list N2000, 2939 câu

### 1. Sạch@K

| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với pmaxsim_c4 [CI 95%] |
|---|---|---|---|---|
| pmaxsim_c4 | 87.5 | 96.2 | 97.4 |  |
| pmaxsim_loc | 88.3 | 96.2 | 97.3 | +0.9 [+0.2; +1.5] |
| pmono_c4 | 85.8 | 96.1 | 97.3 | -1.7 [-2.7; -0.7] |
| pmono_loc | 88.3 | 96.9 | 98.0 | +0.8 [-0.1; +1.8] |

### 2. Báo nhầm: từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu

| hệ | <5s (1522 câu) | 5-10s (963 câu) | 10-20s (414 câu) | ≥20s (40 câu) |
|---|---|---|---|---|
| pmaxsim_c4 | 0.3 | 0.6 | 1.0 | 1.3 |
| pmaxsim_loc | 0.3 | 0.5 | 0.8 | 1.1 |
| pmono_c4 | 0.2 | 0.4 | 0.6 | 0.9 |
| pmono_loc | 0.3 | 0.4 | 0.6 | 0.9 |

### 3. Sạch@1 theo việc Whisper không prompt nghe đúng / sai từ đó

| hệ | nghe sai | nghe đúng |
|---|---|---|
| pmaxsim_c4 | 64.5 (986) | 92.8 (4262) |
| pmaxsim_loc | 65.9 (986) | 93.5 (4262) |
| pmono_c4 | 64.4 (986) | 90.8 (4262) |
| pmono_loc | 69.2 (986) | 92.7 (4262) |

### 4–5. Đúng@chỗ và định vị (hệ có span)

| hệ | Sạch@1 cả câu: 1 / 2-4 / ≥5 từ | **Đúng@chỗ**: 1 / 2-4 / ≥5 từ | nhiễu cùng chỗ TB | sai số tâm trung vị (s) | % tâm trong span ±0.1 s |
|---|---|---|---|---|---|
| pmaxsim_c4 | 91.5 / 87.8 / 84.9 | 94.9 / 94.9 / 95.8 | 13.0 | 0.05 | 99.0 |
| pmaxsim_loc | 92.0 / 88.7 / 85.9 | 94.8 / 95.4 / 95.7 | 13.4 | 0.06 | 99.1 |
| pmono_c4 | 89.8 / 85.9 / 83.6 | 94.9 / 95.6 / 96.3 | 12.4 | 0.02 | 98.9 |
| pmono_loc | 91.8 / 89.0 / 85.4 | 95.9 / 96.2 / 97.5 | 11.7 | 0.02 | 99.0 |

### 6. Từ đúng bắt đầu ≥ 30 s: % không lọt top-50

- pmaxsim_c4: 0/6
- pmaxsim_loc: 0/6
- pmono_c4: 0/6
- pmono_loc: 0/6

### 7. Nhóm lỗi + từ gần âm ở cùng mức recall 98%

| hệ | lượt bị vượt | đồng âm % | gần âm % | khác hẳn % | K98 (số nhiễu cần lấy) | nhiễu gần âm lọt / câu ở K98 |
|---|---|---|---|---|---|---|
| pmaxsim_c4 | 589 | 2.4 | 7.3 | 90.3 | 20 | 1.19 |
| pmaxsim_loc | 551 | 2.5 | 7.6 | 89.8 | 20 | 1.19 |
| pmono_c4 | 684 | 1.9 | 6.3 | 91.8 | 19 | 1.38 |
| pmono_loc | 571 | 2.5 | 7.0 | 90.5 | 11 | 0.99 |
