## test-other, list N2000, 2939 câu

### 1. Sạch@K

| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với hou25_win [CI 95%] |
|---|---|---|---|---|
| hou25_win | 72.6 | 87.1 | 90.9 |  |
| pmaxsim_win | 85.7 | 95.2 | 96.8 | +13.1 [+11.5; +14.7] |
| pmaxsim960_win | 87.5 | 96.2 | 97.5 | +14.8 [+13.3; +16.3] |

### 2. Báo nhầm: từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu

| hệ | <5s (1522 câu) | 5-10s (963 câu) | 10-20s (414 câu) | ≥20s (40 câu) |
|---|---|---|---|---|
| hou25_win | 1.4 | 2.5 | 6.1 | 12.4 |
| pmaxsim_win | 0.5 | 0.8 | 1.2 | 1.6 |
| pmaxsim960_win | 0.7 | 1.2 | 1.9 | 2.5 |

### 3. Sạch@1 theo việc Whisper không prompt nghe đúng / sai từ đó

| hệ | nghe sai | nghe đúng |
|---|---|---|
| hou25_win | 45.9 (986) | 78.8 (4262) |
| pmaxsim_win | 61.3 (986) | 91.4 (4262) |
| pmaxsim960_win | 63.1 (986) | 93.1 (4262) |

### 4–5. Đúng@chỗ và định vị (hệ có span)

| hệ | Sạch@1 cả câu: 1 / 2-4 / ≥5 từ | **Đúng@chỗ**: 1 / 2-4 / ≥5 từ | nhiễu cùng chỗ TB | sai số tâm trung vị (s) | % tâm trong span ±0.1 s |
|---|---|---|---|---|---|
| hou25_win | 81.1 / 75.8 / 62.9 | 90.2 / 91.4 / 92.1 | 8.9 | 0.69 | 44.0 |
| pmaxsim_win | 87.2 / 86.2 / 84.2 | 92.6 / 94.1 / 96.0 | 12.5 | 0.05 | 98.8 |
| pmaxsim960_win | 90.9 / 87.7 / 85.3 | 94.7 / 94.9 / 95.8 | 12.7 | 0.07 | 98.8 |

### 6. Từ đúng bắt đầu ≥ 30 s: % không lọt top-50

- hou25_win: 0/6
- pmaxsim_win: 0/6
- pmaxsim960_win: 0/6

### 7. Nhóm lỗi + từ gần âm ở cùng mức recall 98%

| hệ | lượt bị vượt | đồng âm % | gần âm % | khác hẳn % | K98 (số nhiễu cần lấy) | nhiễu gần âm lọt / câu ở K98 |
|---|---|---|---|---|---|---|
| hou25_win | 1210 | 0.8 | 3.4 | 95.8 | 129 | 2.49 |
| pmaxsim_win | 676 | 2.1 | 6.5 | 91.4 | 28 | 1.40 |
| pmaxsim960_win | 591 | 2.2 | 8.5 | 89.3 | 17 | 0.97 |
