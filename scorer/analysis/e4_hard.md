## test-other, list hard_N2000, 2939 câu

### 1. Sạch@K

| hệ | Sạch@1 | Sạch@5 | Sạch@10 | Δ Sạch@1 so với hou25_win [CI 95%] |
|---|---|---|---|---|
| hou25_win | 51.4 | 80.3 | 87.2 |  |
| pmaxsim_win | 61.2 | 87.5 | 92.6 | +9.7 [+7.9; +11.4] |

### 2. Báo nhầm: từ nhiễu p ≥ 0.5 mỗi câu, theo độ dài câu

| hệ | <5s (1522 câu) | 5-10s (963 câu) | 10-20s (414 câu) | ≥20s (40 câu) |
|---|---|---|---|---|
| hou25_win | 2.7 | 5.3 | 11.8 | 20.5 |
| pmaxsim_win | 2.6 | 5.4 | 9.5 | 13.7 |

### 3. Sạch@1 theo việc Whisper không prompt nghe đúng / sai từ đó

| hệ | nghe sai | nghe đúng |
|---|---|---|
| hou25_win | 28.3 (986) | 56.8 (4262) |
| pmaxsim_win | 36.1 (986) | 67.0 (4262) |

### 4–5. Đúng@chỗ và định vị (hệ có span)

| hệ | Sạch@1 cả câu: 1 / 2-4 / ≥5 từ | **Đúng@chỗ**: 1 / 2-4 / ≥5 từ | nhiễu cùng chỗ TB | sai số tâm trung vị (s) | % tâm trong span ±0.1 s |
|---|---|---|---|---|---|
| hou25_win | 71.0 / 55.1 / 35.2 | 83.5 / 82.9 / 80.5 | 10.3 | 0.69 | 43.9 |
| pmaxsim_win | 77.8 / 63.6 / 48.6 | 82.0 / 83.5 / 85.5 | 15.5 | 0.05 | 98.9 |

### 6. Từ đúng bắt đầu ≥ 30 s: % không lọt top-50

- hou25_win: 0/6
- pmaxsim_win: 0/6

### 7. Nhóm lỗi + từ gần âm ở cùng mức recall 98%

| hệ | lượt bị vượt | đồng âm % | gần âm % | khác hẳn % | K98 (số nhiễu cần lấy) | nhiễu gần âm lọt / câu ở K98 |
|---|---|---|---|---|---|---|
| hou25_win | 2282 | 0.5 | 21.0 | 78.5 | 159 | 15.76 |
| pmaxsim_win | 1924 | 0.5 | 21.3 | 78.2 | 58 | 17.52 |
