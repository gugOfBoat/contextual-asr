
## test-clean: độ trải đỉnh âm vị trong từ (từ ≥ 3 âm vị)

| bản đồ | số đỉnh khác nhau / số âm vị | (đỉnh cuối − đỉnh đầu) / độ dài từ | từ mà ≥ 80% âm vị chung 1 đỉnh (±1 frame) (%) | tăng ngặt (%) |
| --- | --- | --- | --- | --- |
| pmaxsim/cos | 0.38 | 0.25 | 22.1 | 0.0 |
| pxattn/att1 | 0.66 | 0.50 | 1.6 | 4.3 |
| pxattn/att2 | 0.62 | -0.51 | 5.5 | 0.1 |
| pxattn/cos_raw | 0.90 | -0.63 | 0.7 | 0.7 |

## test-clean: định vị (streak) theo xác suất của scorer (từ đúng)

| bản đồ | p ≥ 0.5: n / median \|err\| / trúng span % | p < 0.5: n / median / trúng % |
| --- | --- | --- |
| hou25/S1 | 1023 / 0.06 / 99.3 | 62 / 0.07 / 91.9 |
| pmaxsim/cos | 1049 / 0.04 / 99.9 | 36 / 0.04 / 97.2 |
| pxattn/att1 | 1034 / 0.03 / 99.5 | 51 / 0.03 / 100.0 |

## test-other: độ trải đỉnh âm vị trong từ (từ ≥ 3 âm vị)

| bản đồ | số đỉnh khác nhau / số âm vị | (đỉnh cuối − đỉnh đầu) / độ dài từ | từ mà ≥ 80% âm vị chung 1 đỉnh (±1 frame) (%) | tăng ngặt (%) |
| --- | --- | --- | --- | --- |
| pmaxsim/cos | 0.38 | 0.23 | 28.4 | 0.0 |
| pxattn/att1 | 0.66 | 0.45 | 2.4 | 6.2 |
| pxattn/att2 | 0.62 | 0.04 | 9.1 | 0.4 |
| pxattn/cos_raw | 0.90 | -0.14 | 0.7 | 2.0 |

## test-other: định vị (streak) theo xác suất của scorer (từ đúng)

| bản đồ | p ≥ 0.5: n / median \|err\| / trúng span % | p < 0.5: n / median / trúng % |
| --- | --- | --- |
| hou25/S1 | 857 / 0.06 / 98.5 | 122 / 0.06 / 84.4 |
| pmaxsim/cos | 890 / 0.03 / 99.9 | 89 / 0.04 / 93.3 |
| pxattn/att1 | 864 / 0.03 / 99.9 | 115 / 0.04 / 93.9 |
