## test-other: từ hiếm có được dùng đúng chỗ? (% trên số lượt từ hiếm của nhóm)


### Câu có 1 từ hiếm (837 câu)

| hệ | lượt | đúng | sai chỗ: từ đúng khác | sai chỗ: từ nhiễu | lỗi khác | chèn từ hiếm của câu |
|---|---|---|---|---|---|---|
| none | 852 | 76.17 | 0.00 | 0.00 | 23.83 | 0.00 |
| prompt_oracle | 852 | 88.62 | 0.00 | 0.00 | 11.38 | 68.54 |
| prompt_top50 | 852 | 88.97 | 0.00 | 0.47 | 10.56 | 0.00 |
| boost_khong_gate | 852 | 92.37 | 0.00 | 0.00 | 7.63 | 12.09 |
| gate_MFA | 852 | 91.55 | 0.00 | 0.00 | 8.45 | 0.00 |
| gate_S1 | 852 | 90.49 | 0.00 | 0.00 | 9.51 | 0.00 |

### Câu có 2-4 từ hiếm (1048 câu)

| hệ | lượt | đúng | sai chỗ: từ đúng khác | sai chỗ: từ nhiễu | lỗi khác | chèn từ hiếm của câu |
|---|---|---|---|---|---|---|
| none | 2828 | 81.26 | 0.00 | 0.00 | 18.74 | 0.04 |
| prompt_oracle | 2828 | 90.95 | 0.07 | 0.00 | 8.98 | 0.07 |
| prompt_top50 | 2828 | 89.96 | 0.00 | 0.21 | 9.83 | 0.07 |
| boost_khong_gate | 2828 | 94.80 | 0.25 | 0.00 | 4.95 | 24.43 |
| gate_MFA | 2828 | 94.70 | 0.04 | 0.00 | 5.27 | 3.82 |
| gate_S1 | 2828 | 93.95 | 0.07 | 0.00 | 5.98 | 3.82 |

### Câu có ≥5 từ hiếm (256 câu)

| hệ | lượt | đúng | sai chỗ: từ đúng khác | sai chỗ: từ nhiễu | lỗi khác | chèn từ hiếm của câu |
|---|---|---|---|---|---|---|
| none | 1670 | 83.29 | 0.06 | 0.00 | 16.65 | 0.00 |
| prompt_oracle | 1670 | 92.16 | 0.12 | 0.00 | 7.72 | 0.00 |
| prompt_top50 | 1670 | 90.42 | 0.00 | 0.30 | 9.28 | 0.00 |
| boost_khong_gate | 1670 | 93.47 | 0.36 | 0.00 | 6.17 | 6.59 |
| gate_MFA | 1670 | 93.53 | 0.00 | 0.00 | 6.47 | 0.12 |
| gate_S1 | 1670 | 93.11 | 0.12 | 0.00 | 6.77 | 0.18 |

## Scorer: từ đúng có đứng đầu không — cả câu so với tại chỗ của nó (Hou25 A1 top-50 + vị trí vệt S1, test-other)

| nhóm câu | lượt (từ đúng trong top-50) | đứng đầu so với MỌI nhiễu của câu | đứng đầu so với nhiễu CÙNG CHỖ | TB nhiễu cùng chỗ |
|---|---|---|---|---|
| 1 từ | 828 | 83.9 | 90.2 | 14.1 |
| 2-4 từ | 2686 | 79.0 | 90.6 | 9.2 |
| ≥5 từ | 1538 | 68.1 | 91.0 | 5.1 |
