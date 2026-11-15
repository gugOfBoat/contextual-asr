# Độ mới của "gate theo thời gian cho SpeechLLM" và kết quả phần R (11/11/2026)

> Phần R1–R4 của `PLAN.md`.
>
> Mức đọc: **[TV]** = toàn văn (HTML hoặc PDF) · **[TT]** = abstract hoặc tóm tắt từ trang tìm kiếm · **[đo]** = nhóm tự chạy.
>
> Phạm vi tra: khoảng 14 truy vấn và 15 paper. Mới đủ để nói "**chưa thấy**", chưa đủ để nói "chưa ai làm".

## 1. Bảng so sánh

| Paper | Vị trí của từ? | Vị trí dùng lúc nào | Loại ASR | ASR có train? | List / dữ liệu | Mức |
|---|---|---|---|---|---|---|
| **CTC-WS**, Andrusenko et al., Interspeech 2024 (2406.07096) | Có, khoảng frame (§2.1) | **Sau** greedy CTC: thay từ trùng span nếu điểm cao hơn (§2.1–2.2) | CTC, Hybrid TDT-CTC; không thử AED/LLM (§2.2, §3.1) | Không | 100–1000 từ; F 0.870 → 0.845 khi list tăng (§4.3) | TV |
| **Spike-triggered CB**, Huang et al., ASRU 2023 (2310.04657) | Có, frame spike CTC | **Trong** decode, chỉ ở frame spike | AED Conformer + CTC, tiếng Trung | Có: train mô-đun bias, base đóng băng | WenetSpeech; B-CER 37.49 → 11.76 | TV |
| WCTC-Biasing, Nakagome & Hentschel, IS 2025 (2506.01263) | Có, KWS ở lớp giữa | Bias các lớp encoder sau | CTC | Không | Tiếng Nhật | TT |
| **CB-Whisper**, Li et al. (2309.09552v2) | **Không**: CNN đọc vệt chéo trên ma trận cosine (TTS + Whisper encoder) → có/không cho cả câu (§2.2) | Chỉ **prompt** (naive hoặc "spoken form", §2.4) | Whisper small → large-v2 | Bản đầy đủ fine-tune; bản cắm ngoài thì Whisper đóng băng (Table 6) | Aishell hot word + nội bộ | TV |
| KG-Whisper, Shamsian et al. (2406.02649) | Không: AdaKWS ra vector có/không (§2) | Prompt; fine-tune decoder hoặc học prefix (§2.1–2.2) | Whisper | Có | 20 từ/câu (§3) | TV |
| **BR-ASR**, Gong et al., IS 2025 (2505.19179) | Không: pooling, rồi tích vô hướng | Top-k → prompt | Qwen-Audio, SLAM-ASR | Không (retriever train riêng) | IS21 N2000: B-WER 2.8 / 7.1 | TV |
| H-PRM (2508.18295) | Không rõ | Prompt cho Audio LLM | Paraformer, Audio LLM | ? | ? | TT |
| **Huang & Wang**, ASRU 2025 (2510.23849) | Không: chấm cả cụm theo toàn bộ encoder | Lọc list + bonus shallow fusion một giá trị/câu | Attention + CTC (ESPnet) | Base đóng băng, train decoder bias 30M | IS21 N100–2000; test-other B-WER khoảng 9.3–9.6 | TV |
| **Liu, Peng, Chng**, APSIPA 2025 (2508.17796) | Không | Cây tiền tố + shallow fusion **lúc decode**, không gate (thưởng mỗi token khớp) | Whisper-large-v3, beam 10, zero-shot | Không | IS21 **N1000**: test-other B-WER 17.55 → 9.90, U-WER 3.39 → 3.14 | TV |
| PAC, Fu et al. (2509.12647) | Không | Train LLM (LoRA + RL) phân biệt đồng âm | FireRed-LLM 7B | Có | IS21 N2000: B-WER 1.91 / 6.19 | TV |
| Novitasari et al., IBM (2604.12398) | Nhãn trên chữ, loss phụ (§2.3) | **Bỏ khi chạy** (§2.3) | Granite-Speech 8B | Có | list 200 | TV |
| PTC-Bias (2609.28727) | Có, [a, b] | Lọc list + sửa **sau** decode | Qwen3-ASR | Đầu âm vị | IS21 N2000: B-WER 4.10 / 7.76 | TV |
| Simul-Whisper (2406.10052) | Frame decoder đang nghe = argmax tổng alignment heads, lọc trung vị 7 (§2.1) | Chỉ để **dừng** decode khi streaming; không bias | Whisper base → large-v2 | Không | — | TV |
| SKIP-SALSA; Alibaba interleaved ASR-LLM | Căn token LLM với audio | Streaming, không bias | LLM-ASR | Có | — | TT |
| CTC-aligned KWS, Jin et al. (2406.07923) | Căn động bằng CTC từng frame (§3.1–3.2); không xuất start/end | Chỉ KWS, không ASR | — | — | LibriPhrase | TV |

## 2. Kết luận độ mới

1. **Bias đúng thời điểm đã có**, nhưng chỉ cho ASR có đầu ra theo frame:
   - CTC-WS sửa sau theo span;
   - Spike-triggered bias tại spike, phải train;
   - WCTC bias ở encoder.

   → Không claim "đầu tiên dùng vị trí để bias".
2. **Với Whisper / SpeechLLM đóng băng**, mọi paper đã tra chỉ đưa từ qua **prompt**, hoặc boost token **không theo thời gian** (Liu 2508.17796, Huang & Wang 2510.23849). Riêng PTC dùng thời gian, nhưng để sửa **sau** decode.

   → **Chưa thấy** ai gate boost theo vị trí từ lúc decoder tự hồi quy đóng băng đang sinh token.
3. **Lấy vị trí decoder từ chữ đã viết** (căn chữ đã viết với audio ngay giữa lúc decode) để bias: **chưa thấy**. Simul-Whisper dùng attention để dừng decode, không để bias.
4. **Vệt chéo trên ma trận giống nhau audio TTS × audio câu + CNN** là của **CB-Whisper** (trước Hou25). Không claim gì mới quanh "đọc vệt" nếu chỉ thay CNN. Điểm mới chỉ có thể là:
   - dạy vị trí khi train;
   - ra (p, span) dùng cho gate;
   - có "không từ nào" theo đoạn.
5. **Baseline phải thêm vào bảng chính:** boost lúc decode **không gate** đúng cách (cây tiền tố + shallow fusion, như Liu 2508.17796). Đây là đối thủ trực tiếp để chứng minh "**thời gian**" mới là thứ tạo khác biệt, không phải "boost lúc decode".

## 3. R3: số Hou25 tái hiện cạnh số công bố (test-clean / test-other)

| Hệ | WER | B-WER | Nguồn |
|---|---|---|---|
| Whisper-turbo không prompt | 3.2 / 5.3 | 10.4 / 19.7 | Hou25 Table 1 |
| Hou25 top-50, N2000 | 2.5 / 4.5 | 5.8 / 11.7 | Hou25 Table 1 |
| Không prompt (nhóm, wnorm) | 2.16 / 4.25 | 8.64 / 17.92 | `bench/results.md` |
| Top-50 tái hiện (nhóm, wnorm) | 1.87 / 3.41 | 5.13 / 9.84 | `bench/results.md` §6 |

- Mức giảm B-WER tương đối **khớp**:
  - Hou25: −44% (clean) / −41% (other).
  - Nhóm: −41% (clean) / −45% (other).
- Số tuyệt đối của nhóm thấp hơn vì khác chuẩn hoá: Hou25 không nêu normalizer, nhóm dùng wnorm. Trong paper phải báo cả hai.
- **Còn thiếu:** top-50 tái hiện chấm bằng plain cho test-clean.
- Hou25 xếp từ theo điểm tăng dần, từ điểm cao đứng cuối (Table 3). Cần kiểm `ctx_a1mean_N2000.json` có giữ đúng thứ tự này không.

## 4. R4: lỗi còn lại thuộc loại nào [đo]

`bench/error_breakdown.py`, tách thêm "đồng âm" = cách 0 âm vị theo g2p.
- test-other, list N2000, chuẩn hoá **plain**.
- % = phần trên 5350 lượt từ hiếm.

| Hệ | B-WER | Bỏ sót khi chọn | Đồng âm | Nghe giống (1–2 âm vị) | Ranh giới | Xoá | Thay bằng nhiễu | Khác | **Chèn thêm từ hiếm** |
|---|---|---|---|---|---|---|---|---|---|
| Không prompt | 18.92 | – | 3.01 | 7.23 | 1.53 | 0.86 | – | 6.28 | 0.00 |
| Top-50 → prompt (Hou25) | 10.06 | **2.65** | 0.80 | 2.88 | 0.65 | 0.69 | 0.26 | 2.11 | 0.00 |
| Đúng từ + gate MFA | 8.19 | 0 | 0.60 | 2.37 | 0.71 | 0.47 | 0 | 2.02 | **2.02** |

**Rút ra:**

1. Với prompt Hou25:
   - 26% lỗi còn lại là **bỏ sót khi chọn** (2.65 / 10.06).
   - 74% là từ **đã có trong prompt** mà vẫn sai, khớp Hoàng F12.
   - Trong đó nghe giống + đồng âm là 3.68, khoảng 37% tổng lỗi.
2. **Sửa 13/11:** các lần chạy gate ở E1 **không dùng prompt** (`bench/out/run1108_logs/final.sh`). Vì vậy so đúng phải là gate so với **không prompt**:
   - nghe giống 7.23 → 2.37;
   - đồng âm 3.01 → 0.60.

   → **Gate không cần prompt sửa lỗi viết ngang hoặc hơn prompt top-50** (2.88 / 0.80). Kết luận cũ "gate gần như không sửa lỗi viết" là **sai cách so**, vì đã so với prompt.

   Phần còn đúng: so với prompt top-50 thì chênh ở lỗi viết nhỏ, và gate thêm 2.02 lượt chèn lặp. Kết luận cũ (giữ lại để đối chiếu) — **Gate (oracle) gần như không sửa lỗi viết:**
   - nghe giống 2.88 → 2.37;
   - đồng âm 0.80 → 0.60;
   - khác 2.11 → 2.02;
   - ranh giới không đổi.

   Phần thắng 10.06 → 8.19 chủ yếu do **list đúng**: mất 2.65 bỏ sót khi chọn. Phần này bị trừ lại bởi **2.02 lượt chèn thêm từ hiếm** (lặp chữ) do boost gây ra.

   → **Giả định A1 ("gate sửa lỗi viết") hiện CHƯA được ủng hộ.** So sánh "7.90 so với 9.69" bị lẫn biến nhiễu: hai bên dùng list khác nhau.
3. Đồng âm hoàn toàn chỉ 0.80 khi có prompt (không prompt là 3.01). Nghĩa là **prompt đã giải quyết phần lớn đồng âm**, vì nó cho chính tả. Phần audio không thể phân biệt nhỏ.

**Hệ quả cho kế hoạch:**
- **G2 phải so cùng list:** prompt (top-50 ∪ từ đúng) có và không có gate. Không so oracle-gate với top-50-prompt nữa.
- **B3 (chặn chèn lặp) trở thành điều kiện bắt buộc.** Mỗi từ chỉ được boost một lần trong span của nó, không boost lại sau khi đã viết xong.
- Nếu G2 vẫn cho thấy gate không sửa được lỗi viết, đóng góp chuyển trọng tâm:
  - scorer chọn ít từ, **giảm bỏ sót** (2.65 điểm có thể lấy lại);
  - prompt gọn;
  - gate chỉ là phụ.

## Nguồn

- CTC-WS: https://arxiv.org/abs/2406.07096
- Spike-triggered CB: https://arxiv.org/abs/2310.04657
- WCTC-Biasing: https://arxiv.org/abs/2506.01263
- CB-Whisper: https://arxiv.org/abs/2309.09552
- KG-Whisper: https://arxiv.org/abs/2406.02649
- BR-ASR: https://arxiv.org/abs/2505.19179
- H-PRM: https://arxiv.org/abs/2508.18295
- Huang & Wang: https://arxiv.org/abs/2510.23849
- Liu et al.: https://arxiv.org/abs/2508.17796
- PAC: https://arxiv.org/abs/2509.12647
- IBM: https://arxiv.org/abs/2604.12398
- Simul-Whisper: https://arxiv.org/abs/2406.10052
- CTC-aligned KWS: https://arxiv.org/abs/2406.07923
- LLM-ForcedAligner: https://arxiv.org/abs/2601.18220
