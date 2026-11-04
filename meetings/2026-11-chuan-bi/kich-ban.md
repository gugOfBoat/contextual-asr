# Kịch bản nói — 3 ý (khớp slide `hop-thay.pdf`)

> Nguồn: `scorer/runs/2026-10-28/pipe_hou25tab.sh`, `bench/run_whisper.py`, `scorer/inference/rank.py`, `scorer/training/train.py`, `scorer/runs/2026-10-28/score_plain.txt`, `scorer/results.md` §6–10, Hou25 PDF §2–4, brief 23/10.

## Slide 2 — mở đầu
> "Hôm nay em báo cáo 3 ý. Một: em dựng lại đúng thí nghiệm của Hou25, chỉ thay scorer, và scorer âm vị của nhóm thắng Hou25 cả 8 ô. Hai: vì sao chọn âm vị và attention, có thực nghiệm so các cách. Ba: chỉ scorer thì đóng góp còn mỏng, nên em đề xuất cho scorer trả thêm vị trí của từ để đưa vào phía mô hình nghe, và muốn xin ý kiến thầy về hướng này."

---

## Ý 1 — Tái hiện Hou25

### Slide 3 — thiết lập
> "Em cố gắng giữ **mọi thứ giống paper**, chỉ khác đúng một chỗ là scorer.
> - **Dữ liệu test:** LibriSpeech test-clean 2620 câu, test-other 2939 câu.
> - **Bias list:** benchmark IS21 mà Hou25 dùng. Mỗi câu có các từ hiếm thật sự có trong câu, trung bình khoảng 2 từ, trộn với N từ nhiễu ngẫu nhiên. N là 100, 500, 1000, 2000. Em dùng nguyên file list gốc của IS21.
> - **ASR:** Whisper-large-v3-turbo, không train gì. List đưa vào qua **prompt** dạng `w1, w2, w3…`, Hou25 gọi là prompt naive. Whisper chỉ nhận prompt khoảng 223 token, list dài hơn bị cắt, chỉ giữ phần cuối.
> - **Chọn từ:** scorer chấm điểm cả N từ, giữ **50 từ điểm cao nhất**, xếp **tăng dần** để từ chắc nhất đứng cuối, sát chỗ Whisper bắt đầu chép. Đúng như Hou25.
> - **Train scorer:** train-clean-100, kiểm tra trên dev-clean; từ sai lấy 40% gần chính tả với từ đúng, 60% ngẫu nhiên. Encoder audio là Whisper-large-v3 đóng băng. Đều giống paper.
> - **Chấm:** scorer chính thức của IS21. WER là lỗi trên mọi từ; **B-WER** là lỗi chỉ trên các từ hiếm của câu, đây là số chính.
>
> **Chỗ khác:** Hou25 biểu diễn từ bằng **audio đọc từ đó bằng TTS**, còn em dùng **chuỗi âm vị** từ g2p. Vài chỗ paper không ghi (normalizer, cách decode) em chọn plain và greedy. lr của em là 2e-4, paper là 5e-5.
> Số Hou25 em chép từ paper; các hàng còn lại em chạy thật, 44 lượt decode trên 1 card 4090."

### Slide 4 — Bảng 1
> "Bảng 1 hỏi: **list càng dài thì Whisper càng tệ, scorer có cứu được không?** Mỗi câu em cho Whisper 3 kiểu đầu vào: (a) không prompt, (b) nguyên list N từ, (c) 50 từ do scorer chọn.
>
> Dòng **(b)** chỉ giúp ở N=100: B-WER 9.43 xuống 8.04. Từ N=500 trở lên gần như bằng không prompt. Lý do chính là Whisper chỉ nhận prompt khoảng 223 token, list dài bị cắt chỉ giữ phần cuối: em đếm thì ở N=1000 và N=2000 chỉ còn 2 đến 6% từ đúng sót lại trong prompt, ở N=100 còn khoảng 56%. Paper cũng thấy y như vậy, nhưng không nói list có bị cắt không.
>
> Dòng **(c)**: scorer của nhóm kéo B-WER còn 4.57 / 8.92 ở N=100 và 5.29 / 10.00 ở N=2000, tức **giảm khoảng một nửa**. So với scorer của Hou25 (5.4 / 10.0 và 5.8 / 11.7) thì **thấp hơn ở cả 8 ô**, rõ nhất trên test-other, thấp hơn 1.1 đến 1.7 điểm. WER chung cũng thấp hơn, nghĩa là không đánh đổi các từ thường."

### Slide 5 — Ablation câu có >1 từ hiếm
> "Em tách riêng các câu có hơn 1 từ hiếm, khoảng một nửa tập: 1289 câu clean, 1304 câu other. Paper không có lát cắt này.
> - Với scorer top-50, kết quả cải thiện đều y như trên cả tập. Ở nhóm câu này, pxattn N100 còn ngang oracle: 4.37 / 8.47 so với 4.33 / 8.63.
> - Hai cấu hình tệ trên cả tập là oracle và lọc p ≥ 0.5 với prompt naive. Sang nhóm câu này thì bình thường lại: oracle 4.33 / 8.63. Nghĩa là chỗ hỏng nằm ở các câu chỉ có 0–1 từ hiếm, nơi prompt chỉ còn 1 từ và Whisper lặp chữ. Em giải thích kỹ ở slide Bảng 2."

### Slide 6 — Bảng 2
> "Bảng 2 hỏi một câu khác: **đưa list cho Whisper kiểu nào thì nó dùng tốt nhất?** Bảng này **không dùng scorer**. Em giữ list 100 từ như cũ, chỉ đổi thứ Whisper nhận được, có 4 kiểu, đúng như 4 hàng của paper:
> 1. không prompt, để làm mốc;
> 2. **chỉ các từ đúng** của câu, ví dụ chỉ mỗi chữ `yearly`. Đây giống như có một scorer hoàn hảo;
> 3. nguyên list 100 từ, nối bằng dấu phẩy;
> 4. nguyên list đó nhưng bọc trong một câu nói tự nhiên: 'The topic of today's speech is, ah, …'.
>
> Bên trái là số trong paper, bên phải là em tự dựng lại đúng thí nghiệm đó và tự chạy.
>
> **Hàng 2 so với hàng 1:** chỉ đưa từ đúng thì từ hiếm được chép đúng hơn hẳn, B-WER test-clean từ 9.43 xuống 5.12. Nhưng **WER chung lại tăng**. Paper cũng thấy vậy, của em tăng mạnh hơn nhiều, em giải thích ở slide sau.
>
> **Hàng 3 so với hàng 4:** hai cách viết prompt gần như ngang nhau. Paper thấy kiểu 3 hơn một chút, em chưa thấy rõ bên nào hơn."

### Slide 7 — WER 20% từ đâu ra
> "Con số WER 20% nghe rất lớn, nhưng không phải Whisper chép kém đều. Ví dụ câu 'is thee going to the yearly meeting…', prompt chỉ có chữ `yearly`. Whisper chép đúng cả câu, rồi **lặp 'the girls the girls…'** cho tới hết độ dài cho phép, thừa 429 từ.
>
> Em đếm cả tập: chỉ khoảng **30 câu** trên 2620 bị như vậy, và đều là câu có prompt **đúng 1 từ**. Mỗi câu thừa khoảng 430 từ, cộng lại thành khoảng 9 nghìn từ thừa, tức khoảng 18% số từ của cả tập. WER tính cả từ chèn thừa, nên WER đi từ 2.6 lên 20. Các câu có từ 2 từ hiếm trở lên thì chạy bình thường. Paper chỉ tăng từ 3.2 lên 4.9, có thể họ decode có cơ chế chống lặp, em chưa kiểm.
>
> **Ý nghĩa:** đưa **đúng** từ cần thiết mà mô hình vẫn hỏng. Vậy chọn từ thôi chưa đủ, **cách mô hình dùng list** là một nút thắt riêng. Qwen3-ASR thì không bị lặp, nên cách đưa list còn phụ thuộc từng mô hình."

### Slide 8 — giới hạn (nói nhanh)
> "Em nói trước mấy điểm yếu: số Hou25 là số paper nên chưa có khoảng tin cậy khi so với họ; baseline em chạy lệch paper khoảng ±1 theo cả 2 chiều, có thể do cấu hình decode, em chưa kiểm; lr khác paper; và còn vài ô chưa chạy, như oracle + spoken, Bảng 3, 4 và cùng lưới trên Qwen."

---

## Ý 2 — Vì sao âm vị + attention

### Slide 9 — kiến trúc
> "Scorer nhận audio của câu và một từ trong list, trả ra xác suất từ đó được nói.
> - **Hàng trên (audio):** encoder Whisper đóng băng biến audio thành chuỗi vector, khoảng 20 ms một vector, rồi qua 1 lớp Transformer nhỏ.
> - **Hàng dưới (từ):** g2p đổi chữ thành âm vị, ví dụ *cabinet* thành K AE B AH N AH T, rồi qua 2 lớp Transformer.
> - **So:** mỗi âm vị dùng cross-attention 'tìm' xem mình nằm ở đâu trong chuỗi vector audio, làm 2 lần. Gộp lại, qua MLP, ra xác suất.
> Bản đưa vào ASR là cột giữa, pxattn."

### Slide 10 — 4 cách để trả lời "từ cabinet có trong câu không"
> "Để giải thích vì sao chọn âm vị và attention, em so 4 cách cùng trả lời một câu hỏi: từ *cabinet* có được nói trong câu không.
> 1. **So chữ với chữ:** cho ASR chép câu ra chữ trước, đổi chữ sang âm vị, rồi tìm đoạn nào gần giống *cabinet*.
> 2. **So tiếng với tiếng**, cách của Hou25: cho máy đọc chữ *cabinet* bằng TTS, rồi so đoạn tiếng đó với audio câu nói.
> 3. **Từng âm tự dò:** mỗi âm K, AE, B… tự tìm chỗ giống nhất trong audio rồi lấy trung bình, không quan tâm thứ tự.
> 4. **Các âm bàn nhau rồi dò**, bản của nhóm: các âm nhìn nhau trước, biết K đứng trước AE, rồi cùng tìm trong audio.
>
> Em đo bằng 2 câu hỏi đơn giản trên list 2000 từ: **tìm được**, tức cứ 100 từ đúng thì bao nhiêu từ lọt vào 50 từ được chọn, cao là tốt; và **bị lừa**, tức cứ 100 từ nghe giống từ đúng có trong list (như *cabinets*) thì bao nhiêu từ lọt vào, thấp là tốt."

### Slide 11 — kết quả
> "Cách 1, so chữ với chữ, **kém cả hai**: tìm được 95.4, bị lừa 35.7. Lý do là từ hiếm mà ASR nghe sai thì đã mất ngay lúc chép ra chữ, không còn gì để so.
> Cách 2 của Hou25 **ít bị lừa nhất** (17.6) nhưng **bỏ sót nhiều nhất** (tìm được 95.4), lại chậm và tốn chỗ.
> Cách 3 **tìm giỏi nhất** (98.4) nhưng hay bị lừa (25.5), vì không biết thứ tự âm: *stone* và *notes* có cùng các âm.
> Cách 4 của nhóm **cân bằng nhất**: vẫn tìm được 97.7, bị lừa giảm còn 21.6, ngang PTC. Vì vậy em dùng cách 4 để đưa vào ASR.
> Gọn lại: dùng âm vị thay TTS thì tìm được nhiều hơn và rẻ hơn khoảng 200 lần; cho các âm nhìn nhau bằng attention thì bớt bị lừa."

### Slide 12 — vì sao không chốt audio thành âm vị như PTC
> "Thầy có thể hỏi sao không cho audio qua một model nhận dạng âm vị như PTC.
> - Thứ nhất, để **so công bằng với Hou25**: giữ nguyên phía audio như paper, chỉ đổi phía từ, thì Bảng 1 chứng minh đúng một điều là âm vị tốt hơn TTS.
> - Thứ hai, **chốt cứng mất thông tin**: hypmatch kém rõ; em cũng thử cho scorer nhìn thêm bản chép lần 1 (hxattn) thì kém hẳn, B-WER 6.58 / 14.07 so với 4.78 / 9.71, có vẻ mô hình học tắt theo bản chép thay vì nghe audio.
> - Thứ ba, checkpoint WavLM âm vị của PTC không có link tải.
>
> Cái giá là attention chưa ép thứ tự âm chặt như Viterbi của PTC, nên PTC còn hơn ở chỉ số số từ phải giữ để không sót 1% (16.9 so với 64). Đây là giả thuyết của em, chưa kiểm. Hướng sửa: thêm một head CTC âm vị lên encoder Whisper, vừa có thứ tự, vừa cho ra **vị trí** của từ, cái này dùng cho ý 3."

---

## Ý 3 — Đi tiếp

### Slide 13 — vì sao chỉ scorer là chưa đủ
> "Em nghĩ nếu bài chỉ là một scorer tốt hơn thì đóng góp mỏng:
> - khâu chọn từ đã gần trần: tìm được 98.6–98.8% so với PTC 99.3%;
> - Hou25, PTC, BR-ASR đều làm 'chọn từ rồi nhét vào prompt';
> - buổi trước em đã phân rã: sau khi lọc, khoảng một nửa lỗi còn lại là từ **đã có trong list** mà mô hình vẫn chép sai hoặc bị từ nhiễu kéo;
> - và Bảng 2: đưa đúng từ mà Whisper vẫn hỏng.
>
> Em nghĩ gốc của vấn đề: prompt bằng chữ chỉ nói 'câu này có thể có từ X', **không nói X ở đâu**. Mô hình phải tự đoán chỗ, nên dễ chèn sai chỗ hoặc bỏ qua."

### Slide 14 — đề xuất biasing có vị trí
> "Đề xuất: scorer trả thêm **khoảng thời gian**, ví dụ *cabinet*, điểm 0.97, ở 2.1 đến 2.6 giây. Có 3 cách đưa vị trí vào mô hình:
> - **A:** vẫn prompt chữ nhưng kèm thời gian. Không cần train, nhưng mô hình có thể không hiểu mốc thời gian.
> - **B:** chèn gợi ý **thẳng vào chuỗi audio**: đặt embedding của từ ngay sau các frame 2.1–2.6 giây trong đầu vào của LLM. Đây là đúng chỗ mô hình 'nghe', nhưng cần train một adapter hoặc LoRA nhỏ.
> - **C:** cắt đoạn 2.1–2.6 giây và decode lại riêng đoạn đó với prompt chỉ có từ đó. Không train, nhưng phải xử lý ranh giới đoạn, và prompt 1 từ thì có thể lặp chữ như Bảng 2.
>
> Vì sao đánh đúng nút thắt: không bias ở chỗ không có từ thì bớt chèn sai; chỉ rõ chỗ cần sửa thì bớt chép sai từ đã có trong list.
> Vị trí lấy từ attention của pxattn lên các frame, cái này có sẵn nhưng chưa đo độ chính xác, hoặc từ head CTC ở ý 2.
> Về độ mới, em mới kiểm sơ bộ: PTC có thời gian nhưng chỉ để sửa sau decode; Novitasari có tagger vị trí nhưng bỏ đi khi suy luận. Em sẽ quét kỹ hơn."

### Slide 15 — thí nghiệm đo trần
> "Trước khi làm, em muốn đo trần trong khoảng 1 ngày GPU: dùng **vị trí thật** lấy bằng forced aligner trên câu đúng, coi như scorer định vị hoàn hảo, rồi so trên Qwen3-ASR đóng băng: oracle bằng prompt chữ; oracle thêm 5, 20, 50 từ nhiễu; oracle thêm từ nghe giống; và oracle **có vị trí** theo cách A và C, không cần train.
> Nếu vị trí lấp được phần đáng kể thì làm hướng này, và đóng góp gồm 3 phần: scorer rẻ, định vị (đo độ chính xác, chưa ai đo) và biasing có vị trí. Nếu không thì em viết theo hướng chọn từ cộng phân tích 2 nút thắt, và tìm đóng góp khác."

### Slide 16 — cần thầy quyết
> "Em xin ý kiến thầy 5 điểm: có dừng tối ưu scorer để chuyển trọng tâm không; hướng có vị trí có đáng làm không; đưa vị trí vào kiểu nào, em đề xuất thử A, C trước; giữ mô hình đóng băng hay cho train adapter; và tập dữ liệu thứ hai."

---

## Câu hỏi thầy có thể hỏi
- *Sao baseline khác paper?* → "Lệch ±1 cả 2 chiều, có thể do cấu hình decode. Em so khoảng cách giữa các hàng trong cùng một lần chạy là chính."
- *Có ý nghĩa thống kê không?* → "So với Hou25 thì chưa, vì họ không công bố output từng câu. Giữa các hàng em chạy thì tính bootstrap được, đây là bước tiếp."
- *Sao oracle + spoken chưa có?* → "Chưa chạy, khoảng 20 phút GPU."
- *Cách B có còn 'đóng băng' không?* → "LLM vẫn đóng băng, chỉ train adapter nhỏ để đưa embedding của từ vào; nhưng không còn training-free hoàn toàn."
- *Attention của pxattn có cho vị trí đúng không?* → "Chưa đo. Đo được bằng forced aligner trên câu đúng: lấy frame có attention cao nhất, so với khoảng thời gian thật."
