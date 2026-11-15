## test-other

### 1. Độ sạch: % từ đúng có ít hơn K từ nhiễu đứng trên

| hệ | lượt | Sạch@1 | Sạch@3 | Sạch@5 | Sạch@10 | Sạch@50 | TB từ nhiễu p≥0.5 / câu |
|---|---|---|---|---|---|---|---|
| hou25 | 5248 | 72.8 | 83.8 | 87.2 | 90.9 | 95.4 | 2.6 |
| pmaxsim | 5248 | 85.7 | 93.7 | 95.1 | 96.7 | 98.4 | 0.7 |
| pxattn | 5248 | 77.4 | 89.4 | 92.0 | 94.6 | 97.7 | 1.7 |

### 2. Từ đúng KHÔNG đứng đầu (có ≥1 từ nhiễu trên nó): nhóm lỗi theo từ nhiễu cao nhất

**hou25** (1429 lượt, 27.2% số từ đúng)

| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |
|---|---|---|---|
| bị từ KHÁC HẲN vượt | 1370 | 96 | 37 |
| bị từ GẦN ÂM (≤2 âm vị) vượt | 43 | 3 | 47 |
| bị từ ĐỒNG ÂM vượt | 10 | 1 | 20 |
| sau giây 30 (encoder cắt) | 6 | 0 | 100 |

**pmaxsim** (753 lượt, 14.3% số từ đúng)

| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |
|---|---|---|---|
| bị từ KHÁC HẲN vượt | 685 | 91 | 51 |
| bị từ GẦN ÂM (≤2 âm vị) vượt | 48 | 6 | 62 |
| bị từ ĐỒNG ÂM vượt | 14 | 2 | 21 |
| sau giây 30 (encoder cắt) | 6 | 1 | 100 |

**pxattn** (1184 lượt, 22.6% số từ đúng)

| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |
|---|---|---|---|
| bị từ KHÁC HẲN vượt | 1100 | 93 | 40 |
| bị từ GẦN ÂM (≤2 âm vị) vượt | 66 | 6 | 55 |
| bị từ ĐỒNG ÂM vượt | 12 | 1 | 8 |
| sau giây 30 (encoder cắt) | 6 | 1 | 100 |

### 3. Từ nhiễu hay đứng đầu nhất (đứng trên từ đúng, đếm số câu)

- hou25: raffle (10), assawaman (10), chap's (9), boulevart (8), heber (8), vye's (7), grandpapa (7), wynch (7), shahzada (7), indexograph (7), earthling (7), hipp (6), wazir (6), assurement (6), boxlike (6)
- pmaxsim: jujus (8), storm'd (7), chap's (6), boulevart (6), sinbad (6), ambroses (6), grandpapa (5), aboveboard (5), dear' (5), pints (4), althar (4), djins (4), chinny (4), parley (4), ingenuousness (4)
- pxattn: djins (7), whereout (7), storm'd (7), circumferences (7), axaya (6), chap's (5), trivet (5), aw've (5), chinny (5), ious (5), hedjaz (5), rareness (5), beautie (5), bez (5), prisoner's (5)

### 4. Ví dụ (hou25): từ đúng ← từ nhiễu cao nhất, khoảng cách âm vị

- 7975-280057-0009 **wells** (d=1) ← guarde [4]
- 3528-168669-0043 **fauvent** (d=1) ← stupefaction [9]
- 6128-63240-0024 **luna's** (d=2) ← citizenship [9]
- 5442-41168-0012 **jeered** (d=1) ← appropriately [10]
- 2609-156975-0037 **asia** (d=6) ← essaieff [4]
- 533-131556-0003 **simulates** (d=50) ← intuite [7]
- 4852-28319-0007 **wicker** (d=9) ← perhapses [7]
- 6070-86744-0000 **albert's** (d=2) ← astonish [7]
- 5442-41168-0002 **ivanovitch** (d=50) ← hatreds [7]
- 2033-164914-0008 **eunuch** (d=2) ← d'alencon [6]
- 533-131564-0018 **regretting** (d=2) ← reasserting [4]
- 5442-32873-0006 **cage** (d=50) ← wynch [4]
- 6432-63723-0010 **grafton's** (d=2) ← grafton [1]
- 2609-169640-0014 **pikes** (d=10) ← pints [2]
- 4294-32859-0000 **liquor** (d=4) ← seet [4]
- 3764-168671-0046 **clogged** (d=1) ← riens [5]
- 8461-281231-0024 **athelstane** (d=11) ← oestergoetland [10]
- 2609-156975-0027 **oriental** (d=1) ← or'to [5]
- 3538-142836-0019 **conficere** (d=50) ← wormes [7]
- 7902-96594-0030 **gurr** (d=1) ← soh [2]
