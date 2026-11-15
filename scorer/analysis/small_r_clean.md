## test-clean

### 1. Độ sạch: % từ đúng có ít hơn K từ nhiễu đứng trên

| hệ | lượt | Sạch@1 | Sạch@3 | Sạch@5 | Sạch@10 | Sạch@50 | TB từ nhiễu p≥0.5 / câu |
|---|---|---|---|---|---|---|---|
| hou25 | 5692 | 80.9 | 90.7 | 93.3 | 95.4 | 97.8 | 1.8 |
| pmaxsim | 5692 | 93.0 | 97.5 | 98.1 | 98.7 | 99.5 | 0.5 |

### 2. Từ đúng KHÔNG đứng đầu (có ≥1 từ nhiễu trên nó): nhóm lỗi theo từ nhiễu cao nhất

**hou25** (1090 lượt, 19.1% số từ đúng)

| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |
|---|---|---|---|
| bị từ KHÁC HẲN vượt | 1057 | 97 | 22 |
| bị từ GẦN ÂM (≤2 âm vị) vượt | 29 | 3 | 48 |
| bị từ ĐỒNG ÂM vượt | 3 | 0 | 0 |
| sau giây 30 (encoder cắt) | 1 | 0 | 100 |

**pmaxsim** (397 lượt, 7.0% số từ đúng)

| nhóm | lượt | % | trong đó Whisper không prompt nghe SAI % |
|---|---|---|---|
| bị từ KHÁC HẲN vượt | 363 | 91 | 41 |
| bị từ GẦN ÂM (≤2 âm vị) vượt | 22 | 6 | 55 |
| bị từ ĐỒNG ÂM vượt | 11 | 3 | 9 |
| sau giây 30 (encoder cắt) | 1 | 0 | 100 |

### 3. Từ nhiễu hay đứng đầu nhất (đứng trên từ đúng, đếm số câu)

- hou25: deterring (15), freeholder (11), ethelgiva (11), ironer (10), vandergriff (10), anafarta (9), wantee (8), sandwich (8), guaco (8), clutching (7), goshawk's (7), piziquid (7), harwich (7), conducta (6), affirm (6)
- pmaxsim: wrot (7), inhabitant's (5), f'om (5), poppe (5), lovelier (4), intermit (4), slim's (4), martialed (4), beauty's (4), town's (4), wale (4), duse (3), de'il (3), roe (3), affined (3)

### 4. Ví dụ (hou25): từ đúng ← từ nhiễu cao nhất, khoảng cách âm vị

- 2961-960-0006 **neo** (d=50) ← valentinus [8]
- 4992-41797-0001 **firebugs** (d=38) ← freeholder [7]
- 260-123288-0026 **approaches** (d=1) ← diffuses [6]
- 3570-5695-0006 **decency** (d=2) ← acquittal [7]
- 4446-2273-0010 **watercress** (d=2) ← duklass [6]
- 5105-28233-0007 **arrondissement** (d=11) ← anafarta [9]
- 6829-68769-0000 **watson** (d=1) ← stephani [5]
- 1221-135766-0011 **snatch** (d=1) ← immeasurably [8]
- 4077-13751-0021 **missouri** (d=6) ← mobs [5]
- 1580-141084-0032 **upraised** (d=1) ← pauperized [4]
- 2961-960-0002 **jewish** (d=2) ← competitions [9]
- 7729-102255-0029 **consumed** (d=1) ← yeild [6]
- 2961-960-0000 **physiology** (d=3) ← scaurs [9]
- 4970-29093-0006 **processes** (d=2) ← nt [7]
- 1320-122617-0018 **purposed** (d=50) ← gleamy [6]
- 3729-6852-0009 **silvia** (d=3) ← survey [4]
- 1089-134691-0017 **citadelled** (d=2) ← dishevelled [5]
- 1089-134691-0024 **dedalos** (d=38) ← leastthe [7]
- 8463-294825-0007 **humorous** (d=4) ← ironer [6]
- 4992-23283-0012 **softness** (d=4) ← afanasy [4]
