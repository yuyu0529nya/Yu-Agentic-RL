# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `.\models\Qwen2.5-0.5B-Instruct`
- Device: `cuda`
- Max sequence length: `2048`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `498431872`
- Trainable parameters: `4399104`
- Output dir: `outputs/sft_qwen25_05b_lora_phase1f_2k`
- LoRA: `r=8, alpha=16, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 14 | `1, 12, 20, 27, 34, 38, 42` | 2048.0 | 2048 | 467.9 | 14 |
| valid | 4 | `15, 23, 33` | 2048.0 | 2048 | 689.5 | 4 |

## Training

- Steps: `120`
- First train loss: `1.1699`
- Final train loss: `0.0133`
- Min train loss: `0.0017`
- Max train loss: `1.4918`
- Final rolling mean: `0.0439`
- Valid loss before: `1.2835`
- Valid loss after: `1.4752`
- Elapsed seconds: `418.2`
- Max CUDA memory MB: `5266.33154296875`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 1.2835 |
| 10 | 0.9827 |
| 20 | 0.9091 |
| 30 | 0.9297 |
| 40 | 0.9744 |
| 50 | 0.9763 |
| 60 | 1.0769 |
| 70 | 1.1335 |
| 80 | 1.2523 |
| 90 | 1.2403 |
| 100 | 1.4084 |
| 110 | 1.4944 |
| 120 | 1.4752 |

## Loss Trace

```text
1: 1.1699
2: 0.7297
3: 1.1493
4: 1.0825
5: 1.4918
6: 0.9064
7: 1.1644
8: 1.3221
9: 0.4229
10: 0.7712
11: 1.3931
12: 1.3417
13: 0.4650
14: 0.4329
15: 1.1693
16: 0.8020
17: 1.1872
18: 0.9601
19: 0.5171
20: 0.1711
21: 0.2119
22: 0.5223
23: 0.4413
24: 0.1753
25: 1.0233
26: 0.3503
27: 0.1937
28: 0.1610
29: 0.1194
30: 0.2311
31: 0.2771
32: 0.3311
33: 0.1193
34: 0.1590
35: 0.0839
36: 0.5293
37: 0.0515
38: 0.7054
39: 0.9419
40: 0.0514
41: 0.7383
42: 0.8718
43: 0.5426
44: 0.0229
45: 0.1735
46: 0.3659
47: 0.5982
48: 0.7315
49: 0.0586
50: 0.0182
51: 0.0422
52: 0.1380
53: 0.0495
54: 0.7038
55: 0.0604
56: 0.0616
57: 0.1912
58: 0.0112
59: 0.3339
60: 0.0774
61: 0.0848
62: 0.0287
63: 0.0315
64: 0.0329
65: 0.0154
66: 0.5171
67: 0.0311
68: 0.0185
69: 0.5378
70: 0.3074
71: 0.0257
72: 0.0562
73: 0.0091
74: 0.2377
75: 0.0046
76: 0.0075
77: 0.0938
78: 0.4055
79: 0.3889
80: 0.0513
81: 0.0279
82: 0.1786
83: 0.0167
84: 0.0182
85: 0.0023
86: 0.0106
87: 0.0214
88: 0.0130
89: 0.0047
90: 0.0191
91: 0.1068
92: 0.0113
93: 0.0153
94: 0.1194
95: 0.0456
96: 0.2748
97: 0.2769
98: 0.0062
99: 0.0331
100: 0.0113
101: 0.0030
102: 0.2350
103: 0.0042
104: 0.0026
105: 0.0109
106: 0.0248
107: 0.0738
108: 0.0057
109: 0.2189
110: 0.0017
111: 0.0590
112: 0.0184
113: 0.0521
114: 0.0060
115: 0.0088
116: 0.0145
117: 0.1958
118: 0.0569
119: 0.0143
120: 0.0133
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
