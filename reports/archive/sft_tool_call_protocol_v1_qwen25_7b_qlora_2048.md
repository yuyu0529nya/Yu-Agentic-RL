# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Device: `cuda`
- Max sequence length: `2048`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `4393342464`
- Trainable parameters: `40370176`
- Output dir: `outputs/sft_tool_call_protocol_v1_qwen25_7b_qlora_2048`
- LoRA: `r=16, alpha=32, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Quantization: `4bit nf4, compute=bfloat16, double_quant=True`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 53 | `1, 12, 20, 27, 34, 38, 42` | 1047.8 | 2036 | 35.6 | 0 |
| valid | 27 | `15, 23, 33` | 1391.9 | 2041 | 67.0 | 0 |

## Training

- Steps: `160`
- First train loss: `1.2460`
- Final train loss: `0.0003`
- Min train loss: `0.0000`
- Max train loss: `3.6896`
- Final rolling mean: `0.1497`
- Valid loss before: `2.5809`
- Valid loss after: `0.1528`
- Elapsed seconds: `188.3`
- Max CUDA memory MB: `13406.76025390625`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 2.5809 |
| 20 | 0.3475 |
| 40 | 0.2660 |
| 60 | 0.1763 |
| 80 | 0.1747 |
| 100 | 0.1649 |
| 120 | 0.1518 |
| 140 | 0.1516 |
| 160 | 0.1528 |

## Loss Trace

```text
1: 1.2460
2: 1.1320
3: 2.2356
4: 2.3413
5: 3.6896
6: 2.1505
7: 0.6247
8: 1.3255
9: 0.9001
10: 0.0105
11: 1.1137
12: 3.2458
13: 1.2773
14: 0.6496
15: 0.0953
16: 0.2668
17: 2.2850
18: 1.1625
19: 0.8767
20: 0.0155
21: 0.1010
22: 0.1479
23: 0.0074
24: 0.2524
25: 1.2074
26: 0.1112
27: 1.0937
28: 0.0727
29: 0.0086
30: 0.0008
31: 1.1856
32: 0.0012
33: 0.0005
34: 0.0104
35: 1.0813
36: 0.0016
37: 0.2058
38: 1.3435
39: 0.0010
40: 1.4396
41: 0.0015
42: 0.3396
43: 0.0026
44: 0.9810
45: 1.5544
46: 0.0191
47: 0.0042
48: 0.0014
49: 1.0427
50: 0.0434
51: 0.1335
52: 0.0857
53: 0.0348
54: 0.0005
55: 0.0018
56: 0.0004
57: 0.2682
58: 0.0005
59: 1.1531
60: 0.0008
61: 0.9693
62: 0.0015
63: 0.0038
64: 0.0010
65: 0.0010
66: 0.8793
67: 0.0006
68: 0.0065
69: 0.0005
70: 0.0663
71: 0.7040
72: 0.0029
73: 0.0138
74: 0.0040
75: 0.0013
76: 0.0004
77: 0.0015
78: 0.0328
79: 0.0003
80: 0.5448
81: 0.0003
82: 0.0972
83: 0.9556
84: 0.0005
85: 0.0012
86: 1.2957
87: 0.0001
88: 0.0003
89: 0.0004
90: 0.0002
91: 0.0071
92: 0.0001
93: 0.9005
94: 0.0004
95: 0.0003
96: 0.0426
97: 0.0001
98: 1.0294
99: 0.4833
100: 0.0001
101: 0.0001
102: 0.0001
103: 0.0160
104: 0.2994
105: 0.0186
106: 0.8210
107: 0.0001
108: 0.0001
109: 0.0003
110: 0.3126
111: 0.0981
112: 0.0002
113: 0.0001
114: 0.0000
115: 0.0001
116: 0.0004
117: 0.5509
118: 0.0769
119: 0.0001
120: 0.0001
121: 0.0001
122: 0.0001
123: 0.6560
124: 0.6613
125: 0.4578
126: 0.0002
127: 0.0018
128: 0.0057
129: 0.0040
130: 0.0002
131: 0.1529
132: 0.0002
133: 0.0003
134: 0.0007
135: 0.0001
136: 0.0002
137: 0.4980
138: 0.0002
139: 0.0002
140: 0.0001
141: 0.0007
142: 0.0001
143: 0.0003
144: 0.0006
145: 0.0001
146: 0.0001
147: 0.0002
148: 0.0002
149: 0.3642
150: 0.6075
151: 0.0001
152: 0.0001
153: 0.0002
154: 0.6491
155: 0.1393
156: 0.7075
157: 0.0003
158: 0.0001
159: 0.0001
160: 0.0003
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
