# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Device: `cuda`
- Max sequence length: `1536`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `4393342464`
- Trainable parameters: `40370176`
- Output dir: `outputs/sft_single_tool_protocol_v4_qwen25_7b_qlora_1536`
- LoRA: `r=16, alpha=32, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Quantization: `4bit nf4, compute=bfloat16, double_quant=True`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 42 | `1, 12, 20, 27, 34, 38, 42` | 724.6 | 1528 | 33.6 | 0 |
| valid | 15 | `15, 23, 33` | 894.5 | 1476 | 45.7 | 0 |

## Training

- Steps: `160`
- First train loss: `7.3060`
- Final train loss: `0.0001`
- Min train loss: `0.0000`
- Max train loss: `7.3060`
- Final rolling mean: `0.0107`
- Valid loss before: `1.9999`
- Valid loss after: `0.2489`
- Elapsed seconds: `136.0`
- Max CUDA memory MB: `12024.78466796875`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 1.9999 |
| 20 | 0.4366 |
| 40 | 0.2503 |
| 60 | 0.2619 |
| 80 | 0.3002 |
| 100 | 0.2520 |
| 120 | 0.2360 |
| 140 | 0.2459 |
| 160 | 0.2489 |

## Loss Trace

```text
1: 7.3060
2: 1.5861
3: 4.0744
4: 4.3294
5: 5.2153
6: 4.0580
7: 1.3031
8: 0.8294
9: 2.3808
10: 0.0728
11: 0.3012
12: 0.0824
13: 0.1914
14: 0.6923
15: 0.1840
16: 0.0429
17: 0.0073
18: 0.1268
19: 0.1842
20: 0.0416
21: 0.0010
22: 0.0208
23: 0.1757
24: 0.6458
25: 0.3399
26: 0.6840
27: 0.0029
28: 0.0019
29: 0.0011
30: 1.2294
31: 1.5434
32: 0.0946
33: 0.0095
34: 1.6186
35: 0.0400
36: 0.0026
37: 0.0106
38: 0.0023
39: 0.0020
40: 0.0006
41: 0.0516
42: 0.0149
43: 0.0038
44: 0.0004
45: 0.0493
46: 0.0253
47: 0.9659
48: 0.0422
49: 0.0002
50: 0.1246
51: 0.0002
52: 0.0005
53: 0.0056
54: 0.0005
55: 0.0002
56: 0.6554
57: 0.0026
58: 0.0436
59: 0.0011
60: 0.0009
61: 0.0225
62: 0.0003
63: 0.1987
64: 0.0004
65: 0.0007
66: 0.0002
67: 0.0002
68: 0.0001
69: 0.0011
70: 0.0004
71: 0.0001
72: 0.0010
73: 0.0001
74: 0.0002
75: 0.0001
76: 0.4573
77: 0.0001
78: 2.1818
79: 0.0001
80: 0.0021
81: 0.0001
82: 0.0001
83: 0.0002
84: 0.0133
85: 0.0001
86: 0.0001
87: 0.0003
88: 0.0019
89: 0.0003
90: 0.0007
91: 0.0001
92: 0.0003
93: 0.0018
94: 0.0004
95: 0.0002
96: 0.0003
97: 0.0001
98: 0.0118
99: 0.0698
100: 0.0001
101: 0.0002
102: 0.0001
103: 0.0010
104: 0.0002
105: 0.0001
106: 0.0001
107: 0.0244
108: 0.2919
109: 0.0007
110: 0.0001
111: 0.0001
112: 0.0555
113: 0.0001
114: 0.3335
115: 0.0002
116: 0.0001
117: 0.0002
118: 0.0018
119: 0.0002
120: 0.0059
121: 0.0000
122: 0.0014
123: 0.0002
124: 0.0002
125: 0.0002
126: 0.3033
127: 0.0068
128: 0.2231
129: 0.0070
130: 0.0001
131: 0.0012
132: 0.0004
133: 0.0001
134: 0.0002
135: 0.0001
136: 0.0001
137: 0.0002
138: 0.0029
139: 0.0002
140: 0.0003
141: 0.0002
142: 0.0003
143: 0.0008
144: 0.0001
145: 0.0001
146: 0.0003
147: 0.0001
148: 0.0001
149: 0.0010
150: 0.0001
151: 0.0001
152: 0.0001
153: 0.0009
154: 0.0393
155: 0.0000
156: 0.0007
157: 0.0001
158: 0.0658
159: 0.0001
160: 0.0001
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
