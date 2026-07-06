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
- Output dir: `outputs/sft_action_prefix_phase1i_2k`
- LoRA: `r=8, alpha=16, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 42 | `1, 12, 20, 27, 34, 38, 42` | 721.9 | 2005 | 52.0 | 0 |
| valid | 11 | `15, 23, 33` | 729.2 | 2048 | 78.9 | 1 |

## Training

- Steps: `80`
- First train loss: `1.2729`
- Final train loss: `0.0436`
- Min train loss: `0.0006`
- Max train loss: `2.6602`
- Final rolling mean: `0.3424`
- Valid loss before: `1.5316`
- Valid loss after: `0.6061`
- Elapsed seconds: `716.8`
- Max CUDA memory MB: `5174.568359375`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 1.5316 |
| 20 | 0.7008 |
| 40 | 0.5942 |
| 60 | 0.5979 |
| 80 | 0.6061 |

## Loss Trace

```text
1: 1.2729
2: 1.3201
3: 1.2044
4: 0.5590
5: 1.8438
6: 1.6543
7: 0.1593
8: 1.4905
9: 1.1308
10: 0.2898
11: 1.0568
12: 0.1627
13: 0.0192
14: 0.6198
15: 0.3389
16: 0.7805
17: 0.5885
18: 0.8213
19: 1.0028
20: 0.3414
21: 0.0115
22: 1.0422
23: 0.0053
24: 0.5521
25: 0.2085
26: 0.0372
27: 0.3299
28: 1.9690
29: 1.5985
30: 0.3704
31: 0.0193
32: 0.6812
33: 0.4225
34: 0.0015
35: 0.1136
36: 0.0956
37: 0.4014
38: 0.1219
39: 0.8867
40: 0.1819
41: 2.6602
42: 0.3044
43: 0.1316
44: 0.0006
45: 0.0015
46: 0.0006
47: 0.2429
48: 0.0730
49: 0.2948
50: 0.0929
51: 0.0050
52: 0.4116
53: 0.1280
54: 1.6220
55: 0.0024
56: 0.0784
57: 0.0526
58: 0.0232
59: 0.0060
60: 0.0021
61: 0.0013
62: 1.0190
63: 0.0210
64: 0.4164
65: 0.0208
66: 0.0124
67: 0.1770
68: 0.0034
69: 0.4329
70: 0.0030
71: 1.4278
72: 0.2565
73: 0.5085
74: 0.3828
75: 0.0135
76: 0.0006
77: 0.0777
78: 0.5514
79: 0.1612
80: 0.0436
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
