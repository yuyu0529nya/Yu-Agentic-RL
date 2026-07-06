# Seen/Unseen Tau2 Summary: airline_qwen25_7b_phase2i_m2_phase2i_20260623_141314

- simulations: 50
- tasks evaluated: 50

| Split | pass^1 | successes/trials | tasks |
| --- | ---: | ---: | ---: |
| overall | 0.1000 | 5/50 | 50 |
| seen (SFT train/valid) | 0.1000 | 1/10 | 10 |
| **unseen (generalization)** | 0.1000 | 4/40 | 40 |

> unseen pass^1 is the honest, leakage-free generalization metric.
> A large seen-minus-unseen gap indicates memorization of SFT tasks.

- evaluated seen ids: `['1', '12', '15', '20', '23', '27', '33', '34', '38', '42']`
- evaluated unseen ids: `['0', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16', '17', '18', '19', '21', '22', '24', '25', '26', '28', '29', '30', '31', '32', '35', '36', '37', '39', '40', '41', '43', '44', '45', '46', '47', '48', '49']`
