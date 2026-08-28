# B@30k vs released SmolVLA — matched 200-episode LIBERO-Spatial protocol

Protocol fingerprints: reference `796598d68b5592a8`, candidate `796598d68b5592a8` (identical).

| policy | successes | success | Wilson 95% CI | per-task (of 20) |
| --- | ---: | ---: | ---: | --- |
| released SmolVLA | 164/200 | 82.0% | [76.1, 86.7] | [13, 19, 20, 19, 14, 17, 10, 16, 19, 17] |
| A@30k | 141/200 | 70.5% | [63.8, 76.4] | [13, 16, 15, 15, 13, 17, 19, 19, 6, 8] |
| B@30k | 157/200 | 78.5% | [72.3, 83.6] | [16, 18, 17, 19, 13, 12, 16, 17, 15, 14] |
| C@30k | 132/200 | 66.0% | [59.2, 72.2] | [13, 16, 15, 8, 11, 17, 17, 15, 9, 11] |
| D@30k | 146/200 | 73.0% | [66.5, 78.7] | [11, 19, 17, 15, 13, 11, 18, 16, 10, 16] |

## Paired: B@30k − released SmolVLA

- absolute delta: **-3.5 pts** (relative -4.3%)
- paired bootstrap 95% CI: [-11.0, +4.0]
- episode-level wins / losses / ties: 27 / 34 / 139
- exact McNemar p = 0.443

| task | reference | candidate | Δ |
| --- | ---: | ---: | ---: |
| 0 | 13 | 16 | +3 |
| 1 | 19 | 18 | -1 |
| 2 | 20 | 17 | -3 |
| 3 | 19 | 19 | +0 |
| 4 | 14 | 13 | -1 |
| 5 | 17 | 12 | -5 |
| 6 | 10 | 16 | +6 |
| 7 | 16 | 17 | +1 |
| 8 | 19 | 15 | -4 |
| 9 | 17 | 14 | -3 |

**Verdict: NO RELIABLE DIFFERENCE**
