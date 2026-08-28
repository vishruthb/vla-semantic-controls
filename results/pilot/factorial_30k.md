# 2x2 factorial statistics, matched episodes @ step 30000

10 tasks x 20 matched episodes = 200 A/B/C/D outcome tuples; 20000 bootstrap replicates; primary CI = task-stratified episode bootstrap of complete tuples; task-cluster CI = 10-cluster robustness check only.

| effect | points | episode-stratified 95% CI | task-cluster 95% CI (10 clusters) | sign-flip p |
| --- | ---: | ---: | ---: | ---: |
| VLM-update main effect ((B−A)+(D−C))/2 | +7.50 | [+2.0, +13.0] | [-3.0, +17.2] | 0.017 |
| routing main effect ((C−A)+(D−B))/2 | -5.00 | [-11.0, +1.0] | [-11.8, +1.0] | 0.137 |
| interaction (D−C)−(B−A) | -1.00 | [-11.0, +9.0] | [-13.0, +9.5] | 0.928 |

| paired contrast | points | wins / losses / ties | McNemar p | episode-stratified 95% CI | task-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| B − A (VLM update, all-layer) | +8.0 | 40 / 24 / 136 | 0.060 | [+1.0, +15.0] | [-4.5, +20.5] |
| D − C (VLM update, cross-only) | +7.0 | 40 / 26 / 134 | 0.109 | [-0.5, +14.5] | [-4.0, +17.0] |
| C − A (routing, frozen) | -4.5 | 28 / 37 / 135 | 0.321 | [-12.0, +3.0] | [-14.0, +4.0] |
| D − B (routing, trainable) | -5.5 | 30 / 41 / 129 | 0.235 | [-13.5, +2.5] | [-13.5, +2.0] |

Within-episode correlation of the two VLM-update contrasts corr(B−A, D−C) = +0.139; of the two routing contrasts corr(C−A, D−B) = +0.185.

Per-task effects (points, 20 episodes each):

| task | A | B | C | D | VLM main | routing main | interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 80 | 65 | 55 | +2.5 | -12.5 | -25.0 |
| 1 | 80 | 90 | 80 | 95 | +12.5 | +2.5 | +5.0 |
| 2 | 75 | 85 | 75 | 85 | +10.0 | +0.0 | +0.0 |
| 3 | 75 | 95 | 40 | 75 | +27.5 | -27.5 | +15.0 |
| 4 | 65 | 65 | 55 | 65 | +5.0 | -5.0 | +10.0 |
| 5 | 85 | 60 | 85 | 55 | -27.5 | -2.5 | -5.0 |
| 6 | 95 | 80 | 85 | 90 | -5.0 | +0.0 | +20.0 |
| 7 | 95 | 85 | 75 | 80 | -2.5 | -12.5 | +15.0 |
| 8 | 30 | 75 | 45 | 50 | +25.0 | -5.0 | -40.0 |
| 9 | 40 | 70 | 55 | 80 | +27.5 | +12.5 | -5.0 |
