# 2x2 factorial statistics, matched episodes @ step 20000

10 tasks x 20 matched episodes = 200 A/B/C/D outcome tuples; 20000 bootstrap replicates; primary CI = task-stratified episode bootstrap of complete tuples; task-cluster CI = 10-cluster robustness check only.

| effect | points | episode-stratified 95% CI | task-cluster 95% CI (10 clusters) | sign-flip p |
| --- | ---: | ---: | ---: | ---: |
| VLM-update main effect ((B−A)+(D−C))/2 | +7.25 | [+1.2, +13.0] | [-0.5, +14.5] | 0.026 |
| routing main effect ((C−A)+(D−B))/2 | -1.25 | [-7.0, +4.5] | [-7.0, +4.0] | 0.745 |
| interaction (D−C)−(B−A) | -2.50 | [-14.0, +9.5] | [-17.0, +13.0] | 0.747 |

| paired contrast | points | wins / losses / ties | McNemar p | episode-stratified 95% CI | task-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| B − A (VLM update, all-layer) | +8.5 | 46 / 29 / 125 | 0.064 | [+0.5, +16.5] | [-2.0, +19.0] |
| D − C (VLM update, cross-only) | +6.0 | 47 / 35 / 118 | 0.224 | [-2.5, +14.5] | [-5.0, +15.5] |
| C − A (routing, frozen) | +0.0 | 39 / 39 / 122 | 1.000 | [-8.5, +8.0] | [-11.0, +10.5] |
| D − B (routing, trainable) | -2.5 | 34 / 39 / 127 | 0.640 | [-10.5, +6.0] | [-9.5, +5.5] |

Within-episode correlation of the two VLM-update contrasts corr(B−A, D−C) = +0.013; of the two routing contrasts corr(C−A, D−B) = -0.013.

Per-task effects (points, 20 episodes each):

| task | A | B | C | D | VLM main | routing main | interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 75 | 80 | 50 | 70 | +12.5 | -17.5 | +15.0 |
| 1 | 65 | 85 | 75 | 90 | +17.5 | +7.5 | -5.0 |
| 2 | 70 | 70 | 80 | 75 | -2.5 | +7.5 | -5.0 |
| 3 | 75 | 90 | 60 | 85 | +20.0 | -10.0 | +10.0 |
| 4 | 60 | 75 | 65 | 70 | +10.0 | +0.0 | -10.0 |
| 5 | 65 | 65 | 85 | 50 | -17.5 | +2.5 | -35.0 |
| 6 | 80 | 90 | 90 | 95 | +7.5 | +7.5 | -5.0 |
| 7 | 90 | 90 | 75 | 75 | +0.0 | -15.0 | +0.0 |
| 8 | 20 | 65 | 45 | 50 | +25.0 | +5.0 | -40.0 |
| 9 | 75 | 50 | 50 | 75 | +0.0 | +0.0 | +50.0 |
