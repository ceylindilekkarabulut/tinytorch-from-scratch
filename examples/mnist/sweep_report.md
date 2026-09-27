# Hyperparameter Sweep (10k samples, 2 epochs)
Baseline: lr=0.001, batch=64, channels=8/16, hidden=128, kernel=3

| Experiment | Test Accuracy | Time (s) |
|---|---|---|
| BASELINE (lr.001 bs64 8/16/128) | 95.40% | 234 |
| lr = 0.01 | 96.21% (+0.81%) | 232 |
| lr = 0.0001 | 88.15% (-7.25%) | 230 |
| batch = 16 | 97.18% (+1.78%) | 215 |
| batch = 256 | 92.20% (-3.20%) | 206 |
| channels 4/8 | 95.52% (+0.12%) | 134 |
| channels 16/32 | 95.53% (+0.13%) | 447 |
| hidden = 32 | 93.56% (-1.84%) | 212 |
| hidden = 256 | 96.31% (+0.91%) | 306 |
| kernel = 5 | 95.26% (-0.14%) | 258 |
