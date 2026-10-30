# TI mmWave Radar Fall Detection: Representation Comparison

| Representation                               |   Accuracy (%) |   Precision (%) |   Recall (%) |   Macro F1 (%) |   ROC-AUC |   False Alarm Rate / FPR (%) |   Latency (ms/seq) |
|:---------------------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|-----------------------------:|-------------------:|
| Rep 1: Doppler-Time Map (ResNet-18)          |          53.02 |           53.25 |        64.79 |          50.71 |    0.5612 |                        58.75 |              12.79 |
| Rep 2: 2D Orthogonal Projections (ResNet-18) |          72.5  |           74.41 |        69.17 |          72.37 |    0.8196 |                        24.17 |              11.59 |
| Rep 3: Native 3D Point Tensor (PointNet++)   |          63.96 |           65.81 |        60.21 |          63.2  |    0.6783 |                        32.29 |               5.57 |
