# TI mmWave Radar Fall Detection: Representation Comparison

| Representation                               |   Accuracy (%) |   Precision (%) |   Recall (%) |   Macro F1 (%) |   ROC-AUC |   False Alarm Rate / FPR (%) |   Latency (ms/seq) |
|:---------------------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|-----------------------------:|-------------------:|
| Rep 1: Doppler-Time Map (ResNet-18)          |          53.33 |           53.45 |        61.04 |          52.19 |    0.571  |                        54.38 |               8.14 |
| Rep 2: 2D Orthogonal Projections (ResNet-18) |          72.5  |           74.41 |        69.17 |          72.37 |    0.8196 |                        24.17 |               8.08 |
| Rep 3: Native 3D Point Tensor (PointNet++)   |          71.25 |           76.56 |        61.25 |          70.96 |    0.7865 |                        18.75 |               4.32 |
