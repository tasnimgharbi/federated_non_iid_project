### Overall Experimental Summary

| Dataset       | Partition     | Algorithm   |   Final Accuracy (%) |   Final Loss |   Client STD |   Client Variance |   Communication Cost (MB) |   Avg. Latency (ms) |
|:--------------|:--------------|:------------|---------------------:|-------------:|-------------:|------------------:|--------------------------:|--------------------:|
| Fashion-MNIST | IID           | FedAvg      |                86.2  |       0.386  |       0.0087 |            0.0001 |                     15.38 |              196    |
| Fashion-MNIST | IID           | FedProx     |                87    |       0.3698 |       0.0056 |            0      |                     15.38 |              181.88 |
| Fashion-MNIST | IID           | FedNova     |                86.97 |       0.3706 |       0.0056 |            0      |                     15.38 |              182.42 |
| Fashion-MNIST | Label-Skew    | FedAvg      |                28.07 |       3.0477 |       0.0542 |            0.0029 |                     15.38 |              163.32 |
| Fashion-MNIST | Label-Skew    | FedProx     |                25.41 |       3.1294 |       0.0516 |            0.0027 |                     15.38 |              132.74 |
| Fashion-MNIST | Label-Skew    | FedNova     |                25.66 |       3.2102 |       0.0514 |            0.0026 |                     15.38 |              145.11 |
| Fashion-MNIST | Quantity-Skew | FedAvg      |                88.15 |       0.3249 |       0.0343 |            0.0012 |                     15.38 |              128.79 |
| Fashion-MNIST | Quantity-Skew | FedProx     |                87.48 |       0.3443 |       0.0369 |            0.0014 |                     15.38 |              170.76 |
| Fashion-MNIST | Quantity-Skew | FedNova     |                86.38 |       0.3886 |       0.0312 |            0.001  |                     15.38 |              203.16 |
| HAR           | IID           | FedAvg      |                90.91 |       0.2537 |       0.0163 |            0.0003 |                      6.15 |              156.61 |
| HAR           | IID           | FedProx     |                91.72 |       0.2559 |       0.0198 |            0.0004 |                      6.15 |              163.28 |
| HAR           | IID           | FedNova     |                91.52 |       0.2474 |       0.0115 |            0.0001 |                      6.15 |              145.58 |
| HAR           | Label-Skew    | FedAvg      |                34.24 |       1.5694 |       0.1251 |            0.0156 |                      6.15 |              124.58 |
| HAR           | Label-Skew    | FedProx     |                34.03 |       1.6473 |       0.0734 |            0.0054 |                      6.15 |              145.89 |
| HAR           | Label-Skew    | FedNova     |                34.27 |       1.5738 |       0.1021 |            0.0104 |                      6.15 |              185.8  |
| HAR           | Quantity-Skew | FedAvg      |                92.64 |       0.1818 |       0.0236 |            0.0006 |                      6.15 |              148.46 |
| HAR           | Quantity-Skew | FedProx     |                93.15 |       0.1757 |       0.0288 |            0.0008 |                      6.15 |              209.57 |
| HAR           | Quantity-Skew | FedNova     |                93.62 |       0.1716 |       0.0215 |            0.0005 |                      6.15 |              181.99 |