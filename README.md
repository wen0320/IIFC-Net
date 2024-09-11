# IIFC-Net: A Monaural Speech Enhancement Network with High-Order Information Interaction and Feature Calibration （The article has been accepted by IEEE Signal Processing Letters）

# Demo: Visit our [demo website](https://wen0320.github.io) for audio samples.


# Result
Denoising performance comparison with other systems on the VoiceBank+DEMAND dataset

| Model    | Par .(M) | MACs (G/s) | SISNR | PESQ | STOI (%) | CSIG | CBAK | COVL |
| -----    |--------  | ---------  | ----- | ---- | -------- | ---- | ---- | ---- |
| Noisy    |    ——    |     ——     | 8.45  | 1.97 |    92    | 3.34 | 2.45 | 2.63 |
| PHASEN   |   8.76   |     ——     |   ——  | 2.99 |    ——    | 4.21 | 3.55 | 3.62 |
| DCCRN    |   3.67   |    14.13   | 19.13 | 2.57 |    94    | 3.93 | 2.9  | 3.21 |
| TSTNN    |   0.92   |     ——     | 18.82 | 2.96 |    95    | 4.33 | 3.53 | 3.67 |
| SADNUNet |   2.63   |     ——     |   ——  | 2.82 |    ——    | 4.18 | 3.47 | 3.51 |
| FAF-Net  |   6.9    |   108.96   |   ——  | 3.19 |    95    | 4.13 | 3.38 | 3.66 |
| PFRNet   |   4.61   |     ——     |   ——  | 3.24 |    95    | 4.48 | 3.7  | 3.9  |
| D<sup>2</sup>Net    |   1.13   |    36.21   | 19.78 | 3.27 |    96    | 4.63 | 3.18 | 3.92 |
| IIFC-Net |  0.586   |    14.91   | 19.82 | 3.28 |    96    | 4.52 | 3.72 | 3.92 |


