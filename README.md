# IIFC-Net: A Monaural Speech Enhancement Network with High-Order Information Interaction and Feature Calibration
### Author: Wenbing Wei, Ying Hu, Hao Huang and Liang He


**Abstract :**
In this letter, we propose a monaural speech enhancement network with high-order information interaction and feature calibration (IIFC-Net), which includes high-order information interaction Transformer (HOIIFormer) with high-order information interaction (HOII) block instead of a multi-head self-attention (MHSA) in Transformer. 
IIFC-Net leverages dual-path HOIIFormer (DPH) to model the distant dependency relation along time and frequency dimensions, respectively, and effectively captures deep-level information through the HOII block. We also design a feature calibration (FC) block to enhance the frequency components of target speech, which can be verified by a visualization analysis.
Experimental results on the VoiceBank+DEMAND and WHAMR! datasets demonstrate that IIFC-Net achieves comparable performance in terms of denoising, dereverberation, and simultaneous denoising and dereverberation, while with fewer parameter counts (0.586M) and complexity.

Visit our [demo website](https://yangai520.github.io/APNet) for audio samples.
