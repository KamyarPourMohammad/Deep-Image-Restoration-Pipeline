# Deep Image Restoration Pipeline
## Overview
This project implements a complete deep learning pipeline for image restoration and semantic understanding. Given a heavily corrupted image affected by salt-and-pepper noise and blur, the pipeline recovers both visual quality and semantic information using state-of-the-art deep neural networks. The restored image is then passed to a classifier to verify whether the restoration was successful enough for an AI model to understand its content.

## Project Pipeline
The complete pipeline consists of three sequential phases:

```bash
Corrupted Image 
→ Phase 1: Deep Denoising → Denoised but Blurred Image 
→ Phase 2: Deep Deblurring → Restored Image 
→ Phase 3: Image Classification → Predicted Class + Confidence
```
<p align="center">
    <img src="results\corrupted_image.jpg" width="40%" alt="Lite / Ultimate Download Banner">
    
</p>
<p align="center">
  <em>Figure 1: Blured and noisy image</em>
</p>

## Architecture
### Phase 1: Deep Denoising
For phase one I have downloaded <a href="https://www.kaggle.com/models/aastha2807/denoising?select=denoising_model.h5">denosing_model.h5</a> and used this pretrained tensorflow model.finally it asisst me to remove salt-and-pepper noise while preserving image structure.
here is my output:
<p align="center">
    <img src="results\phase1_denoised.jpg" width="40%" alt="Lite / Ultimate Download Banner">
    
</p>
<p align="center">
  <em>Figure 2: denoised but still blurred image</em>
</p>


### Phase 2: Deep Deblurring
* Recovers sharp edges and high-frequency information

* Implements a ResNet architecture with skip connections to preserve fine details

* Uses PyTorch weights (ResNet.pth)

* Evaluated using PSNR (Peak Signal-to-Noise Ratio) against ground truth

### Phase 3: Image Classification
* Trains a PreActResNet18 classifier on the CIFAR-10 dataset

* Evaluates semantic preservation of restored images

* Generates confusion matrix for performance analysi