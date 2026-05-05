# Frequency-Spatial Decoupled Enhancement Network for Cross-Modality Person Re Identification

This is a pytorch implementation of 《Frequency-Spatial Decoupled Enhancement Network for Cross-Modality Person Re Identification》(Pattern Recognition, under review). 


## Abstract
Visible–infrared person re-identification (VI-ReID) aims to match pedestrian images of the same individual captured by different cameras under significant illumination variations. Existing methods that rely solely on spatial-domain features are highly sensitive to lighting changes and often fail to recover discriminative high-frequency cues such as textures and edges. They also overlook the structural stability of low frequency components, which encode global contour information that is crucial for restoring identity-specific details. To address these limitations, we propose a Frequency–Spatial Decoupled Enhancement Network (FSDE-Net) that jointly models and integrates frequency and spatial representations to achieve robust cross-modality matching. Specifically, we introduce two key modules. The Low-Frequency Guided High-Frequency Enhancement (LF-GHFE) module employs discrete wavelet transforms and adaptive masking to leverage stable low-frequency signals for selectively enhancing identity-discriminative high-frequency components. The Low Frequency Decoupling Multi-Scale Alignment (LF-DMSA) module utilizes low frequency filtering and multi-scale convolution to capture and align global structural information across modalities. Extensive experiments on the SYSU-MM01, RegDB, and LLCM datasets demonstrate that FSDE-Net consistently outperforms state-of-the-art methods under both indoor and outdoor settings in terms of accuracy and robustness.

## Performance


## Architecture

```
<TBD>
```

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/FSDENet
```

## Project Structure


## Dependency


## Train



## Citation

```


```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
