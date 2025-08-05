# [ICME 2025] Exploring State Space Model in Wavelet Domain: An Infrared and Visible Image Fusion Network via Wavelet Transform and State Space Model
### [Arxiv](https://arxiv.org/abs/2503.18378) | [Code](https://github.com/Lmmh058/W-Mamba) 

**Exploring State Space Model in Wavelet Domain: An Infrared and Visible Image Fusion Network via Wavelet Transform and State Space Model**

Tianpei Zhang, Yiming Zhu, Jufeng Zhao, Guangmang Cui, Yuchen Zheng in ICME 2025

![Framework](fig/Architecture.png)

## Create Conda Environment

```bash
# - Create environment from yaml
conda env create -f environment.yaml

# - Activate
conda activate WMamba
```

## Prepare Your Dataset
The dataset used in this paper can be downloaded at:
[TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) | [MSRS](https://github.com/Linfeng-Tang/MSRS)

The images you use should be placed in:
```bash
    test_image/
                infrared/
                visible/
    train_image/
                infrared/
                visible/
```
