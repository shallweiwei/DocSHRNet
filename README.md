# DocSHRNet: Towards Real-World Document Specular Highlight Removal


[![Paper](https://img.shields.io/badge/Paper-PRCV%202025-red)](https://link.springer.com/chapter/10.1007/978-981-95-5676-2_8)
[![Dataset](https://img.shields.io/badge/Dataset-DocHighlight-blue.svg)](https://github.com/SCUT-DLVCLab/DocHighlight)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

This repository contains the official PyTorch implementation for the paper [**"Towards Real-World Document Specular Highlight Removal: The DocHighlight Dataset and DocSHRNet Method"**](https://link.springer.com/chapter/10.1007/978-981-95-5676-2_8) published in *Pattern Recognition and Computer Vision (PRCV 2025)*.


## 📦 Dataset Preparation

Download the **[DocHighlight dataset](https://github.com/SCUT-DLVCLab/DocHighlight)** from the official release and place it in a directory of your choice.

---

## ⚙️ Getting Started

### 1. Environment Setup

```bash
conda create -n docshrnet 
conda activate docshrnet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pyiqa 
```

### 2. Pre-trained Models

Download the pre-trained model checkpoints from the official release [here](https://drive.usercontent.google.com/download?id=1St86y5F_ltZXWGjIhjBfUkB81VsgRBvc&export=download&authuser=0&confirm=t&uuid=9abb9bde-07c3-4aa7-b702-ece4dcb37bbc&at=AKSUxGNjddARw4_pdwdVr9JO1hcU:1759913983487).

### 3. Inference

To restore a single image or a directory of images:
```bash
python infer.py \
  --input_dir /path/to/dataset/ \
  --checkpoint ./checkpoints/docshrnet.pth \
  --output_dir ./results
```
- Restored images are saved as `<name>_result.png` under `--output_dir`.

For large images, you can use tiled prediction with `window_pred.py`:
```bash
python window_pred.py \
  --input_dir /path/to/dataset/ \
  --checkpoint ./checkpoints/docshrnet.pth \
  --output_dir ./window_results
```

## 📊 Evaluation

To evaluate the model performance on the test set, run `evaluate.py`. This script computes PSNR and SSIM metrics between the restored images and the ground truth.

```bash
python evaluate.py \
  --pred_dir ./results \
  --gt_dir /path/to/dataset/test/
```

---

## 🚀 Training

Launch with `torchrun` to enable multi-GPU training:

```bash
torchrun --nproc_per_node=4 train.py \
  --base_dir /path/to/dataset \
  --batch_size 4 \
  --total_iters 50000 \
  --output_dir ./experiments/docshrnet_exp
```

- Resume or warm-start with `--resume` / `--pretrained` checkpoints.
- Checkpoints, logs are saved in `output_dir`.

---

## 📚 Citation

If this project is useful in your research or product, please cite our paper:

```bibtex
@InProceedings{10.1007/978-981-95-5676-2_8,
author="Xu, Haowei
and Zhang, Jiaxin
and Cheng, Hiuyi
and Zhang, Peirong
and Zheng, Xuhan
and Jin, Lianwen",
editor="Kittler, Josef
and Xiong, Hongkai
and Yang, Jian
and Chen, Xilin
and Lu, Jiwen
and Lin, Weiyao
and Yu, Jingyi
and Zheng, Weishi",
title="Towards Real-World Document Specular Highlight Removal: The DocHighlight Dataset and DocSHRNet Method",
booktitle="Pattern Recognition and Computer Vision",
year="2026",
publisher="Springer Nature Singapore",
address="Singapore",
pages="109--124",
abstract="Document images often suffer from specular highlights caused by reflective surfaces or uneven lighting conditions, which significantly compromise document readability and reduce optical character recognition (OCR) accuracy in camera-captured document images. However, current document specular highlight datasets face critical limitations such as low resolution, unrealistic synthetic highlights, and insufficient diversity, restricting their applicability to real-world scenarios. In addition, existing highlight removal methods are primarily designed for natural scenarios, which struggle to preserve fine-grained textual details and structural consistency required in real-world documents. To address these challenges, we first introduce DocHighlight, a high-resolution, real-world dataset specifically designed for document specular highlight removal. DocHighlight comprises 2,201 paired images captured under diverse conditions, featuring various document types, illumination settings, and capture devices. Subsequently, we propose Document Specular Highlight Removal Network (DocSHRNet), a new highlight removal method incorporating the Document Structure Attention (DSA) and Adaptive Receptive Field (ARF) modules. These modules facilitate precise structural preservation and adapt to multi-scale highlight patterns, ensuring high-quality restoration. Extensive experiments on the DocHighlight, RD, and SD1 datasets demonstrate that DocSHRNet delivers competitive performance in reconstruction quality and OCR accuracy. These results demonstrate the effectiveness of DocHighlight as a real-world dataset and the robustness of DocSHRNet in addressing document specular highlight removal challenges, providing a solid foundation for real-world applications. The dataset and code are publicly available at https://github.com/shallweiwei/DocSHRNet.",
isbn="978-981-95-5676-2"
}




