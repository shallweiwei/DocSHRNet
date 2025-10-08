# DocSHRNet: Towards Real-World Document Specular Highlight Removal


[![Dataset](https://img.shields.io/badge/Dataset-DocHighlight-blue.svg)](https://github.com/SCUT-DLVCLab/DocHighlight)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

This repository contains the official PyTorch implementation for the paper **"Towards Real-World Document Specular Highlight Removal: The DocHighlight Dataset and DocSHRNet Method"** (PRCV 2025).


## 📦 Dataset Preparation

Download the **[DocHighlight dataset](https://github.com/SCUT-DLVCLab/DocHighlight)** from the official release and place it in a directory of your choice.

---

## ⚙️ Getting Started

### 1. Environment Setup

```bash
conda create -n docshrnet 
conda activate docshrnet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pyiqa tqdm matplotlib
```

### 2. Pre-trained Models

Download the pre-trained model checkpoints from the official release [here](https://drive.usercontent.google.com/download?id=1St86y5F_ltZXWGjIhjBfUkB81VsgRBvc&export=download&authuser=0&confirm=t&uuid=9abb9bde-07c3-4aa7-b702-ece4dcb37bbc&at=AKSUxGNjddARw4_pdwdVr9JO1hcU:1759913983487).

### 3. Inference

To restore a single image or a directory of images:
```bash
python infer.py \
  --input_dir /path/to/dataset/test/highlight \
  --checkpoint ./checkpoints/docshrnet.pth \
  --output_dir ./results
```
- Restored images are saved as `<name>_result.png` under `--output_dir`.

For large images, you can use tiled prediction with `window_pred.py`:
```bash
python window_pred.py \
  --input_dir /path/to/dataset/test/highlight \
  --checkpoint ./checkpoints/docshrnet.pth \
  --output_dir ./window_results
```

## 📊 Evaluation

To evaluate the model performance on the test set, run `evaluate.py`. This script computes PSNR and SSIM metrics between the restored images and the ground truth.

```bash
python evaluate.py \
  --pred_dir ./results \
  --gt_dir /path/to/dataset/test/highlight_free
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
- Checkpoints, logs, and sample outputs land inside `output_dir`.

---

## 📚 Citation

If this project is useful in your research or product, please cite our paper.




