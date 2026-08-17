# DA²: Dataset-Aware Adaptive Augmentation

<p align="center">
  <strong>Dataset-Aware Adaptive Augmentation for Few-Shot Class-Incremental Learning</strong>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#method">Method</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#training">Training</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/BMVC-2026-blue" alt="BMVC 2026">
  <img src="https://img.shields.io/badge/PyTorch-Implementation-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-FSCIL-purple" alt="FSCIL">
</p>

---

## About

This repository contains the official PyTorch implementation of our **BMVC 2026** paper:

> **DA²: Dataset-Aware Adaptive Augmentation for Few-Shot Class-Incremental Learning**

📄 **Paper:** *To be released soon.*

DA² introduces a **dataset-aware feature augmentation strategy** that automatically adapts the augmentation regime to the geometry of the underlying feature space.

Instead of applying the same augmentation strategy to every dataset, DA² first analyzes the **inter-class structure of base-class prototypes** and selects an appropriate augmentation regime before training.

---

## Abstract

Few-shot class-incremental learning (FSCIL) aims to enable models to continually learn new classes from limited data while retaining performance on previously learned ones. Existing approaches typically freeze the feature extractor during incremental updates, but do not explicitly ensure that base-class representations remain sufficiently separable. As a result, features of newly introduced classes may overlap with those of earlier classes.

We propose **Dataset-Aware Adaptive Augmentation (DA²)**, a feature-level augmentation strategy that automatically adapts the augmentation regime according to the geometry of the dataset feature space. DA² first computes the mean inter-class cosine similarity of base-class prototypes in a single forward pass before training and uses this statistic to select the augmentation strategy.

For **coarse-grained datasets**, where class prototypes are densely clustered, DA² applies a full multi-domain augmentation pipeline. For **fine-grained datasets**, where prototypes are already well separated and discriminability lies in subtle spectral structure, DA² uses a spatial-only regime to preserve these cues.

This mechanism requires no manual dataset annotation and introduces negligible computational overhead. DA² strengthens and diversifies base-class feature distributions before incremental learning, improving the accommodation of novel classes. Complementary transformations across multiple representation domains generate proxy features that increase intra-class diversity while preserving the underlying discriminative structure.

Experiments on **CIFAR-100** and **miniImageNet** demonstrate strong improvements over existing approaches, achieving average accuracy gains of **+8.09%** and **+6.10%**, respectively, over the previous best-performing methods.

---

## Method

### Dataset-Aware Adaptive Augmentation

DA² follows a simple principle:

> **The augmentation strategy should adapt to the geometry of the dataset.**

Before base-class training, we analyze the feature-space structure of the dataset using the mean inter-class cosine similarity of base-class prototypes.

The resulting statistic determines the augmentation regime:

| Dataset Geometry | Feature Structure              | DA² Regime                         |
| ---------------- | ------------------------------ | ---------------------------------- |
| Coarse-grained   | Dense / overlapping prototypes | **Full multi-domain augmentation** |
| Fine-grained     | Well-separated prototypes      | **Spatial-only augmentation**      |

This allows DA² to automatically select an augmentation strategy without requiring dataset-specific manual configuration.

### Architecture

The complete learning workflow is illustrated below.

<p align="center">
  <img src="imgs/bb_nn (2).png" width="95%" alt="DA² Architecture">
</p>


### Key idea

DA² operates in the **feature space** and generates complementary proxy representations through transformations across multiple representation domains.

The resulting features are designed to:

* increase **intra-class diversity**,
* preserve **inter-class discriminability**,
* reduce representation overlap,
* improve the accommodation of **novel classes**, and
* strengthen the base representation before incremental learning.

---

## Results

### CIFAR-100

Performance comparison on **CIFAR-100**.

**Average Acc.** denotes the average accuracy across all incremental sessions.
**ΔFI** denotes the improvement over the fine-tuning baseline in the final session.

| Method         |         0 |         1 |         2 |         3 |         4 |         5 |         6 |         7 |         8 | Average Acc. |        ΔFI |
| :------------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | -----------: | ---------: |
| Finetune       |     64.10 |     39.61 |     15.37 |      9.80 |      6.67 |      3.80 |      3.70 |      3.14 |      2.65 |        16.54 |          — |
| iCaRL          |     64.10 |     53.28 |     41.69 |     34.13 |     27.93 |     25.06 |     20.41 |     15.48 |     13.73 |        32.87 |     +11.08 |
| EEIL           |     64.10 |     53.11 |     43.71 |     35.15 |     28.96 |     24.98 |     21.01 |     17.26 |     15.85 |        33.79 |     +13.20 |
| Rebalancing    |     64.10 |     53.05 |     43.96 |     36.97 |     31.61 |     26.73 |     21.23 |     16.78 |     13.54 |        34.22 |     +10.89 |
| TOPIC          |     64.10 |     55.88 |     47.07 |     45.16 |     40.11 |     36.38 |     33.96 |     31.55 |     29.37 |        42.62 |     +26.72 |
| SPPR           |     63.97 |     65.86 |     61.31 |     57.60 |     53.39 |     50.93 |     48.27 |     45.36 |     43.32 |        54.45 |     +40.67 |
| F2M            |     64.71 |     62.05 |     59.01 |     55.58 |     52.55 |     49.92 |     48.08 |     46.28 |     44.67 |        53.65 |     +42.02 |
| CEC            |     73.07 |     68.88 |     65.26 |     61.19 |     58.09 |     55.57 |     53.22 |     51.34 |     49.14 |        59.53 |     +46.49 |
| MetaFSCIL      |     74.50 |     70.10 |     66.84 |     62.77 |     59.48 |     56.52 |     54.36 |     52.56 |     49.97 |        60.79 |     +47.32 |
| FACT           |     74.60 |     72.09 |     67.56 |     63.52 |     61.38 |     58.36 |     56.28 |     54.24 |     52.64 |        62.24 |     +49.49 |
| TEEN           |     74.92 |     72.65 |     68.74 |     65.01 |     62.01 |     59.29 |     57.90 |     54.76 |     52.64 |        63.21 |     +49.99 |
| LIMIT          |     73.81 |     72.09 |     67.87 |     63.89 |     60.77 |     57.77 |     55.67 |     53.52 |     51.23 |        61.84 |     +48.58 |
| ILAR           |     77.50 |     73.20 |     70.80 |     66.70 |     64.00 |     62.10 |     60.50 |     58.70 |     56.40 |        65.54 |     +53.75 |
| ALICE          |     79.00 |     70.50 |     67.10 |     63.40 |     61.20 |     59.20 |     58.10 |     56.30 |     54.10 |        63.21 |     +51.45 |
| MICS           |     78.18 |     73.49 |     68.97 |     65.01 |     62.25 |     59.34 |     57.31 |     55.11 |     52.94 |        63.62 |     +50.29 |
| SAVC           |     78.77 |     73.31 |     69.31 |     64.93 |     61.70 |     59.25 |     57.13 |     55.19 |     53.12 |        63.63 |     +50.47 |
| SAGG           |     79.13 |     74.68 |     71.29 |     66.98 |     64.39 |     61.35 |     59.57 |     57.93 |     55.33 |        65.63 |     +52.68 |
| FACL           |     86.20 |     81.55 |     76.95 |     72.50 |     68.75 |     65.68 |     63.16 |     60.62 |     58.20 |        70.40 |     +55.55 |
| Flexi-FSCIL    |     79.54 |     76.35 |     73.89 |     70.07 |     68.51 |     66.43 |     63.17 |     61.33 |     59.78 |        68.79 |     +57.13 |
| **DA² (Ours)** | **95.02** | **88.11** | **83.64** | **78.36** | **74.74** | **71.14** | **68.34** | **64.82** | **62.32** |    **76.50** | **+59.67** |

### Summary

| Dataset          | DA² Gain over Previous Best |
| :--------------- | --------------------------: |
| **CIFAR-100**    |                  **+8.09%** |
| **miniImageNet** |                  **+6.10%** |

Results on additional datasets, including **CUB-200**, will be released alongside the paper and code update.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/anon2909conf-web/DA-2-Dataset-Aware-Adaptive-Augmentation-for-Few-Shot-Class-Incremental-Learning.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

We recommend using a dedicated Conda or virtual environment.

---

## Datasets

We evaluate DA² on three standard FSCIL benchmarks:

* **CIFAR-100**
* **CUB-200**
* **miniImageNet**

### CIFAR-100

CIFAR-100 is downloaded automatically by the training pipeline.

### CUB-200 & miniImageNet

Download the datasets from the following repository:

<p align="center">

### 📦 Datasets

* [**📦 Download miniImageNet**](https://drive.google.com/file/d/1p2mQ4ruI3ux8ffyA00cR0Q-k2a8XaY0E/view?usp=sharing)
* [**📦 Download CUB-200**](https://drive.google.com/file/d/1rTkDS_ERCcKV9Sr1qWPIbVUWBT9PHzz6/view?usp=sharing)


</p>

After downloading, extract the datasets into:

```text
dataset/
├── cub200/
└── mini_imagenet/
```

For dataset organization and FSCIL settings, we follow the protocol used by [CEC](https://github.com/icoz69/CEC-CVPR2021).

Example extraction commands:

```bash
tar -xvf miniimagenet.tar
tar -xvzf CUB_200_2011.tgz
```

---

## Training

### CIFAR-100

```bash
python train.py \
    -project facl \
    -dataset cifar100 \
    -lr_base 0.1 \
    -lr_new 0.001 \
    -epochs_base 600 \
    -schedule Cosine \
    -gpu 0 \
    -moco_dim 32 \
    -mlp \
    -moco_t 0.07 \
    -moco_m 0.995 \
    -size_crops 32 18 \
    -min_scale_crops 0.9 0.2 \
    -max_scale_crops 1.0 0.7 \
    -num_crops 2 4 \
    -constrained_cropping \
    -fantasy rotation2
```

### CUB-200

```bash
python train.py \
    -project facl \
    -dataset cub200 \
    -gamma 0.1 \
    -lr_base 0.002 \
    -lr_new 0.000005 \
    -epochs_base 120 \
    -schedule Milestone \
    -milestones 60 80 100 \
    -gpu 0 \
    -moco_dim 128 \
    -mlp \
    -moco_t 0.07 \
    -moco_m 0.999 \
    -size_crops 224 96 \
    -min_scale_crops 0.2 0.05 \
    -max_scale_crops 1.0 0.14 \
    -num_crops 2 4 \
    -constrained_cropping \
    -fantasy rotation2
```

### miniImageNet

```bash
python train.py \
    -project facl \
    -dataset mini_imagenet \
    -gamma 0.1 \
    -lr_base 0.1 \
    -lr_new 0.1 \
    -epochs_base 120 \
    -schedule Milestone \
    -milestones 40 70 100 \
    -gpu 0 \
    -moco_dim 128 \
    -mlp \
    -moco_t 0.07 \
    -moco_m 0.999 \
    -size_crops 84 50 \
    -min_scale_crops 0.2 0.05 \
    -max_scale_crops 1.0 0.14 \
    -num_crops 2 4 \
    -constrained_cropping \
    -fantasy rotation2
```

---

## Repository Structure

```text
DA-2/
├── dataloader/
│   ├── cifar100/
│   ├── cub200/
│   └── miniimagenet/
│
├── models/
│   ├── facl/
│   ├── resnet18_encoder.py
│   └── resnet20_cifar.py
│
├── dataset/
│
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Reproducibility

All experiments are conducted using the same FSCIL protocol and training configuration across datasets. Dataset-specific augmentation is selected automatically from the feature-space geometry of the base classes.

We will release additional configuration files, pretrained models, and evaluation details with the final project release.

---

## Citation

If you find **DA²** useful in your research, please consider citing our work and starring this repository.

```bibtex
@inproceedings{da2_2026,
  title     = {DA²: Dataset-Aware Adaptive Augmentation for Few-Shot Class-Incremental Learning},
  author    = {Hitika Tiwari, Rajesh Bhatt, and Rashi Niyas P.},
  booktitle = {British Machine Vision Conference},
  year      = {2026}
}
```

> 📄 **Paper and full citation details will be updated upon release.**

---

## Acknowledgements

This project builds upon prior work in few-shot class-incremental learning and feature augmentation. We thank the authors of the corresponding open-source implementations for making their research publicly available.

---

<p align="center">
  <strong>DA² — Let the dataset determine the augmentation.</strong>
</p>
