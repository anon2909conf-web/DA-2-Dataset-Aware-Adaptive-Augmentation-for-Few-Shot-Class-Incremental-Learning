<div align="center">

# 🌗 DA²
### Dataset-Aware Adaptive Augmentation
#### for Few-Shot Class-Incremental Learning

<br/>

<img src="https://img.shields.io/badge/BMVC-2026-0057B7?style=for-the-badge" alt="BMVC 2026"/>
<img src="https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Task-FSCIL-8A2BE2?style=for-the-badge" alt="FSCIL"/>
<img src="https://img.shields.io/badge/License-MIT-3CB371?style=for-the-badge" alt="License"/>

<br/><br/>

**⭐ +8.09% on miniImageNet· +5.58% on CIFAR-100 ⭐**
<br/>
*over the previous best-performing methods*

<br/>

▸ [**About**](#-about) · [**Method**](#-method) · [**Results**](#-results) · [**Installation**](#-installation) · [**Datasets**](#-datasets) · [**Building Datasets**](#-building-the-mixed-domain-datasets) · [**Training**](#-training) · [**Citation**](#-citation) ◂

</div>

<br/>

---

## 📖 About

This repository contains the **official PyTorch implementation** of our BMVC 2026 paper:

> **DA²: Dataset-Aware Adaptive Augmentation for Few-Shot Class-Incremental Learning**

📄 **Paper:** *To be released soon.*

> [!TIP]
> **TL;DR** — DA² looks at how tightly a dataset's base-class prototypes are clustered, and picks the right augmentation strategy automatically. No manual tuning per dataset, negligible overhead, consistent gains everywhere we tested it.

DA² introduces a **dataset-aware feature augmentation strategy** that automatically adapts the augmentation regime to the geometry of the underlying feature space. Instead of applying the same augmentation strategy to every dataset, DA² first analyzes the **inter-class structure of base-class prototypes** and selects an appropriate augmentation regime *before* training even begins.

<br/>

## 📝 Abstract

<details open>
<summary><strong>Click to expand / collapse</strong></summary>
<br/>

Few-shot class-incremental learning (FSCIL) aims to enable models to continually learn new classes from limited data while retaining performance on previously learned ones. Existing approaches typically freeze the feature extractor during incremental updates, but do not explicitly ensure that base-class representations remain sufficiently separable. As a result, features of newly introduced classes may overlap with those of earlier classes.

We propose **Dataset-Aware Adaptive Augmentation (DA²)**, a feature-level augmentation strategy that automatically adapts the augmentation regime according to the geometry of the dataset feature space. DA² first computes the mean inter-class cosine similarity of base-class prototypes in a single forward pass before training and uses this statistic to select the augmentation strategy.

For **coarse-grained datasets**, where class prototypes are densely clustered, DA² applies a full multi-domain augmentation pipeline. For **fine-grained datasets**, where prototypes are already well separated and discriminability lies in subtle spectral structure, DA² uses a spatial-only regime to preserve these cues.

This mechanism requires no manual dataset annotation and introduces negligible computational overhead. DA² strengthens and diversifies base-class feature distributions before incremental learning, improving the accommodation of novel classes. Complementary transformations across multiple representation domains generate proxy features that increase intra-class diversity while preserving the underlying discriminative structure.

Experiments on **miniImageNet** and **CIFAR-100** demonstrate strong improvements over existing approaches, achieving average accuracy gains of **+8.09%** and **+5.58%**, respectively, over the previous best-performing methods.

</details>

<br/>

---

## 🧬 Method

### Dataset-Aware Adaptive Augmentation

DA² follows one simple principle:

> 💡 **The augmentation strategy should adapt to the geometry of the dataset.**

Before base-class training, DA² analyzes the feature-space structure of the dataset using the **mean inter-class cosine similarity** of base-class prototypes. The resulting statistic determines the augmentation regime:

<div align="center">

| 🔍 Dataset Geometry | 🧩 Feature Structure              | ⚙️ DA² Regime                       |
| :------------------: | :--------------------------------- | :----------------------------------- |
| **Coarse-grained**    | Dense / overlapping prototypes      | 🟣 **Full multi-domain augmentation** |
| **Fine-grained**      | Well-separated prototypes           | 🔵 **Spatial-only augmentation**       |

</div>

This allows DA² to automatically select an augmentation strategy **without any dataset-specific manual configuration.**

<br/>

### 🏗️ Architecture

The complete learning workflow is illustrated below.

<p align="center">
  <img src="bb_nn (2).png" width="95%" alt="DA² Architecture">
</p>

<br/>

### ✨ Key Idea

DA² operates in the **feature space** and generates complementary proxy representations through transformations across multiple representation domains. The resulting features are designed to:

- 📈 increase **intra-class diversity**
- 🎯 preserve **inter-class discriminability**
- 🔽 reduce representation overlap
- 🆕 improve the accommodation of **novel classes**
- 🧱 strengthen the base representation before incremental learning

<br/>

---

## 📊 Results

### CIFAR-100

Performance comparison on **CIFAR-100**. **Average Acc.** is the mean accuracy across all incremental sessions; **ΔFI** is the improvement over the fine-tuning baseline in the final session.

<details open>
<summary><strong>📋 Full session-by-session comparison (click to collapse)</strong></summary>
<br/>

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
| **DA² (Ours)** | **95.02** | **88.11** | **83.64** | **78.36** | **74.74** | **71.14** | **68.34** | **64.82** | **62.32** |    **76.28** | **+59.67** |

</details>

<br/>

### 🏆 Summary

<div align="center">

| Dataset          |     DA² Gain over Previous Best     |
| :---------------- | :-----------------------------------: |
| 🖼️ **CIFAR-100**    | ![+5.58%](https://img.shields.io/badge/-%2B8.09%25-3CB371?style=flat-square) |
| 🌄 **miniImageNet** | ![+8.09%](https://img.shields.io/badge/-%2B6.10%25-3CB371?style=flat-square) |

</div>
<br/>

---

## ⚡ Installation

**1. Clone the repository**

```bash
git clone https://github.com/anon2909conf-web/DA-2-Dataset-Aware-Adaptive-Augmentation-for-Few-Shot-Class-Incremental-Learning.git
cd DA-2-Dataset-Aware-Adaptive-Augmentation-for-Few-Shot-Class-Incremental-Learning
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> [!TIP]
> We recommend using a dedicated **Conda** or **virtual environment**.

<br/>

---

## 🗂️ Datasets

We evaluate DA² on **three standard FSCIL benchmarks**, plus **four additional mixed-domain / coarse-grained benchmarks** built directly from them.

### 📦 Standard benchmarks

| Dataset | How to get it |
| :------- | :-------------- |
| 🌫️ **CIFAR-100** | Downloaded automatically by the training pipeline |
| 🐦 **CUB-200** | Manual download — [**link below**](#-downloads) |
| 🌄 **miniImageNet** | Manual download — [**link below**](#-downloads) |

<h4 align="center" id="-downloads">📦 Downloads</h4>

<div align="center">

[![Download miniImageNet](https://img.shields.io/badge/⬇️_Download-miniImageNet-4285F4?style=for-the-badge)](https://drive.google.com/file/d/1p2mQ4ruI3ux8ffyA00cR0Q-k2a8XaY0E/view?usp=sharing)
[![Download CUB-200](https://img.shields.io/badge/⬇️_Download-CUB--200-4285F4?style=for-the-badge)](https://drive.google.com/file/d/1rTkDS_ERCcKV9Sr1qWPIbVUWBT9PHzz6/view?usp=sharing)

</div>

<br/>

After downloading, extract the datasets into:

```text
dataset/
├── cub200/
└── mini_imagenet/
```

For dataset organization and FSCIL settings, we follow the protocol used by [**CEC**](https://github.com/icoz69/CEC-CVPR2021).

```bash
tar -xvf miniimagenet.tar
tar -xvzf CUB_200_2011.tgz
```

<br/>

### 🌸 Oxford 102 Flowers *(for `mix120flowers`)*

`mix120flowers` additionally requires the **Oxford 102 Flowers** dataset. `build_mix120flowers.py` reads it in its *raw* form (not resplit or reorganized) and builds its own train/test split internally, so you only need the images and the label file — **no split files required**.

<div align="center">

[![Flower images](https://img.shields.io/badge/⬇️_Download-102flowers.tgz-FF69B4?style=for-the-badge)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
[![Image labels](https://img.shields.io/badge/⬇️_Download-imagelabels.mat-FF69B4?style=for-the-badge)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)

</div>

<br/>

Extract and organize the files into `dataset/flowers102_raw/` (this exact folder name is what `build_mix120flowers.py` looks for by default):

```text
dataset/
└── flowers102_raw/
    ├── jpg/                 # all images, flat — extracted from 102flowers.tgz
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   └── ...
    └── imagelabels.mat      # per-image class labels
```

```bash
mkdir -p dataset/flowers102_raw
tar -xvzf 102flowers.tgz -C dataset/flowers102_raw     # extracts into dataset/flowers102_raw/jpg/
mv imagelabels.mat dataset/flowers102_raw/
```

> [!IMPORTANT]
> `jpg/` is a **flat** folder (all images directly inside it, not split into per-class subfolders). `build_mix120flowers.py` reads `imagelabels.mat` to figure out each image's class and pools images per class itself, holding out 15 images/class for test by default (`--flowers-test-per-class`). Oxford's own official train/val/test split (`setid.mat`) is **not** used — it only gives 10 train images per class, too few for many-shot base classes — so you don't need to download `setid.mat` at all.

<br/>

### 🧬 Mixed-domain and other benchmarks

In addition to the three standard benchmarks, we introduce four datasets that test DA²'s dataset-awareness under domain mixing and at a coarser granularity:

<div align="center">

| Dataset             | Composition                                            | Build script                |
| :------------------- | :------------------------------------------------------- | :---------------------------- |
| 🔀 `mix120`           | 100 MiniImageNet + 20 CUB-200 classes                      | `build_mix120.py`            |
| 🌸 `mix120flowers`     | 100 MiniImageNet + 20 Oxford-102-Flowers classes             | `build_mix120flowers.py`     |
| 🐦 `mix200`             | 160 CUB-200 + 40 MiniImageNet classes *(CUB-dominant)*         | `build_mix200.py`            |
| 🌐 `timgnet200`          | 200 Tiny-ImageNet-200 classes                                    | `build_timgnet200.py`        |

</div>

These are **not** downloaded directly — they are constructed from the standard datasets already present under `dataset/`. See [**Building the Mixed-Domain Datasets**](#-building-the-mixed-domain-datasets) below.

<br/>

---

## 🏗️ Building the Mixed-Domain Datasets

Once `dataset/cub200/` and `dataset/mini_imagenet/` are populated (see [Datasets](#-datasets)), build the additional benchmarks with the corresponding `build_*.py` script from the repository root. Each script reads from `dataset/` and writes a new dataset folder alongside the existing ones.

```bash
# 🔀 Build mix120 (MiniImageNet + CUB-200)
python build_mix120.py

# 🌸 Build mix120flowers (MiniImageNet + Oxford-102-Flowers)
python build_mix120flowers.py \
    --root dataset \
    --out-root dataset/mix120flowers \
    --flowers-total 20 --flowers-in-base 8 \
    --sessions 12 --way 5 --shot 5 --seed 1

# 🐦 Build mix200 (CUB-dominant mix of CUB-200 + MiniImageNet)
python build_mix200.py

# 🌐 Build timgnet200 (Tiny-ImageNet-200)
python build_timgnet200.py
```

> [!TIP]
> Run each script with `--help` to see the full list of configurable options (e.g. source/output paths, class split ratios). Defaults assume the standard `dataset/` layout described above.

> [!WARNING]
> **`mix120flowers` requires one extra manual step.** With the default `--sessions 12`, the script writes `session_1.txt` (base) through `session_13.txt` (12 incremental sessions), so `dataloader/data_utils.py`'s `set_up_datasets()` must have `args.sessions = 13` for the `'mix120flowers'` dataset key. The build script prints this exact value at the end of its run — `=> set args.sessions = 13 for 'mix120flowers' ...` — so you don't have to compute it by hand.

After building, your `dataset/` directory should look like:

```text
dataset/
├── cifar100/
├── cub200/
├── mini_imagenet/
├── flowers102_raw/
├── mix120/
├── mix120flowers/
├── mix200/
└── timgnet200/
```

📎 Class attribute metadata used by the fine-grained branch (e.g. for CUB-200/Flowers-based mixes) is read from `attributes.txt` at the repository root — no separate setup is needed for it.

<br/>

---

## 🚀 Training

Once the required dataset(s) are downloaded and, if applicable, built (see above), training is launched the same way for every dataset via `train.py`, switching only the `-dataset` flag and its associated hyperparameters.

<details>
<summary><strong>🌫️ CIFAR-100</strong></summary>
<br/>

```bash
python train.py \
    -project DA \
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

</details>

<details>
<summary><strong>🐦 CUB-200</strong></summary>
<br/>

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

</details>

<details>
<summary><strong>🌄 miniImageNet</strong></summary>
<br/>

```bash
python train.py \
    -project DA \
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

</details>

<br/>

### 🧬 Mixed-domain and other benchmarks

These datasets share the same ResNet-18, non-pretrained encoder and training schedule as miniImageNet — only `-dataset` and `-epochs_base`/session count change to match each benchmark's protocol. Build the dataset first (see [**Building the Mixed-Domain Datasets**](#-building-the-mixed-domain-datasets)), then run:

<details>
<summary><strong>🔀 mix120</strong> — MiniImageNet + CUB-200, 12 incremental sessions</summary>
<br/>

```bash
python train.py \
    -project DA \
    -dataset mix120 \
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

</details>

<details>
<summary><strong>🌸 mix120flowers</strong> — MiniImageNet + Oxford-102-Flowers, 12 incremental sessions</summary>
<br/>

```bash
python train.py \
    -project DA \
    -dataset mix120flowers \
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

</details>

<details>
<summary><strong>🐦 mix200</strong> — CUB-dominant mix, 10 incremental sessions (mirrors CUB-200 protocol)</summary>
<br/>

```bash
python train.py \
    -project DA \
    -dataset mix200 \
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

</details>

<details>
<summary><strong>🌐 timgnet200</strong> — Tiny-ImageNet-200, 10 incremental sessions (mirrors CUB-200 protocol)</summary>
<br/>

```bash
python train.py \
    -project DA \
    -dataset timgnet200 \
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

</details>

<br/>

> [!NOTE]
> `mix120` and `mix120flowers` use the **miniImageNet-style** schedule (120 base epochs, milestones at 40/70/100, 84×50 crop sizes) since they are MiniImageNet-majority. `mix200` and `timgnet200` use the **CUB-200-style** schedule (120 base epochs, milestones at 60/80/100, 224×96 crop sizes) since they mirror CUB-200's 10-session, 10-way protocol.

<br/>

---

## 📁 Repository Structure

```text
DA-2-Dataset-Aware-Adaptive-Augmentation-for-Few-Shot-Class-Incremental-Learning/
├── 📄 attributes.txt              Class attribute metadata (fine-grained branch)
├── 📁 augmentations/               Spatial + spectral augmentation modules
├── 📁 dataloader/                   Dataset loaders for all benchmarks
├── 📁 dataset/                       Downloaded / built datasets live here
├── 📁 models/                         ResNet-18 / ResNet-20 encoders + DA² pipeline
├── 🐍 build_mix120.py                  Builds mix120 (MiniImageNet + CUB-200)
├── 🐍 build_mix120flowers.py            Builds mix120flowers (MiniImageNet + Flowers-102)
├── 🐍 build_mix200.py                    Builds mix200 (CUB-dominant mix)
├── 🐍 build_timgnet200.py                 Builds timgnet200 (Tiny-ImageNet-200)
├── 🐍 timgnet200.py                         Tiny-ImageNet-200 dataset handling
├── 🐍 losses.py                              Loss functions
├── 🐍 utils.py                                Shared utilities
├── 🐍 train.py                                 Training entry point
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

<br/>

---

## ♻️ Reproducibility

All experiments are conducted using the same FSCIL protocol and training configuration across datasets. Dataset-specific augmentation is selected automatically from the feature-space geometry of the base classes.


<br/>

---

## 📚 Citation

If you find **DA²** useful in your research, please consider citing our work and ⭐ starring this repository.

```bibtex
@inproceedings{da2_2026,
  title     = {DA²: Dataset-Aware Adaptive Augmentation for Few-Shot Class-Incremental Learning},
  author    = {Hitika Tiwari, Rajesh Bhatt, and Rashi Niyas P.},
  booktitle = {British Machine Vision Conference},
  year      = {2026}
}
```

> [!NOTE]
> 📄 Paper and full citation details will be updated upon release.

<br/>

---

## 🙏 Acknowledgements

This project builds upon prior work in few-shot class-incremental learning and feature augmentation. We thank the authors of the corresponding open-source implementations for making their research publicly available.

<br/>

---

<div align="center">

### 🌗 DA² — *Let the dataset determine the augmentation.*

<br/>

<sub>Made with 🧠 for the FSCIL community</sub>

</div>
