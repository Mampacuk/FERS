# Rebellious Student: A Complementary Learning Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection

**Authors**: Wenping Jin, Yuyang Tang, Li Zhu, Fei Guo

## Overview
This repository provides the official implementation for the training pipeline and the final fusion-based anomaly detection framework introduced in our paper [pdf](https://arxiv.org/pdf/2510.18781) :

**"Rebellious Student: A Complementary Learning Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection"**

Our method introduces a complementary learning paradigm that enhances background spectral and spatial representations through a novel rebellious student mechanism. The framework is composed of two sequential training stages, as illustrated below.

Stage 1 — Spectral Enhancement via Reverse Distillation

We employ a reverse-distillation-based pruning strategy to obtain a compact network that focuses on enhancing background spectral features.

![image](https://github.com/xjpp2016/FERS/blob/main/spe_train.png)

Stage 2 — Training the Spatial–Spectral Rebellious Student

Using the spectral enhancement network (FEN) as a teacher, we train a spatial–spectral network that intentionally learns features complementary to the teacher, rather than imitating it.

![image](https://github.com/xjpp2016/FERS/blob/main/spa_train.png)

## Abstract
A recent class of hyperspectral anomaly detection methods that can be trained once on background datasets and then universally deployed---without per-scene retraining or parameter tuning---has demonstrated remarkable efficiency and robustness. Building upon this paradigm, we focus on the integration of spectral and spatial cues and introduce a novel "Rebellious Student" framework for complementary feature learning. Unlike conventional teacher-student paradigms driven by imitation, our method intentionally trains the spatial branch to diverge from the spectral teacher, thereby learning complementary spatial patterns that the teacher fails to capture. A two-stage learning strategy is adopted: (1) a spectral enhancement network is first trained via reverse distillation to obtain robust background spectral representations; and (2) a spatial network---the rebellious student---is subsequently optimized using decorrelation losses that enforce feature orthogonality while maintaining reconstruction fidelity to avoid irrelevant noise. Once trained, the framework enhances both spectral and spatial background features, enabling parameter-free and training-free anomaly detection when paired with conventional detectors. Extensive experiments on the HAD100 benchmark show substantial improvements over several established baselines with minimal computational overhead, confirming the effectiveness and generality of the proposed complementary learning paradigm. Our code is publicly available at https://github.com/xjpp2016/FERS.


## HAD100 Dataset
HAD100: https://zhaoxuli123.github.io/HAD100/

## Obtaining Detection Results on HAD100
To obtain anomaly detection results on the HAD100 dataset, execute the following commands. You can specify the number of spectral bands to use with the --input_channel argument:

- For first 50 bands test:
```bash
python test.py --input_channel 50
```
- For first 100 bands test:
```bash
python test.py --input_channel 100
```
- For first 200 bands test:
```bash
python test.py --input_channel 200
```

## Training on HAD100
To train your own model on the HAD100 dataset, run the following command:

step 1:
```bash
cd train_FERD
python train.py
```

step 2:
```bash
python train.py
```
## Anomaly Detection Results on HAD100
The following table presents the mean AUC (mAUC) results achieved using different numbers of spectral bands:

| Bands | mAUC |
| :--: | :--: |
| The First 50 Bands | 0.9953 |
| The First 100 Bands | 0.9910 |
| The First 200 Bands | 0.9902 |


## Obtaining Detection Results on ABU
The corresponding code is located in the `FERS_ABU` directory.

Using the RX detector:
```bash
python test.py
```

Using the AE detector:
```bash
python test.py --detect AE
```

## Citation
If this repo works positively for your research, please consider citing our paper. Thanks all!

```
@misc{jin2025_FERS,
      title={Rebellious Student: A Complementary Learning Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection}, 
      author={Wenping Jin and Yuyang Tang and Li Zhu and Fei Guo},
      year={2025},
      eprint={2510.18781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.18781}, 
}
```

```
@article{jin2024_FERD,
  author={Jin, Wenping and Dang, Feng and Zhu, Li},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection}, 
  year={2024},
  doi={10.1109/LGRS.2024.3456178}
}
```