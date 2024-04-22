<h1 align="center">LSH-Based Efficient Point Transformer (HEPT)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2402.12535"><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/HEPT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
</p>

## TODO
- [ ] Put more details in the README.
- [ ] Add support for FlashAttn.
- [x] Add an example of HEPT with minimal code.

## Introduction
This study introduces a novel transformer model optimized for large-scale point cloud processing in scientific domains such as high-energy physics (HEP) and astrophysics. Addressing the limitations of graph neural networks and standard transformers, our model integrates local inductive bias and achieves near-linear complexity with hardware-friendly regular operations. One contribution of this work is the quantitative analysis of the error-complexity tradeoff of various sparsification techniques for building efficient transformers. Our findings highlight the superiority of using locality-sensitive hashing (LSH), especially OR \& AND-construction LSH, in kernel approximation for large-scale point cloud data with local inductive bias. Based on this finding, we propose LSH-based Efficient Point Transformer (**HEPT**), which combines E2LSH with OR \& AND constructions and is built upon regular computations. HEPT demonstrates remarkable performance in two critical yet time-consuming HEP tasks, significantly outperforming existing GNNs and transformers in accuracy and computational speed, marking a significant advancement in geometric deep learning and large-scale scientific data processing.

<p align="center"><img src="./data/HEPT.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em>Pipline of HEPT.</p>

## Datasets
All the datasets can be downloaded and processed automatically by running the scripts in `./src/datasets`, i.e.,
```
cd ./src/datasets
python pileup.py
python tracking.py -d tracking-6k
python tracking.py -d tracking-60k
```

## Installation

#### Environment
We are using `torch 2.0.1` and `pyg 2.4.0` with `python 3.10.0` and `cuda 11.8`.

#### Running the code
To run the code, you can use the following command:
```
python tracking_trainer.py -m hept
```

Or
```
python pileup_trainer.py -m hept
```
Configurations will be loaded from those located in `./configs/` directory.

## Reference
```bibtex
@article{miao2024hept,
  title   = {Locality-Sensitive Hashing-Based Efficient Point Transformer with Applications in High-Energy Physics},
  author  = {Miao, Siqi and Lu, Zhiyuan and Liu, Mia and Duarte, Javier and Li, Pan},
  journal = {arXiv preprint arXiv:2402.12535},
  year    = {2024}
}
