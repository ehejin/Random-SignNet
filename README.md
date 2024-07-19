# Random-SignNet
Based on SignNet By Derek Lim*, Joshua Robinson*, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka.  
[[arXiv](https://arxiv.org/abs/2202.13013)] [[pdf](https://arxiv.org/pdf/2202.13013.pdf)]

## Experiments

`GINESignNetPyG` contains the core code for graph-level regression on ZINC.

In this directory, to run random-gnn on zinc run the following:
'''python -m train.zinc --config configs/random-zinc.yaml'''

## Implementations

PyTorch Geometric SignNet for graph prediction: in `Alchemy`.

PyTorch Geometric SignNet for graph prediction on ZINC: in `GINESignNetPyG`.

DGL SignNet for graph prediction: in `GraphPrediction`.

BasisNet for single graphs: in `LearningFilters`.

The SignNet architecture is rather simple. Here is an example of pseudo-code for SignNet, as used for graph prediction tasks with a GNN base model:  
<img src="images/pseudo-code.png" width=50%>

<img src="images/LPE_symmetries.png" width=50%>

**Coming Soon:** More experiments and implementations of our models! This repo and our paper are still a work in progress.

