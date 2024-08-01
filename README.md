# Random-SignNet
Based on SignNet By Derek Lim*, Joshua Robinson*, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka.  
[[arXiv](https://arxiv.org/abs/2202.13013)] [[pdf](https://arxiv.org/pdf/2202.13013.pdf)]

## Experiments

`GINESignNetPyG` contains the core code for graph-level regression on ZINC.

In this directory, to run random-gnn on zinc run the following:
```
python -m train.zinc --config configs/CONFIG_NAME.yaml
```

## Implementations

PyTorch Geometric SignNet for graph prediction: in `Alchemy`.

PyTorch Geometric SignNet for graph prediction on ZINC: in `GINESignNetPyG`.

DGL SignNet for graph prediction: in `GraphPrediction`.

BasisNet for single graphs: in `LearningFilters`.
