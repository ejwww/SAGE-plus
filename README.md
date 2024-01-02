If you make use of this code in your own work, please cite our paper: Jiawei E，Yinglong Zhang, Shangying Yang, Hong Wang, Xuewwen Xia, Xing Xu. GraphSAGE++: Weighted Multi-scale GNN for Graph Representation Learning.  Neural Processing Letters, accepted

### PyTorch Geometric Datasets Loading Guide

PyTorch Geometric (PyG) is a library built on PyTorch that offers a wide range of tools for dealing with graph data. This document outlines how to obtain some common datasets - Amazon, Cora, Pubmed, Citeseer, and PPI - using PyG.

### Installation of PyTorch Geometric
Before accessing the datasets, make sure that you have PyTorch Geometric installed. You can install it using the following commands:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

Ensure that these installation commands are compatible with your PyTorch and CUDA versions.

### Datasets Acquisition
### Cora, Pubmed, Citeseer
These datasets are part of the Planetoid datasets and can be loaded as follows:
from torch_geometric.datasets import Planetoid

### Select the dataset
datasets = ["Cora", "Pubmed", "Citeseer"]

for dataset_name in datasets:
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    print(f'{dataset_name} dataset loaded. Contains {len(dataset)} graphs.')

### Amazon
The Amazon datasets can be loaded as follows:
from torch_geometric.datasets import Amazon

### Select the dataset, e.g., 'Computers' or 'Photo'
dataset_name = 'Computers'  # or 'Photo'
dataset = Amazon(root=f'./data/Amazon/{dataset_name}', name=dataset_name)
print(f'Amazon {dataset_name} dataset loaded. Contains {len(dataset)} graphs.')

### PPI
The PPI (Protein-Protein Interaction) dataset can be loaded as follows:
from torch_geometric.datasets import PPI
dataset = PPI(root='./data/PPI')
print(f'PPI dataset loaded. Contains {len(dataset)} graphs.')


