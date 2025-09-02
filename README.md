# SA-GSL
Official code for [Do We Need Deep Models or Just Better Graphs? Scalable Link Prediction through Simulated Annealing in Graph Structure Learning](). SA-GSL is an algorithm for **Graph Structure Learning(GSL)**. GSL is a family of data-centric learning approaches which jointly optimize the graph structure and the corresponding GNN models.

## Installation
<!--
[PyTorch](https://pytorch.org/get-started/previous-versions/)
[PyTorch Geometric, PyTorch Sparse](https://data.pyg.org/whl/)
[DEEP GRAPH LIBRARY (DGL)](https://data.dgl.ai/wheels/repo.html)
-->
**Note:** OpenGSL depends on [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [DEEP GRAPH LIBRARY (DGL)](https://www.dgl.ai/pages/start.html). To streamline the installation, OpenGSL does **NOT** install these libraries for you. Please install them from the above links for running OpenGSL:

- torch>=1.9.1
- torch_geometric>=2.1.0
- torch_sparse>=0.6.12
- dgl>=0.9.0

**Installation for local development:**
``` bash
git clone https://github.com/ashba93/sa-gsl
cd sa-gsl
pip install -e .
```

#### Required Dependencies:
- Python 3.8+
- ruamel.yaml
- pandas
- scipy
- scikit-learn
- pyro-api
- pyro-ppl
- numba


## 🚀Quick Start

You can use the command `python examples/gcn_cora.py` or follow the 4 steps.

The following example shows you how to run [GRCN](https://arxiv.org/abs/1911.07123) on the Cora dataset. 

## Citation

```bibtex
HERE
```
