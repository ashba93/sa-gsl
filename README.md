# SA-GSL
Official code for [Do We Need Deep Models or Just Better Graphs? Scalable Link Prediction through Simulated Annealing in Graph Structure Learning](). SA-GSL is an algorithm for **Graph Structure Learning(GSL)**. GSL is a family of data-centric learning approaches which jointly optimize the graph structure and the corresponding GNN models.

## Installation
<!--
[PyTorch](https://pytorch.org/get-started/previous-versions/)
[PyTorch Geometric, PyTorch Sparse](https://data.pyg.org/whl/)
[DEEP GRAPH LIBRARY (DGL)](https://data.dgl.ai/wheels/repo.html)
-->
**Note:** Based on OpenGSL, it depends on [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [DEEP GRAPH LIBRARY (DGL)](https://www.dgl.ai/pages/start.html). To streamline the installation, it does **NOT** install these libraries for you. Please install them from the above links for running SA-GSL:

- torch>=1.9.1
- torch_geometric>=2.5.0
- torch_sparse>=0.6.12
- torch_scatter>=2.0.9
- dgl>=0.9.0

**Installation for local development:**
``` bash
git clone https://github.com/ashba93/sa-gsl
cd sa-gsl
pip install -r requirements.txt
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
- optuna


## 🚀Quick Start

You can use the command `python opt_sagsl.py --dataset cora`.

## Citation

```bibtex
HERE
```
