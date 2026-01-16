import torch
import opengsl
from opengsl.utils import set_seed
import torch.nn.functional as F

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = torch.device('cpu')
set_seed(42)

conf = opengsl.config.load_conf(method="nodeformer", dataset="cora")

if hasattr(conf, 'device'):
    conf.device = 'cpu'

dataset = opengsl.data.Dataset("cora", n_splits=1, feat_norm=conf.dataset['feat_norm'])
solver = opengsl.method.NODEFORMERSolver(conf,dataset)
exp = opengsl.ExpManager(solver)
exp.run(n_runs = 10)