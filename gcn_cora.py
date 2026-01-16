from opengsl.data.dataset import Dataset
import torch
from opengsl.module.encoder import GNNEncoder_OpenGSL, GCNDiagEncoder
from opengsl.module import GraphLearner
from opengsl.module.transform import KNN
from opengsl.module.metric import Cosine
from opengsl.module.fuse import Interpolate
from opengsl.utils import set_seed
import time
from copy import deepcopy
from opengsl.module.functional import normalize
from opengsl.utils import accuracy
import torch.nn.functional as F
from opengsl.module.solver import GSLSolver
from opengsl import ExpManager
import argparse


dataset = Dataset("cora", n_splits=1)
train_mask = dataset.train_masks[0]
val_mask = dataset.val_masks[0]
test_mask = dataset.test_masks[0]


device = torch.device('cpu')
set_seed(42)
encoder = GCNDiagEncoder(2, dataset.dim_feats)
metric = Cosine()
postprocess = [KNN(150)]
fuse = Interpolate(1, 1)
# build the graphlearner
graphlearner = GraphLearner(encoder=encoder, metric=metric, postprocess=postprocess, fuse=fuse).to(device)
# define gnn model
gnn = GNNEncoder_OpenGSL(dataset.dim_feats, n_hidden=64, n_class=dataset.n_classes, n_layers=2, dropout=0.5).to(device)

n_epochs = 100
lr = 1e-2
wd = 5e-4
best_valid = 0
gsl_weights =None
gnn_weights =None
start_time = time.time()
optim = torch.optim.Adam([{'params': gnn.parameters()}, {'params': graphlearner.parameters()}], lr=lr, weight_decay=wd)

for epoch in range(n_epochs):
    improve = ''
    t0 = time.time()
    gnn.train()
    graphlearner.train()
    optim.zero_grad()

    # forward and backward
    adj = graphlearner(dataset.feats, normalize(dataset.adj))
    output = gnn(dataset.feats, normalize(adj, add_loop=False))

    loss_train = F.cross_entropy(output[train_mask], dataset.labels[train_mask])
    acc_train = accuracy(dataset.labels[train_mask].cpu().numpy(), output[train_mask].detach().cpu().numpy())
    loss_train.backward()
    optim.step()

    # Evaluate
    gnn.eval()
    graphlearner.eval()
    with torch.no_grad():
        adj = graphlearner(dataset.feats, normalize(dataset.adj))
        output = gnn(dataset.feats, normalize(adj, add_loop=False))
        loss_val = F.cross_entropy(output[val_mask], dataset.labels[val_mask])
        acc_val = accuracy(dataset.labels[val_mask].cpu().numpy(), output[val_mask].detach().cpu().numpy())

    # save
    if acc_val > best_valid:
        improve = '*'
        gsl_weights = deepcopy(graphlearner.state_dict())
        gnn_weights = deepcopy(gnn.state_dict())
        total_time = time.time() - start_time
        best_val_loss = loss_val
        best_valid = acc_val
        best_adj = adj.detach().clone()

    # debug
    print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
        epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

print('Optimization Finished!')
print('Time(s): {:.4f}'.format(total_time))
# test
graphlearner.load_state_dict(gsl_weights)
gnn.load_state_dict(gnn_weights)
with torch.no_grad():
    adj = graphlearner(dataset.feats, normalize(dataset.adj))
    output = gnn(dataset.feats, normalize(adj, add_loop=False))
    loss_test = F.cross_entropy(output[test_mask], dataset.labels[test_mask])
    acc_test = accuracy(dataset.labels[test_mask].cpu().numpy(), output[test_mask].detach().cpu().numpy())

print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))


class MyGSL(GSLSolver):
    def set_method(self):
        encoder = GCNDiagEncoder(2, dataset.dim_feats)
        metric = Cosine()
        postprocess = [KNN(150)]
        fuse = Interpolate(1, 1)
        # build the graphlearner
        self.graphlearner = GraphLearner(encoder=encoder, metric=metric, postprocess=postprocess, fuse=fuse).to(device)
        # define gnn model
        self.model = GNNEncoder_OpenGSL(dataset.dim_feats, n_hidden=64, n_class=dataset.n_classes, n_layers=2, dropout=0.5).to(device)
        self.optim = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.graphlearner.parameters()}], lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

conf = {'use_deterministic': False,
    'model': {'n_hidden': 64, 'n_layer': 2},
    'training': {'lr': 1e-2,
    'weight_decay': 5e-4,
    'n_epochs': 100,
    'patience': None,
    'criterion': 'metric'},
    'dataset': {'feat_norm': False, 'sparse': True},
    'analysis': {'flag': False, 'save_graph': False}}
mygsl = MyGSL(argparse.Namespace(**conf), dataset)
exp = ExpManager(solver=mygsl)
exp.run(n_runs=3, debug=True)